"""UsageMeter aggregation and the scan-log line format (no API calls)."""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import leasehound.metrics as metrics
from leasehound.metrics import UsageMeter, log_scan


def completion_response(prompt=100, completion=40):
    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=prompt,
                                                 completion_tokens=completion))


def embedding_response(tokens=30):
    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=tokens))


def test_meter_totals_across_threads(monkeypatch):
    # Clause judgments add from pool workers concurrently; totals must not race.
    monkeypatch.setattr(metrics, "completion_cost", lambda completion_response: 0.001)
    monkeypatch.setattr(metrics, "cost_per_token", lambda model, prompt_tokens: (0.0002, 0.0))
    meter = UsageMeter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in range(50):
            pool.submit(meter.add_completion, completion_response())
            pool.submit(meter.add_embedding, embedding_response(), "text-embedding-3-large")
    summary = meter.summary()
    assert summary["llm_calls"] == 50
    assert summary["embedding_calls"] == 50
    assert summary["prompt_tokens"] == 5000
    assert summary["completion_tokens"] == 2000
    assert summary["embedding_tokens"] == 1500
    assert summary["cost_usd"] == round(50 * 0.001 + 50 * 0.0002, 6)


def test_unknown_model_counts_the_call_but_skips_cost(monkeypatch):
    def boom(**kwargs):
        raise ValueError("model not in price map")

    monkeypatch.setattr(metrics, "completion_cost", boom)
    meter = UsageMeter()
    meter.add_completion(completion_response())
    summary = meter.summary()
    assert summary["llm_calls"] == 1
    assert summary["cost_usd"] == 0.0


def test_log_scan_writes_one_json_line_with_file_name_only(monkeypatch, tmp_path):
    monkeypatch.setattr(metrics, "completion_cost", lambda completion_response: 0.001)
    monkeypatch.setattr(metrics, "LOG_PATH", tmp_path / "logs" / "scan_metrics.jsonl")
    meter = UsageMeter()
    meter.add_completion(completion_response())
    log_scan(meter, "/private/uploads/secret_lease.pdf", clauses=15,
             verdicts={"red": 7, "yellow": 1, "green": 7}, missing=2)

    lines = (tmp_path / "logs" / "scan_metrics.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    # Privacy: the log carries the file name, never the upload path or lease text.
    assert record["source"] == "secret_lease.pdf"
    assert record["clauses"] == 15
    assert record["verdicts"] == {"red": 7, "yellow": 1, "green": 7}
    assert record["missing_protections"] == 2
    assert record["llm_calls"] == 1
    assert record["seconds"] >= 0


def test_log_scan_records_split_mode_cache_hit_and_echoes_to_stdout(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(metrics, "LOG_PATH", tmp_path / "scan_metrics.jsonl")
    record = log_scan(UsageMeter(), "lease.pdf", 5,
                      split_mode="paragraphs", cache_hit=True)
    assert record["split_mode"] == "paragraphs"
    assert record["cache_hit"] is True
    # The JSONL file is ephemeral on Cloud Run; the same record must reach
    # stdout so Cloud Logging keeps it.
    assert json.loads(capsys.readouterr().out.strip()) == record


def test_log_scan_records_the_gate_flag(tmp_path, monkeypatch):
    # A flagged scan's verdicts are not comparable with the rest, so the log has
    # to say so — otherwise the metrics summary silently mixes them in.
    monkeypatch.setattr(metrics, "LOG_PATH", tmp_path / "scan_metrics.jsonl")
    meter = metrics.UsageMeter()
    flagged = metrics.log_scan(meter, "odd.md", 3, gate_flagged=True)
    ordinary = metrics.log_scan(meter, "lease.md", 3)
    assert flagged["gate_flagged"] is True
    assert "gate_flagged" not in ordinary


def test_the_yellow_share_is_reported_and_attributed(tmp_path, monkeypatch):
    """Nothing measured yellow. The labelled sets score red against a manifest and
    count yellow only as a hedge on a planted violation, so a scanner that cautioned
    on every clause would post zero false reds and look immaculate. This is the share
    on real traffic — with the judges it rests on, because three of them were measured
    against these sets in one afternoon and only one ships."""
    records = [
        {"clauses": 10, "cost_usd": 0.01, "seconds": 8.0, "llm_calls": 11,
         "embedding_calls": 10, "judge": "aaaa1111",
         "verdicts": {"red": 2, "yellow": 1, "green": 7}},
        {"clauses": 10, "cost_usd": 0.01, "seconds": 9.0, "llm_calls": 11,
         "embedding_calls": 10,  # written before the field existed
         "verdicts": {"red": 0, "yellow": 3, "green": 7}},
    ]
    summary = metrics.summarize_log(records)
    assert summary["verdict_clauses"] == 20
    assert summary["verdict_share"] == {"red": 0.1, "yellow": 0.2, "green": 0.7}
    assert summary["verdict_judges"] == ["aaaa1111", "unrecorded"]

    # A log with no verdicts recorded reports no share, rather than an all-green one.
    bare = metrics.summarize_log([{k: v for k, v in records[0].items()
                                   if k not in ("verdicts", "judge")}])
    assert "verdict_share" not in bare


def test_log_scan_records_an_out_of_state_lease(tmp_path, monkeypatch):
    """The published cost and latency figures are computed from this log, and a scan
    of a lease governed by another state's law is a scan whose verdicts nobody should
    be quoting — the same reason `gate_flagged` is recorded."""
    monkeypatch.setattr(metrics, "LOG_PATH", tmp_path / "scan_metrics.jsonl")
    meter = metrics.UsageMeter()
    elsewhere = metrics.log_scan(meter, "ca_lease.pdf", 3, jurisdiction="ca")
    ordinary = metrics.log_scan(meter, "lease.md", 3)
    assert elsewhere["jurisdiction"] == "ca"
    # The caller passes None when the document's state is the one being applied, so
    # this function never has to know which state that was — a `!= "wa"` here would
    # have hard-coded the default and quietly mislabelled every non-WA deployment.
    assert "jurisdiction" not in ordinary


def streamed_usage(prompt=800, completion=120, cached=0):
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion,
        prompt_tokens_details={"cached_tokens": cached} if cached else None,
    )


def test_a_streamed_call_is_booked_from_its_usage_alone(monkeypatch):
    # The reason ask mode went unmetered: a streamed answer has no response object
    # to price, only token totals on a final chunk. Both halves of the price map
    # entry apply — booking input only would undercount every answer.
    monkeypatch.setattr(metrics, "cost_per_token",
                        lambda model, prompt_tokens, completion_tokens=0:
                        (prompt_tokens * 1e-6, completion_tokens * 4e-6))
    meter = UsageMeter()
    meter.add_streamed_completion(streamed_usage(800, 120), "openai/gpt-4.1-mini")
    summary = meter.summary()
    assert summary["llm_calls"] == 1
    assert summary["prompt_tokens"] == 800
    assert summary["completion_tokens"] == 120
    assert summary["cost_usd"] == round(800e-6 + 120 * 4e-6, 6)


def test_a_streamed_call_still_reports_its_cached_input_tokens(monkeypatch):
    # Same reason the scan log carries this: without it a warm answer looks like a
    # cheaper pipeline rather than the same pipeline at a discounted rate.
    monkeypatch.setattr(metrics, "cost_per_token",
                        lambda **kwargs: (0.0, 0.0))
    meter = UsageMeter()
    meter.add_streamed_completion(streamed_usage(cached=512), "openai/gpt-4.1-mini")
    assert meter.summary()["cached_prompt_tokens"] == 512


def test_an_unpriced_model_still_counts_the_streamed_call(monkeypatch):
    def boom(**kwargs):
        raise ValueError("model not in price map")

    monkeypatch.setattr(metrics, "cost_per_token", boom)
    meter = UsageMeter()
    meter.add_streamed_completion(streamed_usage(), "openai/not-a-real-model")
    summary = meter.summary()
    assert summary["llm_calls"] == 1
    assert summary["prompt_tokens"] == 800, "the tokens are known even when the price isn't"
    assert summary["cost_usd"] == 0.0


def test_log_ask_records_the_cost_and_nothing_about_the_question(
    monkeypatch, tmp_path, capsys
):
    """The scan log names a file because a developer needs to match a row to a
    document. A question is the user's own words and there is no such need, so the
    row carries cost and shape only — this test is what keeps it that way."""
    monkeypatch.setattr(metrics, "ASK_LOG_PATH", tmp_path / "logs" / "ask_metrics.jsonl")
    monkeypatch.setattr(metrics, "cost_per_token", lambda **kwargs: (0.0, 0.0))
    meter = UsageMeter()
    meter.add_streamed_completion(streamed_usage(), "openai/gpt-4.1-mini")
    record = metrics.log_ask(meter, retrieved=10, routed=True, with_report=True)

    lines = (tmp_path / "logs" / "ask_metrics.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == record
    assert record["routed_to_retrieval"] is True
    assert record["retrieved_chunks"] == 10
    assert record["with_report_context"] is True
    assert set(record) == {"ts", "routed_to_retrieval", "retrieved_chunks",
                           "with_report_context", "llm_calls", "embedding_calls",
                           "prompt_tokens", "cached_prompt_tokens", "completion_tokens",
                           "embedding_tokens", "cost_usd", "seconds"}
    # Ephemeral filesystem on Cloud Run, same as the scan log.
    assert json.loads(capsys.readouterr().out.strip()) == record


def test_the_ask_summary_leaves_chitchat_out_of_the_cost_per_question():
    # A greeting is one call and no retrieval. Averaging those in would make the
    # six-stage pipeline look cheaper than answering a legal question costs.
    rows = [
        {"routed_to_retrieval": True, "cost_usd": 0.004, "seconds": 6.0, "llm_calls": 5,
         "embedding_calls": 2, "prompt_tokens": 8000, "cached_prompt_tokens": 2000},
        {"routed_to_retrieval": True, "cost_usd": 0.006, "seconds": 8.0, "llm_calls": 5,
         "embedding_calls": 2, "prompt_tokens": 12000, "cached_prompt_tokens": 0},
        {"routed_to_retrieval": False, "cost_usd": 0.0001, "seconds": 1.0, "llm_calls": 2,
         "embedding_calls": 0, "prompt_tokens": 300, "cached_prompt_tokens": 0},
    ]
    summary = metrics.summarize_ask_log(rows)
    assert summary["questions"] == 2
    assert summary["chitchat_excluded"] == 1
    assert summary["mean_cost_usd"] == 0.005
    assert summary["mean_llm_calls"] == 5.0
    assert summary["cached_prompt_token_share"] == 0.1


def test_the_ask_summary_survives_an_empty_log():
    assert metrics.summarize_ask_log([]) == {"questions": 0}


def test_the_scan_summary_leaves_refusals_out_of_the_cost_per_scan():
    """A refused document judged nothing — one classification call, ~$0.0002 — so it
    is not a scan at any clause count. Averaged in, a visitor uploading a recipe
    silently lowers the project's published cost-per-scan, which is the same reason
    cache hits are already excluded. The 9-15 band dropped these by luck (they log 0
    clauses); `all_scans` counted them."""
    def row(**kw):
        return {"clauses": 12, "cost_usd": 0.012, "seconds": 9.0, "llm_calls": 14,
                "embedding_calls": 12, **kw}

    rows = [
        row(),
        row(clauses=14, cost_usd=0.014, seconds=11.0, llm_calls=16, embedding_calls=14),
        row(clauses=0, cost_usd=0.0002, seconds=1.4, llm_calls=1, embedding_calls=0,
            refused=True, gate_flagged=True),
    ]
    summary = metrics.summarize_log(rows)
    assert summary["scans"] == 2
    assert summary["refusals_excluded"] == 1
    assert summary["mean_cost_usd"] == 0.013
    # And the giveaway the old code would have produced: a floor of zero clauses.
    assert summary["clauses_min"] == 12


def test_a_scan_row_records_which_surface_asked_for_it(tmp_path, monkeypatch):
    """The published cost figure is a mean over this log, and until this field existed
    the log could not say what it was a sample of: eval runs rescanning the same six
    labelled leases, manual UI clicks, the demo recorder, and deploy smoke tests were
    averaged together and quoted as a per-scan cost."""
    monkeypatch.setattr(metrics, "LOG_PATH", tmp_path / "scan_metrics.jsonl")
    log_scan(UsageMeter(), "lease.md", 12, client="ui")
    log_scan(UsageMeter(), "lease.md", 12, client="eval")
    log_scan(UsageMeter(), "lease.md", 12)  # a caller that did not say

    rows = [json.loads(line) for line in
            (tmp_path / "scan_metrics.jsonl").read_text().splitlines()]
    assert [r["client"] for r in rows] == ["ui", "eval", "unknown"]


def test_the_summary_names_the_population_it_averaged():
    """A mean without its population is a number nobody can check. Rows written before
    the tag existed count as `unknown` rather than being guessed at."""
    def row(client=None, **kw):
        record = {"clauses": 12, "seconds": 9.0, "cost_usd": 0.011,
                  "llm_calls": 14, "embedding_calls": 12, **kw}
        if client:
            record["client"] = client
        return record

    summary = metrics.summarize_log(
        [row("ui"), row("ui"), row("eval"), row("api"), row()])

    assert summary["scans"] == 5
    assert summary["by_client"] == {"api": 1, "eval": 1, "ui": 2, "unknown": 1}


def test_every_surface_tags_its_scans(tmp_path, monkeypatch):
    """The three clients and the eval harness each name themselves, so a row can be
    attributed without guessing from the file name. Checked here rather than in each
    caller's own test file because the point is that the set is COMPLETE — an untagged
    fourth entry point is exactly what would quietly pollute the mean again."""
    import leasehound.scan as scan

    monkeypatch.setattr(metrics, "LOG_PATH", tmp_path / "scan_metrics.jsonl")
    monkeypatch.setattr(scan, "classify_document",
                        lambda clauses, meter=None: scan.DocumentCheck(
                            kind="lease_agreement", jurisdiction="wa"))
    monkeypatch.setattr(scan, "scan_clauses", lambda *a, **k: iter(()))
    monkeypatch.setattr(scan, "check_protections", lambda *a, **k: [])

    for client in ("cli", "ui", "api", "eval"):
        scan.run_scan("1. RENT. Tenant shall pay rent on the first of each month.",
                      "lease.md", client=client)

    rows = [json.loads(line) for line in
            (tmp_path / "scan_metrics.jsonl").read_text().splitlines()]
    assert [r["client"] for r in rows] == ["cli", "ui", "api", "eval"]


# --- the file name is PII on a hosted surface -----------------------------------


def hosted_row(monkeypatch, tmp_path, source, client):
    monkeypatch.setattr(metrics, "completion_cost", lambda completion_response: 0.001)
    monkeypatch.setattr(metrics, "LOG_PATH", tmp_path / "scan_metrics.jsonl")
    meter = UsageMeter()
    meter.add_completion(completion_response())
    return log_scan(meter, source, clauses=15, client=client)


def test_a_visitors_file_name_does_not_reach_the_log(monkeypatch, tmp_path):
    """A lease file name is a person: real ones name the tenant or the address.

    log_scan also prints its record to stdout, which on Cloud Run is Cloud Logging —
    so before this the demo was accumulating third-party PII in the operator's own
    project, with the README offering "only the file name" as the reassuring half.
    """
    for client in ("ui", "api"):
        record = hosted_row(monkeypatch, tmp_path, "/tmp/up/SmithJohn_1425_Pine.pdf",
                            client)
        assert "SmithJohn" not in json.dumps(record)
        assert "Pine" not in json.dumps(record)
        # The extension survives: it is about the document, not its owner, and
        # split_mode is only interpretable next to the format it came from.
        assert record["source"] == "upload.pdf"


def test_a_developers_own_file_name_is_still_recorded(monkeypatch, tmp_path):
    """The point of the field is matching a row to a document, which is a real need on
    a laptop and an impossible one on a hosted demo."""
    for client in ("cli", "eval", None):
        record = hosted_row(monkeypatch, tmp_path, "leases/hud_sample.pdf", client)
        assert record["source"] == "hud_sample.pdf"
