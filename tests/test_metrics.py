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
