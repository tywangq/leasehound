"""ScanMeter aggregation and the scan-log line format (no API calls)."""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import leasehound.metrics as metrics
from leasehound.metrics import ScanMeter, log_scan


def completion_response(prompt=100, completion=40):
    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=prompt,
                                                 completion_tokens=completion))


def embedding_response(tokens=30):
    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=tokens))


def test_meter_totals_across_threads(monkeypatch):
    # Clause judgments add from pool workers concurrently; totals must not race.
    monkeypatch.setattr(metrics, "completion_cost", lambda completion_response: 0.001)
    monkeypatch.setattr(metrics, "cost_per_token", lambda model, prompt_tokens: (0.0002, 0.0))
    meter = ScanMeter()
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
    meter = ScanMeter()
    meter.add_completion(completion_response())
    summary = meter.summary()
    assert summary["llm_calls"] == 1
    assert summary["cost_usd"] == 0.0


def test_log_scan_writes_one_json_line_with_file_name_only(monkeypatch, tmp_path):
    monkeypatch.setattr(metrics, "completion_cost", lambda completion_response: 0.001)
    monkeypatch.setattr(metrics, "LOG_PATH", tmp_path / "logs" / "scan_metrics.jsonl")
    meter = ScanMeter()
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
    record = log_scan(ScanMeter(), "lease.pdf", 5,
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
    meter = metrics.ScanMeter()
    flagged = metrics.log_scan(meter, "odd.md", 3, gate_flagged=True)
    ordinary = metrics.log_scan(meter, "lease.md", 3)
    assert flagged["gate_flagged"] is True
    assert "gate_flagged" not in ordinary
