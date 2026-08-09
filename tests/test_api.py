"""The HTTP contract, with the pipeline stubbed (no API calls).

Three things are worth pinning. The token gate, because the paid routes being shut
by default is a spend decision and a silent regression there is an open tab on the
API key. The response shape, because it is a contract other code depends on and it
is declared separately from the pipeline's internal dicts precisely so refactors
cannot quietly change it. And that a partial scan reports itself as partial, since
a caller that cannot tell 60-of-270 from 60-of-60 has been handed a wrong answer
rather than a short one.
"""

import pytest
from fastapi.testclient import TestClient

import leasehound.metrics as metrics
import leasehound.scan as scan
from leasehound.api import TOKEN_ENV, api
from leasehound.retrieval import Result


def gate_returns(kind: str = "lease_agreement", state: str = "wa"):
    """A stand-in for `classify_document`, which returns a model rather than a kind.

    Jurisdiction defaults to "wa" — the state every scan in these tests applies — so
    a test that says nothing about jurisdiction gets no mismatch warning.
    """
    return lambda clauses, meter=None: scan.DocumentCheck(kind=kind, jurisdiction=state)


TOKEN = "test-token"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    monkeypatch.setattr(metrics, "LOG_PATH", tmp_path / "scan_metrics.jsonl")
    return TestClient(api)


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Everything that would cost money, replaced. Patched on `scan` alone, which
    is only possible because there is one orchestration for all three callers."""
    def one_clause(clause, index, config, meter=None):
        return {"index": index, "clause": clause, "verdict": "red",
                "citations": ["RCW 59.18.230"],
                "urls": {"RCW 59.18.230": "https://example.test/230"},
                "explanation": "Void under the prohibited-provisions section."}

    monkeypatch.setattr(scan, "classify_document", gate_returns())
    monkeypatch.setattr(scan, "scan_clause", one_clause)
    monkeypatch.setattr(scan, "check_protections", lambda clauses, meter=None: [
        {"name": "Deposit location", "status": "missing", "requirement": "Name the bank.",
         "citation": "RCW 59.18.270", "evidence": ""}])


def lease(clause_count: int) -> bytes:
    filler = "The parties agree to the terms set forth in this provision as written. "
    text = "\n\n".join(f"{i}. CLAUSE HEADING. {filler}" for i in range(1, clause_count + 1))
    return text.encode("utf-8")


def post_scan(client, content: bytes, name: str = "lease.md", token: str | None = TOKEN):
    headers = {"X-API-Token": token} if token else {}
    return client.post("/v1/scan", files={"file": (name, content, "text/markdown")},
                       headers=headers)


def test_health_needs_no_token_and_says_which_law_it_knows():
    body = TestClient(api).get("/v1/health").json()
    assert body["status"] == "ok"
    assert body["corpus_snapshot"] == scan.CORPUS_SNAPSHOT
    assert body["collection"] == "wa_reference"


def test_paid_routes_are_shut_when_no_token_is_configured(monkeypatch, tmp_path):
    """The hosted demo leaves the variable unset on purpose: nothing rate-limits an
    unauthenticated caller, so an open scan route is unmetered spend. 503, not 200."""
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    response = post_scan(TestClient(api), lease(2), token=None)
    assert response.status_code == 503
    assert TOKEN_ENV in response.json()["detail"]


def test_a_wrong_token_is_rejected(client):
    assert post_scan(client, lease(2), token="not-the-token").status_code == 401
    assert post_scan(client, lease(2), token=None).status_code == 401


def test_a_scan_returns_the_verdicts_the_cost_and_the_report(client, stub_pipeline):
    body = post_scan(client, lease(3), name="my_lease.md").json()
    summary = body["summary"]
    assert summary["source"] == "my_lease.md"
    assert summary["clauses_total"] == summary["clauses_judged"] == 3
    assert summary["partial"] is False
    assert summary["red"] == 3
    assert summary["missing_protections"] == 1
    # What the request spent travels with the answer, so a caller can meter itself.
    assert summary["cost_usd"] is not None and summary["seconds"] is not None
    assert len(body["findings"]) == 3
    assert body["findings"][0]["citations"] == ["RCW 59.18.230"]
    assert body["protections"][0]["name"] == "Deposit location"
    # The same report the UI pins, not a second rendering that could disagree.
    assert "LeaseHound scan report" in body["report_markdown"]


def test_the_summary_reports_the_document_state_beside_the_applied_one(
        client, stub_pipeline, monkeypatch):
    """Two fields, because the whole point is that they can disagree. A caller with
    only `state` cannot tell a Washington lease from a California one judged against
    Washington law, and those are not the same answer."""
    monkeypatch.setattr(scan, "classify_document", gate_returns(state="ca"))
    body = post_scan(client, lease(3)).json()
    assert body["summary"]["state"] == "wa", "the law that was applied"
    assert body["summary"]["jurisdiction"] == "ca", "the law the document points to"
    assert "🌎" in body["report_markdown"]

    monkeypatch.setattr(scan, "classify_document", gate_returns())
    agreed = post_scan(client, lease(3)).json()
    assert agreed["summary"]["jurisdiction"] == "wa"
    assert "🌎" not in agreed["report_markdown"]


def test_an_unrelated_document_is_refused_and_says_so_in_the_summary(
        client, stub_pipeline, monkeypatch):
    """Zeroes are the trap. A caller reading red/yellow/green without `refused`
    would see 0/0/0 and conclude the document is clean, when nothing was read."""
    monkeypatch.setattr(scan, "classify_document", gate_returns("other"))
    summary = post_scan(client, lease(3)).json()["summary"]
    assert summary["refused"] is True
    assert summary["clauses_judged"] == 0
    assert summary["red"] == summary["yellow"] == summary["green"] == 0
    body = post_scan(client, lease(3)).json()
    assert "Not scanned" in body["report_markdown"]
    assert "0 red flags" not in body["report_markdown"]

    # The override is a query parameter, and it has to actually reach the scan.
    overridden = client.post("/v1/scan?scan_anyway=true",
                             files={"file": ("lease.md", lease(3), "text/markdown")},
                             headers={"X-API-Token": TOKEN}).json()["summary"]
    assert overridden["refused"] is False
    assert overridden["clauses_judged"] == 3


def test_a_scan_over_the_cap_says_it_is_partial(client, stub_pipeline):
    total = scan.MAX_CLAUSES + 12
    summary = post_scan(client, lease(total)).json()["summary"]
    assert summary["clauses_total"] == total
    assert summary["clauses_judged"] == scan.MAX_CLAUSES
    assert summary["partial"] is True, "a caller must be able to tell 60-of-72 from 60-of-60"


def test_a_document_with_no_text_layer_is_422(client, stub_pipeline):
    """Extraction worked and produced nothing to scan — the scanned-photo PDF case.
    Distinct from a file that cannot be parsed at all, below, and worth keeping
    distinct: one is "your document has no text", the other is "your upload is
    broken", and telling a visitor the wrong one sends them to fix the wrong thing."""
    response = post_scan(client, b"   \n\n  \t ", name="photo_of_a_lease.md")
    assert response.status_code == 422
    assert "no extractable text" in response.json()["detail"]


def test_an_unparseable_upload_is_400(client, stub_pipeline):
    response = post_scan(client, b"not a pdf at all", name="broken.pdf")
    assert response.status_code == 400
    assert "Could not read broken.pdf" in response.json()["detail"]


def test_a_document_over_the_size_limit_is_413(client, stub_pipeline):
    """413 and not 422: the upload parsed fine and the text is real — it is the size
    that is refused, and a caller retrying with a smaller document is doing the right
    thing. 422 would say the content was unprocessable, which sends an integrator
    looking for a malformed file that isn't there."""
    oversized = b"1. RENT. Tenant shall pay rent.\n\n" * 20_000
    assert len(oversized) > scan.MAX_DOCUMENT_CHARS
    response = post_scan(client, oversized, name="every_document_i_own.md")
    assert response.status_code == 413
    detail = response.json()["detail"]
    # Both numbers, because "too large" without the limit gives a caller nothing to
    # act on, and this endpoint's audience is a program.
    assert f"{scan.MAX_DOCUMENT_CHARS:,}" in detail
    assert "nothing was spent" in detail


def test_the_openapi_schema_describes_both_surfaces():
    """The schema is the artifact a reader can check without running anything."""
    paths = TestClient(api).get("/openapi.json").json()["paths"]
    assert {"/v1/health", "/v1/scan", "/v1/ask"} <= set(paths)


@pytest.fixture
def stub_answer(monkeypatch):
    """Ask mode with the models replaced, returning a fixed answer and its cost."""
    import leasehound.answer as answer

    def answered(question, history=None, config=None, report_context=False):
        result = answer.AskResult(
            stream=iter(()), chunks=[Result(page_content="Waivers are void.",
                                            metadata={"section": "RCW 59.18.230"})],
            meter=metrics.UsageMeter(), routed=True)

        def stream():
            yield "Waivers of that right are void "
            yield "under RCW 59.18.230."
            result.record = {"llm_calls": 5, "cost_usd": 0.0041, "seconds": 6.2}

        result.stream = stream()
        return result

    monkeypatch.setattr("leasehound.api.answer_question", answered)


def test_an_answer_comes_back_with_its_sources_and_what_it_spent(client, stub_answer):
    """/v1/ask returns cost for the same reason /v1/scan does: a caller that cannot
    see what a request spent cannot meter itself, and ask mode is the mode whose
    price this project left unmeasured longest."""
    body = client.post("/v1/ask", json={"question": "can they waive my rights?"},
                       headers={"X-API-Token": TOKEN}).json()
    assert "RCW 59.18.230" in body["answer"]
    assert body["sources"] == ["RCW 59.18.230"]
    assert body["retrieved"] is True
    # Populated only because the route drains the stream before answering — that is
    # what completes the metering, exactly as run_scan drains the step iterator.
    assert body["llm_calls"] == 5
    assert body["cost_usd"] == 0.0041


def test_the_ask_route_is_behind_the_same_spend_gate(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    response = TestClient(api).post("/v1/ask", json={"question": "hi"})
    assert response.status_code == 503
