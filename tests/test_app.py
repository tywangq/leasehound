"""App helpers: cited-only sources footer and upload cleanup (privacy)."""

import leasehound.app as app
from leasehound.app import SAMPLE_LEASE, cleanup_upload, sources_footer
from leasehound.retrieval import Result


def chunk(section: str) -> Result:
    return Result(page_content="text", metadata={"section": section, "url": f"https://law/{section}"})


RETRIEVED = [chunk("RCW 59.18.170"), chunk("RCW 59.18.230"), chunk("RCW 59.18.060")]


def test_footer_lists_only_statutes_the_answer_cites():
    answer = "Late fees can only start on day six under RCW 59.18.170(2)."
    footer = sources_footer(answer, RETRIEVED)
    assert "RCW 59.18.170" in footer
    assert "RCW 59.18.230" not in footer
    assert "RCW 59.18.060" not in footer


def test_footer_empty_when_answer_cites_nothing():
    # An off-topic question retrieves chunks but the answer declines — no footer.
    assert sources_footer("I only cover Washington State law.", RETRIEVED) == ""


def test_footer_ignores_citations_not_in_the_retrieved_set():
    assert sources_footer("See RCW 99.99.999 for details.", RETRIEVED) == ""


def test_cleanup_removes_upload_but_never_the_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(app.tempfile, "gettempdir", lambda: str(tmp_path))
    upload = tmp_path / "lease.pdf"
    upload.write_text("uploaded")
    cleanup_upload(upload)
    assert not upload.exists()

    outside = tmp_path.parent / "keep.pdf"
    outside.write_text("keep")
    monkeypatch.setattr(app.tempfile, "gettempdir", lambda: str(tmp_path / "elsewhere"))
    cleanup_upload(outside)
    assert outside.exists()
    outside.unlink()

    cleanup_upload(SAMPLE_LEASE)
    assert SAMPLE_LEASE.exists()
