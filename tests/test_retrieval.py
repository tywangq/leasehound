"""RRF merging: the stage the ablation study caught failing as a plain append."""

from types import SimpleNamespace

import leasehound.retrieval as retrieval
from leasehound.retrieval import (
    HYBRID_CANDIDATES,
    PipelineConfig,
    Result,
    candidate_k,
    fetch_unranked,
    merge_chunks,
)


def chunk(name: str) -> Result:
    return Result(page_content=name, metadata={"section": name})


def names(chunks: list[Result]) -> list[str]:
    return [c.page_content for c in chunks]


def test_chunk_present_in_both_lists_ranks_first():
    a = [chunk("shared"), chunk("a1"), chunk("a2")]
    b = [chunk("b1"), chunk("shared"), chunk("b2")]
    merged = merge_chunks(a, b)
    assert names(merged)[0] == "shared"
    assert len(merged) == 5  # deduplicated by content


def test_top_of_second_list_beats_tail_of_first():
    # The regression the ablation caught: with append-style merging, the second
    # query's chunks could never reach top-k. Under RRF, its #1 must outrank
    # the first list's tail.
    a = [chunk(f"a{i}") for i in range(10)]
    b = [chunk("b-top")]
    merged = names(merge_chunks(a, b))
    assert merged.index("b-top") < merged.index("a9")


def test_single_list_preserves_order():
    a = [chunk("x"), chunk("y"), chunk("z")]
    assert names(merge_chunks(a)) == ["x", "y", "z"]


def test_default_config_matches_full_pipeline():
    config = PipelineConfig()
    assert (config.dual_query, config.grader, config.rerank) == (True, True, True)
    assert config.final_k <= config.retrieval_k
    # Off by default, so every ablation row recorded before hybrid retrieval
    # existed still means what it measured.
    assert config.bm25 is False


def test_hybrid_looks_deeper_than_the_window_it_returns():
    # The whole point of the merge is to recover a chunk that neither channel
    # ranks in the top few. Scan mode shows the judge six chunks, so querying
    # only six per channel would discard that chunk before the merge ran.
    scan_like = PipelineConfig(bm25=True, retrieval_k=6)
    assert candidate_k(scan_like) == HYBRID_CANDIDATES > scan_like.retrieval_k
    ask_like = PipelineConfig(bm25=True, retrieval_k=40)
    assert candidate_k(ask_like) == 40


def fake_dense(sections: list[str]):
    """Stand in for Chroma: returns one row per section, honoring n_results."""
    def query(query_embeddings, n_results):
        selected = sections[:n_results]
        return {"documents": [selected], "metadatas": [[{"section": s} for s in selected]]}
    return SimpleNamespace(query=query)


def test_hybrid_merge_promotes_a_chunk_both_channels_rank_mid_list(monkeypatch):
    # The measured case, in miniature: the governing section is 5th in the dense
    # channel and 4th in the lexical one, so a top-3 window would miss it from
    # either alone — but agreement under RRF pulls it into the top 3.
    dense_order = ["far-a", "far-b", "far-c", "far-d", "governing", "far-e"]
    lexical_order = ["lex-a", "lex-b", "lex-c", "governing", "lex-d"]

    monkeypatch.setattr(retrieval, "_get_collection", lambda name: fake_dense(dense_order))
    monkeypatch.setattr(
        retrieval.openai.embeddings, "create",
        lambda model, input: SimpleNamespace(data=[SimpleNamespace(embedding=[0.0])]),
    )
    monkeypatch.setattr(
        retrieval, "bm25_search",
        lambda query, config, k=None: [
            Result(page_content=s, metadata={"section": s}) for s in lexical_order[:k]
        ],
    )
    config = PipelineConfig(bm25=True, retrieval_k=3)
    got = [c.metadata["section"] for c in fetch_unranked("a clause", config)]
    assert len(got) == 3, "the judge's window must not grow"
    assert "governing" in got

    config.bm25 = False
    dense_only = [c.metadata["section"] for c in fetch_unranked("a clause", config)]
    assert "governing" not in dense_only, "the dense channel alone must still miss it"
