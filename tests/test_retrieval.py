"""RRF merging: the stage the ablation study caught failing as a plain append."""

from leasehound.retrieval import PipelineConfig, Result, merge_chunks


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
