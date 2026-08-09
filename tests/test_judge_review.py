"""The human-audit tool: sampling, agreement arithmetic, and the statistic it reports.

This is the only thing in the project that checks an LLM judge against a person, so
its arithmetic has to be right for the number to be worth quoting.
"""

from evaluation.review_generation import choose_sample, interesting, kappa


def case(question: str, verdict: str = "consistent", unsupported: bool = False,
         cites_gt: bool = True, outside: list | None = None) -> dict:
    return {
        "question": question,
        "section": "RCW 59.18.230",
        "answer": "An answer.",
        "verdict": verdict,
        "unsupported_claim": unsupported,
        "cites_gt": cites_gt,
        "cites_outside_corpus": outside or [],
        "retrieval_had_gt": True,
    }


def test_kappa_matches_hand_computed_values():
    assert kappa([("c", "c"), ("c", "c"), ("x", "x"), ("x", "x")]) == 1.0
    # Two raters, two labels, agreeing exactly as often as chance predicts.
    assert kappa([("c", "c"), ("c", "x"), ("x", "c"), ("x", "x")]) == 0.0
    assert kappa([]) is None


def test_kappa_is_undefined_when_a_rater_never_varies():
    """The situation this project is actually in: the judge returned "consistent" for
    all 82 answers. Reported as None rather than 1.0 — unanimity on one label is where
    the statistic has nothing to say, and a number there would imply it did."""
    assert kappa([("c", "c")] * 20) is None


def test_agreeing_nineteen_times_out_of_twenty_still_scores_kappa_zero():
    """Why the tool leads with a binomial bound instead of kappa. Against a judge that
    only ever says one thing, near-perfect agreement is worth exactly chance — so a
    headline kappa would read as damning when it is merely inapplicable."""
    assert kappa([("c", "c")] * 19 + [("x", "c")]) == 0.0


def test_the_sample_takes_every_case_the_judge_hesitated_on_first():
    """Stratification is what stops the audit being circular. The judge waved 82 of 82
    through, so a uniform sample is 20 easy cases and a reviewer who agrees 20 times has
    demonstrated nothing. The cases where it hesitated are where disagreement lives."""
    cases = (
        [case(f"plain-{i}") for i in range(30)]
        + [case("ungrounded", unsupported=True),
           case("wrong-section", cites_gt=False),
           case("outside-corpus", outside=["RCW 19.86.020"]),
           case("declined", verdict="declined")]
    )
    sample = choose_sample(cases, size=10, seed=0)

    assert len(sample) == 10
    flagged = {c["question"] for c in sample if interesting(c)}
    assert flagged == {"ungrounded", "wrong-section", "outside-corpus", "declined"}


def test_every_flagged_case_survives_even_past_the_requested_size():
    """Asking for 3 when 4 are interesting must not drop one: the sample size is a floor
    on effort, not a cap on the cases that matter."""
    cases = [case("a", unsupported=True), case("b", cites_gt=False),
             case("c", verdict="declined"), case("d", outside=["RCW 19.86.020"]),
             case("plain")]
    sample = choose_sample(cases, size=3, seed=0)
    assert len(sample) == 4
    assert all(interesting(c) for c in sample)


def test_sampling_is_reproducible_from_the_seed():
    """A hand-audit nobody can re-derive is an anecdote. Same seed, same 20 cases."""
    cases = [case(f"q-{i}") for i in range(50)]
    first = [c["question"] for c in choose_sample(cases, 20, seed=7)]
    second = [c["question"] for c in choose_sample(cases, 20, seed=7)]
    other = [c["question"] for c in choose_sample(cases, 20, seed=8)]
    assert first == second
    assert first != other


def test_a_clean_case_is_not_flagged():
    assert not interesting(case("clean"))
    assert interesting(case("x", verdict="contradicts"))
    assert interesting(case("x", unsupported=True))
    assert interesting(case("x", cites_gt=False))
    assert interesting(case("x", outside=["RCW 19.86.020"]))
