"""Which state's law governs — the document's, or the asker's, against the corpus's.

This started inside scan.py, where the gate reads a lease and reports the state its
own words point to. Ask mode has the same hole for the same reason: the corpus is
Washington's, `state` defaults to `"wa"`, and a renter in Oregon typing "my landlord
kept my whole deposit" gets RCW 59.18 with nothing saying it does not govern them.
The scan side warns and the ask side did not, which made the fix a shared concept
rather than a scan detail — so it lives here and both modes import it.

Nothing here calls a model. The two classifications that produce a jurisdiction ride
along on calls both modes already make: scan's is-this-a-lease gate, and ask's router.
"""

from typing import Literal

# An enum rather than a free string so an answer is comparable to the state being
# applied without normalising "California" / "CA" / "Calif." by hand — a
# structured-output enum cannot come back as a value this code has no branch for.
US_STATES = (
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il",
    "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt",
    "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri",
    "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy", "dc",
)
UNKNOWN_JURISDICTION = "unknown"
# Literal takes a tuple at runtime exactly as it takes a series of literals, which is
# what keeps the 51 codes in one list instead of two that can drift.
Jurisdiction = Literal[(*US_STATES, UNKNOWN_JURISDICTION)]  # type: ignore[valid-type]


def jurisdiction_mismatch(named: str, applied: str) -> bool:
    """Whether the law being applied is not the law that was named.

    "unknown" is not a mismatch, and that is the load-bearing decision. Most short
    leases name no state anywhere and most questions never mention one, so treating
    silence as a mismatch would fire the warning almost every time — and a warning
    that fires almost every time costs the true ones their credibility, which is the
    only thing they have.
    """
    return named != UNKNOWN_JURISDICTION and named.lower() != applied.lower()
