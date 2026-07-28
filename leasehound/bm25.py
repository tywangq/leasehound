"""Lexical retrieval channel: BM25 over the same chunks as the dense index.

**Measured and not enabled anywhere by default.** Kept because the experiment
that rejected it is the useful artifact — see the README's hybrid-retrieval
section for the numbers. In short:

- Ask mode: on the original question set it helps (hit@10 .915 → .951), but that
  set inherits statute vocabulary by construction. On the renter-voice
  rephrasing — the honest set — it clearly hurts (MRR .708 → .609): a casual
  question shares only common words with the statutes, so the lexical channel
  contributes noise that RRF then weights equally.
- Scan mode: it cost two false reds on the hand-labeled leases, including one on
  the fully compliant lease, and recovered nothing. Catalog sections like
  RCW 59.18.230 (a list of prohibited clause types) are lexical magnets — they
  share vocabulary with almost any clause — and putting a catalog of illegal
  clauses in front of the judge biases it toward red.

The chain of hypotheses is worth keeping, because each one was wrong in an
instructive way. (1) "The keyword channel will catch *hold harmless*" — no: the
statute never uses that phrase, it says "waiver of rights", "indemnify",
"liable". (2) "RRF consensus will recover the section anyway" — true at the
section level: RCW 59.18.230 moves from dense rank 15 into the judge's top six.
(3) "So the verdict will change" — no: the judge still returned green, because
the *chunk* retrieved was the one about distress for rent and landlord's liens,
not the one prohibiting exculpation clauses. The section is split across four
chunks and the merge surfaced the wrong one.

So the real defect behind that miss is chunk granularity, not the retrieval
channel — and the retrieval eval could not have revealed it, because it scores a
section-level hit that a chunk with none of the governing rule still satisfies.

Written here rather than pulled in as a dependency because it is ~30 lines of
transparent scoring, it keeps the container image free of another package, and
the tokenizer needs to be ours: legal text carries citations like "59.18.230"
that must survive as ONE token, which a whitespace/word tokenizer shreds into
three numbers.

Deterministic and API-free — like clause splitting, it costs nothing and gets
exact tests.
"""

import math
import re
import threading
from collections import Counter

# k1 damps how much repeating a term keeps helping; b controls length
# normalization. The literature's defaults — this project has no labeled data
# for tuning them, and inventing values would be worse than using the standard.
K1 = 1.5
B = 0.75

# Alphanumeric runs, optionally dot-joined, so "59.18.230" stays one token while
# a sentence-ending "rent." yields "rent" (the group needs alnum after the dot).
TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if len(token) > 1]


class Bm25Index:
    """In-memory BM25 index over a fixed corpus of chunks.

    Postings are term-major, so scoring touches only the chunks that share a
    term with the query instead of the whole corpus.
    """

    def __init__(self, documents: list[str], metadatas: list[dict]):
        self.documents = documents
        self.metadatas = metadatas
        self.doc_count = len(documents)
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.lengths: list[int] = []
        for index, document in enumerate(documents):
            counts = Counter(tokenize(document))
            self.lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self.postings.setdefault(term, []).append((index, frequency))
        self.average_length = (
            sum(self.lengths) / self.doc_count if self.doc_count else 0.0
        )

    def _idf(self, term: str) -> float:
        document_frequency = len(self.postings.get(term, ()))
        # Robertson–Sparck-Jones idf, +1 inside the log so a term appearing in
        # most of the corpus scores ~0 rather than going negative.
        return math.log(
            1 + (self.doc_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )

    def scores(self, query: str) -> dict[int, float]:
        scores: dict[int, float] = {}
        for term, query_frequency in Counter(tokenize(query)).items():
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for index, frequency in postings:
                normalizer = K1 * (
                    1 - B + B * self.lengths[index] / self.average_length
                )
                # Query-term frequency multiplies the contribution: a clause
                # that says "harmless" three times means it three times.
                scores[index] = scores.get(index, 0.0) + query_frequency * idf * (
                    frequency * (K1 + 1) / (frequency + normalizer)
                )
        return scores

    def search(self, query: str, k: int) -> list[tuple[str, dict]]:
        """Top-k (document, metadata) pairs; chunks sharing no term are omitted."""
        scores = self.scores(query)
        ranked = sorted(scores, key=lambda index: (-scores[index], index))[:k]
        return [(self.documents[i], self.metadatas[i]) for i in ranked]


_indexes: dict[str, Bm25Index] = {}
_index_lock = threading.Lock()


def get_index(collection_name: str, load) -> Bm25Index:
    """Build (once) and cache the index for a collection.

    `load` returns (documents, metadatas) — injected so this module never
    imports the vector store, which keeps it unit-testable without Chroma.
    Parallel scan threads share one index; building it for 359 statute chunks
    takes milliseconds and a few MB.
    """
    with _index_lock:
        index = _indexes.get(collection_name)
        if index is None:
            index = Bm25Index(*load())
            _indexes[collection_name] = index
        return index
