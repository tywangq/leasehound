# 🐕 LeaseHound

**Upload your lease. LeaseHound sniffs out the clauses that shouldn't be there.** A two-layer RAG system that answers tenant-law questions and scans rental agreements for prohibited provisions — grounded in Washington State's Residential Landlord-Tenant Act (RCW 59.18).

**🐕 [Live demo](https://leasehound-671004460975.us-west1.run.app)** — one click scans the sample lease, warm and answering in ~100 ms ([how that stays free](#try-it)).

![LeaseHound in motion: the sample lease is scanned clause by clause, the red-flag report pins to the side panel, and a follow-up question gets a cited answer](docs/demo.gif)

> ⚖️ LeaseHound is an educational tool, not legal advice. Verdicts are judged against a snapshot of RCW 59.18 fetched 2026-07-25 — the law may have changed since.

**Washington's RCW 59.18.230 alone prohibits ten kinds of lease provision, and a landlord who knowingly includes one owes up to 2× monthly rent in statutory damages.** Most tenants sign without knowing that — and the law is the part a chatbot gets wrong, not the lease ([why](#why-retrieval-and-not-a-long-context)).

**Every number here is measured, and several of the features behind them were measured and then not shipped** ([evaluation suite](#evaluation)):

| what was tested | measured result |
| --- | --- |
| scan mode · 6 hand-labeled leases · 18 planted violations | 18/18 flagged red · **0 false reds** · 18/18 citations correct · protections exact on 6/6 — [the acceptance bar](evaluation/README.md#scan-layer-evaluation--red-flag-precision--recall-6-labeled-leases) |
| scan mode · the same 6 leases, same model, no retrieval | 14/18 flagged red · 1 false red · **3/14 citations correct** · protections exact on 0/6 — [what retrieval is for](evaluation/README.md#no-retrieval-baseline--the-same-leases-and-model-closed-book) |
| scan mode · 40 generated leases · 61 planted violations | **60/61 flagged red** · 7 reds outside the label set, [6 of them real, so precision ≥ .896](evaluation/README.md#scaling-past-the-ceiling--40-generated-leases-labels-for-free) · 60/60 citations correct · protections exact on 39/40 |
| scan mode · retrieval · 79 labelled clauses | lenient 79/79 · [**strict 39/79, and all 40 shortfalls trace to one statute**](evaluation/README.md#the-labelled-set-said-ship-it-the-40-lease-set-said-no) |
| scan mode · permissive vs prohibited · 35 labelled clauses | shipped index 8/10 prohibitions flagged · 2 false reds · candidate fix **10/10 flagged · 3 false reds** — [opposite defects, neither shipped](evaluation/README.md#the-labelled-set-said-ship-it-the-40-lease-set-said-no) |
| scan mode · cost & latency · 135 logged scans (9–15 clauses) | **≈ $0.011/scan** · p50 8.8 s · [p95 18.5 s is the provider, not the pipeline](#what-a-scan-costs) · [that mean is a warm-cache number](#what-a-scan-costs) |
| ask mode · 82 generated questions | **82/82 answers consistent with the statute** · 81/82 grounded · 75/82 cite the ground-truth section — [graded against reference statute text](evaluation/README.md#generation-layer-evaluation--is-the-final-answer-right-n82--2-configs) |
| ask mode · router · 15 cases × 15 samples | **15/15 on the generation model** · 14/15 on the cheaper utility model — [after "there are cockroaches everywhere" reached no statute in 5 tries out of 5](evaluation/README.md#the-router--every-metric-above-assumes-retrieval-ran-at-all) |
| ask mode · cost & latency · 20 logged questions | **≈ $0.003/question** · p50 6.8 s · [1.5× the two-stage pipeline it beat by 1–2 questions](#what-the-extra-stages-cost) · [3% cached, against 65% for a warm scan](#what-a-question-costs-and-what-the-extra-stages-buy) |
| both modes · 5 prompt-injection payloads inside hostile leases | **5/5 held** · every planted violation still red · no scan suppressed · report stays clean carried into ask mode — [one payload beat the first run](evaluation/README.md#prompt-injection-resistance--the-lease-is-hostile-input) |

---

**That table is the claim. Everything below is how each number was produced, what it cost, and what it changed** — including the features that were built, measured and *not* shipped, which is how the architecture earned its shape instead of accumulating it.

[Why retrieval, not a long context](#why-retrieval-and-not-a-long-context) · [Architecture](#architecture) · [Try it](#try-it) · [What a scan costs](#what-a-scan-costs) · [The is-this-a-lease gate](#the-gate--three-answers-that-used-to-be-two) · [Privacy](#privacy) · [Development](#development) · [Evaluation](#evaluation) · [Roadmap](#roadmap)

## Why retrieval and not a long context

The ten provisions RCW 59.18.230(2) prohibits are specific, which is what makes them checkable: waiving rights under the chapter, waiving class participation, an NDA covering rent terms, a confessed judgment, paying the landlord's attorney fees outside a court award, exculpation or indemnity for the landlord's own liability, an arbitrator named at signing, arbitration the tenant helps pay for, a late fee inside the five-day grace period, and rent by electronic means only. Each has its own subsection, and each is the kind of thing a lease states plainly enough to be read against the statute.

**Why not just paste your lease into ChatGPT?** The lease fits in a context window; the law shouldn't come from parametric memory. Statutes change (RCW 59.18.230 was amended in 2025), models mix up states and invent citation numbers, and a chat answer can't be verified. LeaseHound retrieves the current statute text and cites the exact section — every claim has a clickable source. This claim is measured, not asserted (see the [no-retrieval baseline](evaluation/README.md#no-retrieval-baseline--the-same-leases-and-model-closed-book)).

## Architecture

Two corpus layers, two query modes:

| | Layer 1: Reference corpus | Layer 2: Your document |
| --- | --- | --- |
| Content | State statutes + official guidance | The lease you upload |
| Processing | Offline ingestion, evaluated with an ablation suite | Split into clauses on the fly (deterministic, no LLM) |

- **Scan mode** — walks your lease clause by clause, retrieves the governing statute for each, and produces a structured red-flag report with citations. Each clause queries the statutes directly, skipping the query-rewriting stages: those bridge renter vocabulary to statute vocabulary (see the adversarial experiment below), and a lease clause already speaks statute — a hybrid lexical channel was [measured here and rejected](evaluation/README.md#hybrid-retrieval-bm25--dense--measured-and-not-shipped) too. Clause judgments are independent API calls, so they run concurrently — a 15-clause lease scans in ~10 seconds instead of ~90. A second, negative-space pass checks a hand-curated, statute-cited checklist of required protections and reports what the lease *fails to include* (the LLM only judges presence — it never invents requirements); because "this lease omits X" is a claim about the whole document, that pass reads the document in 24k windows and merges them — concurrently, on the same pool the clause pass uses — rather than reading the first 24k and inferring the rest. A document sanity check runs before any clause is judged: a document *about* renting is scanned with its verdicts marked unreliable, and one unrelated to renting is [refused rather than judged](#the-gate--three-answers-that-used-to-be-two) — overridably, because a document that can suppress its own report is the most effective attack on a scanner. Scans judge at most 60 clauses to bound what one upload can spend; past that the scan is [partial and says so](#when-a-lease-is-longer-than-the-cap) rather than refused, because [real WA housing agreements run to 270 clauses](evaluation/README.md#document-formats--real-published-leases-and-real-numbering)
- **Ask mode** — RAG Q&A over the statute corpus ("Can my landlord charge a late fee on day 3?"); once you've scanned a lease, the report joins the chat context so answers are about *your* lease. A session-scoped vector collection for full lease-text retrieval is planned.

Retrieval pipeline: LLM-driven semantic chunking & augmentation (Pydantic structured outputs, parallel ingestion with validation + retry) → query router (greetings and small talk skip retrieval entirely — [and it used to skip real questions too](evaluation/README.md#the-router--every-metric-above-assumes-retrieval-ran-at-all), which is why it now runs on the generation model) → dual-query retrieval (original + rewritten) merged with reciprocal rank fusion → self-grading retrieval (CRAG-style) → LLM reranking → grounded generation with source citations.

### Code map

```mermaid
graph TD
    FETCH["scripts/fetch_corpus.py<br>download + normalize RCW 59.18"] --> INGEST["ingest.py<br>LLM chunking + augmentation"]
    FETCH --> NAIVE["ingest_naive.py<br>fixed-size baseline"]
    INGEST --> DB[("vector_db<br>Chroma, one collection per variant")]
    NAIVE --> DB

    CLI["python -m leasehound.scan<br>command line"] --> SCAN["scan.py<br>scan_steps: the one orchestration<br>+ clause-by-clause judge"]
    APP["app.py<br>Gradio UI + scan cache"] --> SCAN
    API["api.py<br>FastAPI /v1, typed contract"] --> SCAN
    APP -.->|mounts| API
    APP --> ANSWER["answer.py<br>grounded Q&A"]
    API --> ANSWER
    UPLOAD["upload.py<br>split lease into clauses"] --> SCAN
    SCAN --> METRICS["metrics.py<br>per-request cost & latency log"]
    ANSWER --> METRICS
    SCAN --> RETRIEVAL["retrieval.py<br>shared retrieval layer"]
    ANSWER --> RETRIEVAL
    BM25["bm25.py<br>lexical channel, measured & off"] -.-> RETRIEVAL
    EVAL["evaluation/<br>testsets + eval suite<br>retrieval · generation · scan · baseline · injection"] --> RETRIEVAL
    EVAL --> SCAN
    RETRIEVAL --> DB
```

Top half runs offline (build the statute library); bottom half is runtime. Three clients — command line, browser, HTTP — enter through the same `scan_steps`, and both query modes and the evaluation suite sit on the same `retrieval.py`, so an ablation result measures exactly the code the product runs. `app.py` mounts `api.py` so one process and one command serve both surfaces.

## Setup

```bash
pip install -e .
echo "OPENAI_API_KEY=sk-..." > .env
python scripts/fetch_corpus.py     # fetch RCW 59.18 → corpus/wa/statutes/ (98 sections)
python -m leasehound.ingest        # chunk + embed → vector_db/ (one-time, a few cents)
```

## Try it

```bash
python -m leasehound.scan examples/sample_lease.md   # CLI
python -m leasehound.app                             # web UI at / , HTTP API at /v1 , docs at /docs
```

<a id="three-clients-one-pipeline"></a>**Three clients, one pipeline.** The CLI, the web UI and the HTTP API are all callers of `scan.scan_steps`, which yields the scan as events — split, gate, each clause, protections, done — and says nothing about how they should look. The CLI prints them, the UI streams them into chat, the API ignores them and returns the outcome. That shape exists because the first two used to be *separate copies* of the same sequence, assembled from the same primitives, and had already drifted: the UI hard-coded the jurisdiction the CLI took as an argument. Adding a third caller is what made the duplication untenable rather than merely untidy.

The API is FastAPI with a typed contract and generated docs, and it added **no dependency** — Gradio already ships FastAPI, uvicorn and httpx, so `/v1` cost 0 MB in the image. `POST /v1/scan` returns the verdicts, the protections, the same rendered report, and **what the request spent**; `summary.partial` tells a caller whether the clause cap left anything unjudged, because 60-of-270 and 60-of-60 are different answers.

```bash
LEASEHOUND_API_TOKEN=dev python -m leasehound.app
curl -s -H "X-API-Token: dev" -F file=@examples/sample_lease.md localhost:7860/v1/scan | jq .summary
```

**The paid routes are shut unless that variable is set, and the hosted demo does not set it.** Gradio's queue bounds what browser visitors can start and bounds nothing at all about `curl` in a loop, so an open unauthenticated `/v1/scan` is an unmetered hole straight into the API key — the same reasoning as [the clause cap](#what-a-scan-costs). Unset, the routes answer 503 and say why, while `/docs` and `/v1/health` stay open, which is the part worth showing anyway.

The web UI is a single chat with an artifact-style side panel: attach a lease (or one-click the sample) to scan it, or just type to ask questions. Scan progress streams into the chat clause by clause, and the finished red-flag report pins to the panel so it stays visible while you ask follow-ups — answered token-by-token with the report in context and statute citations.

![The finished state, readable: a question about the late-fee clause, its cited answer, and the pinned scan report](docs/screenshot.png)

No setup at all: the **[hosted demo](https://leasehound-671004460975.us-west1.run.app)** runs the same code on Cloud Run — the `Dockerfile` bakes the vector store into the image, so the container is stateless and scales to zero between visitors. Cold starts are the cost of that, a bad first impression for the one visitor who matters, so a Cloud Monitoring uptime check probes every five minutes: it keeps an instance warm, alerts me on failure, and stays inside the free tier at 288 requests/day — `min-instances=1` would have solved the same problem for about $15/month.

<a id="what-the-image-ships"></a>**What the image actually ships**, since the cold start is what visitors feel. The container is **738 MB**, of which the vector store is **9.8 MB** — so the store was never the interesting number, and this README used to imply otherwise by listing "a 42 MB vector store" beside Chroma and Gradio as if they were peers. Three things were wrong underneath that, and they came off in three steps:

| | before | now |
|---|---|---|
| vector store in the image | 49 MB · 4 collections · 1655 chunks | **9.8 MB · 1 collection · 359 chunks** |
| image | 998 MB | **738 MB** (−130 unused deps, −146 Gradio 6, +16 the August dependency refresh) |
| known advisories in the shipped set | **44** across 4 packages | **0** actionable, 1 unreachable and named |

`COPY vector_db` shipped the whole *development* store — the three ablation collections the [evaluation](evaluation/README.md) needs and the app never queries, including `_230split`, an experiment that was [measured and rejected](evaluation/README.md#scan-mode-retrieval--one-miss-and-three-fixes-that-did-not-ship). `scripts/export_runtime_db.py` now writes a purpose-built store holding only the collection the app serves; it costs nothing, because every chunk's embedding already exists and is copied rather than recomputed.

Separately, chromadb declares `kubernetes` (for running Chroma as a server) and `onnxruntime` (for its built-in local embedder) — 133 MB for two features a `PersistentClient` embedding through the OpenAI API cannot use. Neither is in `sys.modules` after the app boots, which is what makes dropping them safe and what CI now re-checks on every commit.

**The third one is the interesting one, because pinning caused it.** Locking the dependency set made the image reproducible and simultaneously froze its vulnerabilities: `pip-audit` reported **44 advisories across pillow, gradio, starlette and chromadb**, and **43 of them were held in place by one line** — the `gradio<6` cap in `pyproject.toml`, since Gradio 5 pins `pillow<12` and `starlette<1.0` while the fixes are 12.1.1 and 1.0.1. Moving one floor to `gradio>=6.6` cleared all 43 and took 146 MB off the image. What remains is a pre-auth code-injection advisory in ChromaDB's *server* API, which this deployment does not serve — it embeds Chroma as a `PersistentClient`, the same reason the `kubernetes` dependency comes out — so it is ignored by name with that reason in `ci.yml` rather than left to turn the job permanently red. **A lockfile without an audit step is a promise to keep shipping a fixed set of versions including their known holes**, so `ci.yml` audits the lockfile now.

**Re-measured on the deployed image: cold start 14.3 s median** (3 samples, 13.5–15.2 s), down from the ~25 s this README used to quote with an admission attached that it predated the slimming. What moved is the part a cold instance actually pays for — Artifact Registry puts the *compressed* pull at **313 MB before, 234 MB now**, which is the transfer, not the 738 MB on disk. `scripts/measure_cold_start.py` re-derives it; the old figure went stale because it was typed by hand once and then the image moved underneath it.

**No visitor pays those 14 seconds, and finding out why was the more useful result.** An uptime check hits the service every 5 minutes from several of Google's probe regions, so the instance is never idle long enough to be reclaimed and the landing page comes back in **0.1 s**. Which also means the cold start is not measurable on the live URL at all: the first run of that script sat out three 17-minute windows and reported 0.34, 0.14 and 0.18 s as cold starts — one of them faster than its own warm follow-up, which is the tell. The fix was not a longer wait, it was to stop assuming the window was quiet: the script now reads Cloud Logging for the window it just waited through and **refuses to label a sample cold if anything else touched the host**, the same way `record_demo.py` refuses to write a GIF it caught on the cache path. The number above comes from a `--tag cold --no-traffic` revision of the same image, which has its own instance pool and so can go cold while the demo stays warm.

`examples/sample_lease.md` is a synthetic lease with **seven deliberately planted violations** (late fees inside the grace period, landlord attorney-fee clause, no-notice entry, rent NDA, rights waiver, electronic-payment-only, exculpation) — it doubles as the scanner's acceptance test. Current result: **7/7 planted violations flagged red with statute citations, zero false reds** among the ordinary clauses; the security-deposit clause comes back yellow (fact-dependent), which is the intended behavior. See `examples/README.md` for the expected-flags table and `examples/scan_report.md` for the full output.

## What a scan costs

Every scan is metered: one JSON line per scan (API calls, token usage, estimated cost, latency, verdict counts, clause-split mode — the file name, never lease text) appends to `logs/scan_metrics.jsonl` *and* stdout, so Cloud Logging keeps the records that the container's ephemeral filesystem doesn't. [Every question is metered too](#what-a-question-costs-and-what-the-extra-stages-buy), to `logs/ask_metrics.jsonl` — a separate file, because the scan summary aggregates on clause count and one mixed log would put questions into the cost-per-scan mean. `python -m leasehound.metrics` summarizes both.

The 15-clause sample lease: 17 LLM calls + 15 embeddings, ≈ $0.015, ~9 s. Across the 135 scans in the log (9–15 clauses each — evaluation-set builds, demo recordings and verification runs): **mean ≈ $0.0110/scan, p50 8.8 s, p95 18.5 s, max 71.4 s** — latency is dominated by the slowest clause in the pool, not by clause count, which is why the eight-way pool matters more than prompt size.

**That mean is optimistic, and it took building the HTTP API to notice.** Two scans of the identical document reported byte-identical usage — 17 calls, 31k prompt tokens — and cost **$0.0149 and $0.0087**. Same work, different rate: the provider serves recently-seen prompts from its own cache at a discount, and on the warm run **20,480 of 31,294 input tokens (65%) came from that cache**. Every number above was measured while scanning the same small set of documents over and over, so it sits nearer the warm end than the cold one. `cached_prompt_tokens` is in the metrics log now, because without it a cost log shows the pipeline getting cheaper when nothing about the pipeline changed. **Budget from $0.015, not $0.011, for a lease nobody has scanned before.**

Those figures are regenerated from the log by `python -m leasehound.metrics --write` into [`evaluation/scan_cost_summary.json`](evaluation/scan_cost_summary.json). The log itself is gitignored because it records a file name per scan; the summary is the shareable projection — counts and percentiles, no documents. This matters because the numbers above are the most-quoted in the project and were, until recently, reproducible from nothing.

**Cost is a property of the system; the latency tail is a property of the provider.** Split the same band by day: mean cost per scan moves between $0.0110 and $0.0129, while p50 moves from 8.4 s to 15.3 s and p95 from 13.0 s to 45.1 s. Same code, same clause counts, same prompts — 8% of scans simply took over 15 s. So p50 is the number that describes this pipeline, and quoting a tight p95 as if it were an engineering property would be quoting someone else's infrastructure. The [ask-mode measurement](#what-a-question-costs-and-what-the-extra-stages-buy) ran into the same thing hard enough to invert a conclusion.

Finished scans are cached in process memory, keyed by document **content hash** — not upload path, not browser session — so every visitor who clicks the sample lease after the first gets the saved report at zero API cost (logged as a `cache_hit` with cost 0), and re-uploading a renamed copy of the same file can't trigger a paid rescan. Attach a different lease to watch a live scan.

**Those figures come from 9–15 clause leases, so here is a real one.** The 49-clause HUD model lease from the [format probe](evaluation/README.md#document-formats--real-published-leases-and-real-numbering): **$0.0613 and 18.5 s** — 5.6× the cost and 2.3× the latency of the headline numbers. Anyone quoting $0.011/scan should know which size it describes.

### When a lease is longer than the cap

One scan judges at most 60 clauses. Real Washington housing agreements run past that — the UW 12-month agreement splits into 270 numbered provisions — so this bound decides what a visitor with a genuine lease actually gets.

It used to decide they got nothing: over the cap, the scan was refused. That was the wrong way to spend nothing extra, because judging 60 clauses and *naming the 210 it skipped* costs exactly the same as judging 60 clauses. The demo's single likeliest visitor action was returning an apology. Now the scan runs on the first 60 and the report opens with the clauses it did not read:

> ⚠️ **Partial scan — clauses 61–270 were not judged.** … The red / caution / clear verdicts below cover clauses 1–60 only; a red flag may well be sitting in the part that wasn't read.

The required-protections pass is deliberately exempt from the cap, and closing that hole was the harder half of this change. It answers what a lease *omits*, so it reads every clause — a prefix cannot support "this lease never says where your deposit is held." It had been sending one 24,000-character prompt, which for a long lease is not a partial answer but a wrong one; the [format probe](evaluation/README.md#the-same-question-against-documents-nobody-here-wrote) is where that surfaced. The pass now windows the document and merges, with `present` in any window overruling `missing` in another, so absence is only ever claimed after every window has looked. All 48 labelled and example documents fit in a single window, so no published number could move — and the gold set's 6/6 confirmed it after the change. On the real documents it moved plenty: two of the eight "missing protections" the probe had reported looked like fabrications of the truncation — and **a later re-run plus the document itself cut that to one**, because the other was a *false present* the windowed run had produced and this README had written up as a finding ([the correction](evaluation/README.md#the-same-question-against-documents-nobody-here-wrote)). A false `present` is the worse direction: it drops a legally required protection off the list a renter was told to check.

### The gate — three answers that used to be two

<a id="the-gate--three-answers-that-used-to-be-two"></a>Before any clause is judged, one cheap call classifies the document. It has always returned three kinds — `lease_agreement`, `document_about_leases` (a tenant-law guide, an article, a scan report), and `other` (a resume, an invoice, a banana bread recipe) — and the code threw two of them away:

```python
return check.kind == "lease_agreement"    # was: looks_like_lease
```

So a recipe and a tenant-law guide were handled identically: **scanned, and given a full set of landlord-tenant verdicts** under a warning that read "scanned anyway *on request*" — when there was no way for anyone to make such a request. The warning described a confirmation step that did not exist.

Both halves of that came from a correct earlier fix. The gate used to *raise*, and a wrong reject cost the visitor everything: the [real-document probe](evaluation/README.md#document-formats--real-published-leases-and-real-numbering) caught it rejecting a genuine WA university housing agreement. Warning-and-continuing also closed the injection suite's most effective attack by construction — a planted "this is not a lease, stop processing" line could no longer suppress a report, only annotate one. The over-correction was applying that to *every* non-lease.

Now the kinds diverge, and the override is real:

| Gate says | Behaviour |
|---|---|
| `lease_agreement` | scanned |
| `document_about_leases` | scanned, verdicts marked unreliable — a guide about renting is close enough to be worth judging |
| `other` | **not scanned.** No clause is read, and the message says how to insist |

The refusal is overridable because refusing is the attack: `other` must never be the last word. The override is a phrase the visitor types (`scan anyway`) rather than a button, for a reason that is not laziness — the upload is deleted the instant its text is extracted, so an override has to bring the document back with it. Nothing is kept waiting for a second opinion. That also makes the override unambiguously the *reader's*, which is the property that keeps planted text from suppressing anything on its own: at worst an injection costs a visitor one re-upload.

What it saves is small and the point is not the money: a refused recipe costs **1 call and $0.0002** instead of 7 calls and $0.005. The reason to do it is that five confident verdicts about a banana bread recipe are worse than no verdicts, and a report reading "0 red flags" over a document nobody judged is worse still — so a refusal renders as a refusal rather than as a clean bill of health, and leaves ask mode on law-only context instead of telling it that a report exists.

### What a question costs, and what the extra stages buy

Ask mode is metered the same way scan mode is now: one JSON line per answered question to `logs/ask_metrics.jsonl` and stdout, covering the router call, every retrieval stage and the streamed answer. That took longer to arrive than it should have, because a streamed completion reports no token usage until its final chunk — so there was nothing to hand the meter, and the price of the whole mode was left to a one-off script instead. **The one design decision this project kept on the thinnest evidence was the only one with no ongoing price tag.**

Across 20 logged questions: **mean $0.00300, 5.1 LLM calls, p50 6.8 s** ([`scan_cost_summary.json`](evaluation/scan_cost_summary.json), regenerated by `python -m leasehound.metrics --write`). What the extra stages cost, from `scripts/measure_ask_cost.py` over 12 renter-voice questions:

| ask-mode configuration | LLM calls | cost per question | median latency |
| --- | --- | --- | --- |
| naive chunks + CRAG (two-stage) | 3.33 | $0.00167 | 5.57 s |
| **full pipeline (shipped)** | **5.00** | **$0.00257** | **6.65 s** |

<a id="what-the-extra-stages-cost"></a>**So the full pipeline costs 1.5× and adds 1.1 s** for the [+.060 MRR and +5 questions on hit@5](evaluation/README.md#adversarial-rephrasing--the-same-82-questions-renter-voice) that the reworded set gave it. Stated plainly: that gain is one or two flipped questions per metric, bought at half again the price. It is kept because $0.0026 and 6.7 s are both far inside what a renter waiting on a legal answer will tolerate, and because the gain pointed the same way on all four metrics — but the honest version of "we keep the full pipeline" includes the price, and until this was measured it didn't.

**Every number in that table moved when the measurement moved into production, and one of them was simply wrong.** The published figure was 4.17 calls at $0.00288, 1.8× the two-stage pipeline. The call count was too low because the script reached past `answer_question()` into `fetch_context()`, so the router — which runs on every single message — was never counted, while the docstring claimed one known deviation from production. It is 5.00 now, and exactly 5.00 on all 12 questions: router, rewrite, grade, rerank, answer. The dollar figures moved for a duller reason — n went from 6 to 12, which restrides the sample — so this is not a like-for-like correction of the old row, and the ratio fell from 1.8× to 1.5× with it.

**Ask mode gets almost no prompt-cache discount: 3% of input tokens cached, against 65% for a warm scan.** (It read 0% over the first ten questions; twenty questions in, two of them repeats, it is 3% — the direction is what matters, not the digit.) The two modes are opposites on this, and the reason is prompt layout — the retrieved statutes sit at the *front* of the ask-mode system prompt and differ for every question, so consecutive questions share almost no cacheable prefix. So the [budget-from-the-cold-number caveat](#what-a-scan-costs) that applies to the scan figures barely applies here: $0.0030 is close to the cold price already.

One methodological note, since it changed a conclusion. An early run's *mean* latency said the two-stage pipeline was 6 s **slower**, on the strength of a single 54 s outlier against a 3–9 s spread — a provider hiccup, not a property of the configuration. Latency is reported as a median for that reason, and summaries can be recomputed from the saved per-question rows (`--rescore`) rather than re-paying for a cleaner sample.

**And metering ask mode is what caught the bug the whole evaluation suite could not see: [the router was sending real legal questions to the chitchat path](evaluation/README.md#the-router--every-metric-above-assumes-retrieval-ran-at-all).** "There are cockroaches everywhere. What can I do?" reached no statute 5 times out of 5. It showed up as a row costing $0.00015 in 1.8 s among rows costing $0.0025 in 6 s — a misroute raises nothing, and the only trace it leaves is being suspiciously cheap.

### What stops a visitor running up the bill

Four bounds, and only the first two live in this repo:

- **Per scan** — the clause cap is 60, so the clause pass is bounded at 60 embeddings + 60 completions. That is not the whole bound: add **one gate call, plus one protections call per 24,000 characters**, because the protections pass is [deliberately exempt from the cap](#when-a-lease-is-longer-than-the-cap) — it reports what the lease omits, so it has to read the part the cap skipped. So the ceiling is `60 + 1 + ceil(chars / 24000)` completions, which is *bounded in clauses but linear in document length*. Measured, from the [format probe](evaluation/README.md#document-formats--real-published-leases-and-real-numbering): the 15-clause sample lease is 17 completions (≈ $0.015), the 49-clause / 40k-character HUD lease is 52 (≈ $0.06), the 270-clause / 68k-character UW agreement is 64. The second term is small and grows slowly — 24,000 characters is a lot of lease — but "60 clauses" alone is not the bound, and a *per-scan dollar budget* would be ([Roadmap](#roadmap)).
- **Per document** — finished scans are cached by content hash, so re-uploading the same lease (or a renamed copy) costs $0.
- **Per instance** — `--max-instances 1` plus a 4-concurrent / 16-queued admission gate, so the fan-out cannot multiply across instances.
- **Per account** — the actual hard stop is a spend limit set on the OpenAI project. The GCP budget alert is an *alert*: it emails, it does not block. Worth saying out loud, because a queue that turns away visitor 21 bounds concurrency, not the daily total — a patient script could still spend real money, and only the provider-side limit stops that.

### What happens when more than one person shows up

Concurrency was reasoned about and never measured, so `scripts/measure_concurrency.py` measures the part that is measurable for free — the pool's shape, with the per-clause API call stubbed at its logged latency — and does the arithmetic for the part that is configuration.

The README used to assert that "latency is dominated by the slowest clause in the pool, not by clause count". That predicts a *staircase*, and there is one: 1, 4, and 8 clauses all cost one wave; 9 and 16 both cost two; 17 and 24 both cost three. The real 49-clause scan agrees — 7 waves × ~2.6 s = 18.5 s measured — so the model predicts real behaviour rather than just fitting the stub.

Admission control is `demo.queue(max_size=QUEUE_MAX_SIZE, default_concurrency_limit=QUEUE_CONCURRENCY)` — 16 and 4, imported by the measurement script rather than restated in it: **4 visitors scan at once** (a peak fan-out of 32 concurrent API calls), 16 more wait, and **visitor 21 is turned away**. At a full queue the worst wait is about 32 s. Peak resident memory across four concurrent scans is **≈300 MB** (297–302 across runs, so the spread is quoted rather than a false-precision single figure), which is what makes one Cloud Run instance enough — it was 236 MB on Gradio 5, and re-measuring after [that upgrade](#what-the-image-ships) is the only reason this number is not stale.

```bash
python -m scripts.measure_concurrency          # $0, no API calls
```

Not measured, and stated rather than implied: real API latency under load, provider rate limiting at 32 concurrent calls, and Cloud Run CPU throttling. This bounds the queueing behaviour, not the provider's.

## Privacy

A lease is sensitive: names, address, rent. LeaseHound keeps handling minimal —

- Uploads are parsed in memory; the uploaded temp file is deleted immediately after parsing, and lease text never enters the vector store. The per-scan metrics log records counts, cost, and the file name — never lease text.
- Scan results are cached in process memory only (keyed by content hash, bounded, never written to disk) and vanish when the instance scales to zero.
- Clause text is sent to the OpenAI API for embedding and analysis. OpenAI does not train on API data (retained ~30 days for abuse monitoring), but treat it as a third-party disclosure: don't upload a document you couldn't share — or use the sample lease.
- Local PII redaction before the API call (regex + local NER — it can't be done by the same cloud LLM you're redacting *from*) is a considered follow-up.
- Prefer fully local inference? All LLM calls route through litellm: set `LEASEHOUND_UTILITY_MODEL` / `LEASEHOUND_GENERATION_MODEL` to e.g. `ollama/llama3.1` (expect weaker verdicts from small local models; embeddings still require an OpenAI key).

## Development

```bash
pip install -e ".[dev]"
pytest          # 173 unit tests: clause splitting across seven real numbering conventions, the judge-window invariant, protections windowing + merge precedence + the window pool preserving document order, the partial-scan contract, the gate refusing an unrelated document without judging a clause and the override reaching the scan, RRF merge + hybrid wiring, BM25 scoring, enumerated-catalog parsing and chunk-id ordering, ask-mode prompt assembly + stream unwrapping, the router call being counted and a usage-only stream chunk not crashing on it, statute-drift comparison, section completion, the injection scorer's negation handling, eval artifacts surviving a cheap re-run, the documented artifact inventory matching the directory, the HTTP contract and its spend gate, one orchestration serving all three clients, report rendering, cancellation, per-request metrics for both modes, scan cache, the synthetic-dataset verifier, prompts that must treat lease text as data, cited-sources footer, privacy cleanup
ruff check leasehound evaluation scripts tests
```

The tests render no CSS, which is a real gap and not a theoretical one: the gradio 6 upgrade flipped the composer's flex container from a row to a column, so a rule of ours that had been centering it vertically started collapsing it *horizontally* — 734 px of input bar down to 237, the placeholder wrapped onto two lines with "Washington…" clipped off the end. 166 green tests and a stale screenshot between them said nothing. Re-recording the README's assets is what surfaced it, which is the argument for `scripts/record_demo.py` existing at all. It needs a browser, dev-only and deliberately absent from `pyproject` and `requirements-lock.txt` — nothing that ships needs one:

```bash
pip install playwright && playwright install chromium
python -m leasehound.app &                 # a cold cache, so the GIF has a live scan in it
python -m scripts.record_demo              # ≈ $0.018: one scan, one question
```

CI (GitHub Actions) runs both on every push, plus a third job that **builds the container and boots it**. That job exists because the deployed artifact was the least-tested thing here: the test job installs `pyproject`'s version ranges and is the canary for upstream drift, while the image installs the pinned `requirements-lock.txt` it actually ships and proves those pins still resolve and import together. It builds against a placeholder store — the real one isn't in git, and re-embedding the corpus per commit would spend money re-proving something no commit changes. That job also **audits `requirements-lock.txt` with `pip-audit`**, because a lockfile with no audit step is a promise to keep shipping a fixed set of versions including their known holes; it found 44 advisories the first time it ran. Tests themselves cover the deterministic core only: no API calls, no vector DB.

Deploying is three steps, because the image ships a different store than development uses:

```bash
python -m scripts.export_runtime_db --force   # vector_db/ → vector_db_runtime/, 359 chunks, $0
docker build -t leasehound .                  # local check; see "What the image ships" above
gcloud run deploy leasehound --source . --region us-west1
```

**The third line is the one this README used to leave out, and leaving it out cost a week.** The live demo sat **31 commits behind `main`** — still serving Gradio 5, still serving the `og:title = "Gradio"` share card, still running the router that [sent real legal questions to the chitchat path](evaluation/README.md#the-router--every-metric-above-assumes-retrieval-ran-at-all) — while every CI run stayed green, because CI proves the image *builds and boots* and nothing at all proves the image was *shipped*. A step that only lives in shell history is a step that silently stops happening. `--source` uploads the set `.gcloudignore` allows, which is why that file exists rather than falling back to `.gitignore`: the latter excludes both vector stores, and the image needs one of them.

The experiment write-ups live in [`evaluation/README.md`](evaluation/README.md) next to the artifacts they describe; `scripts/` holds the one-off measurement tools (`measure_concurrency.py`, `measure_ask_cost.py`, `measure_cold_start.py`, `probe_router.py`, `build_enumerated_collection.py`, `check_corpus_drift.py`), each of which prints what it costs before it costs it, plus the two asset generators — `record_demo.py`, which regenerates the GIF and screenshot above by driving the real UI, and `render_favicon.py`, which draws the 🐕 tab icon in a real browser because the emoji has to come from a real emoji font — plus the two deployment tools above (`export_runtime_db.py`, `image_smoke_test.py`). Three of these exist because an artifact went stale or silently wrong rather than loudly broken: the GIF drifted from the product it illustrated, the cold-start figure outlived the image it described, and the favicon served a correct `200 image/svg+xml` that no browser drew.

Two more workflows sit alongside it. `corpus.yml` watches the statute snapshot for drift — free, no secret, [described above](#keeping-the-snapshot-honest). `eval.yml` runs the paid evaluations: the gold-set scan eval, the ask-mode retrieval eval, and the scan-mode retrieval eval — that last one first, since at ~$0.0002 it says whether a regression is retrieval or judgment before anything expensive runs — on every push to `main` that touches pipeline code (≈ $0.15/run — path-filtered so docs commits cost nothing), with the pricier generation eval and the 40-lease synthetic set on manual dispatch. Scores land in the job summary as a report, not a gate: temperature-0 API calls still drift a flag's worth between runs, and a hard threshold would flake. Forks never see the API key (main-only triggers), and the workflow skips gracefully when the secret isn't configured.

## Corpus status

- ✅ `corpus/wa/statutes/` — RCW Chapter 59.18, all 98 sections (public domain, includes 2025 amendments), fetched and normalized by `scripts/fetch_corpus.py`
- ⬜ `corpus/wa/guides/` — WA Attorney General landlord-tenant guidance (planned)
- ⬜ Seattle municipal layer (planned)
- States are first-class metadata: each state is an independent collection; CA is the planned follow-up.

### Keeping the snapshot honest

Every verdict inherits the date of that fetch — the report footer stamps it — and Washington amends RCW 59.18 most sessions (.230 was amended in 2025). Nothing in the repo would have noticed.

```bash
python -m scripts.check_corpus_drift    # exit 0 clean · 1 drifted · 2 the check itself broke
```

A weekly GitHub Action (`corpus.yml`) re-fetches the chapter, renders it through `fetch_corpus.py`'s *own* normalization, and compares byte-for-byte against what is committed. On drift it files an issue carrying the per-section diff and the checklist a human owes: read the amendment, re-fetch, bump `CORPUS_SNAPSHOT`, re-embed, re-run the gold set. It needs no API key and nothing beyond the standard library, so it costs **$0** whether or not the law changed.

Two choices there are deliberate. It **alerts rather than auto-updates** — amended legal text is the scanner's ground truth, and a changed prohibition may need a new required-protections checklist item, which is a judgment call, not a rebuild. And it **separates drift from its own failure**: if leg.wa.gov reorganizes its markup the parser returns nothing, and a naive check would announce that all 98 sections were repealed, so a section-count guard reports a scraper bug (exit 2) instead. Different fix, different urgency, different person.

## Evaluation

Five measured layers, ordered cheapest-first so an idea can be killed before it is paid
for. Four features were built, measured, and **not shipped** — the negative results are
kept on purpose, because they are how the architecture earned its shape.

**Full write-ups, with every table and the reasoning behind each decision, are in
[`evaluation/README.md`](evaluation/README.md).**

| experiment | the question it answers | what it found |
| --- | --- | --- |
| [Scan layer](evaluation/README.md#scan-layer-evaluation--red-flag-precision--recall-6-labeled-leases) | Does it catch planted violations in hand-labeled leases? | 18/18 red, 0 false reds, 18/18 cited |
| [No-retrieval baseline](evaluation/README.md#no-retrieval-baseline--the-same-leases-and-model-closed-book) | What does the pipeline add over pasting the lease into the model? | citations 18/18 vs **3/14** — retrieval is the difference |
| [40 generated leases](evaluation/README.md#scaling-past-the-ceiling--40-generated-leases-labels-for-free) | Does it hold past the hand-labeled ceiling? | 60/61 red; found and fixed an evidence-bleed bug |
| [Prompt injection](evaluation/README.md#prompt-injection-resistance--the-lease-is-hostile-input) | Can a lease talk to the model? | 5/5 held — after one payload suppressed a whole scan |
| [Document formats](evaluation/README.md#document-formats--real-published-leases-and-real-numbering) | Does the pipeline survive documents nobody here wrote? | **6 of 7 conventions failed silently**, and a silent truncation invented two missing protections |
| [Scan-mode retrieval](evaluation/README.md#scan-mode-retrieval--one-miss-and-three-fixes-that-did-not-ship) | Does the governing statute reach the judge, and do the candidate fixes work? | **all 40 partial misses are one section**; the fix that closed them (.492 → .984) cost a false red in July |
| [Permissive vs prohibited](evaluation/README.md#the-labelled-set-said-ship-it-the-40-lease-set-said-no) | Can the judge tell "you may" from "you must only"? | the gold set cleared the rejected fix, the 40-lease set failed it again — and **the shipped index reads two prohibitions as green** |
| [Retrieval ablation](evaluation/README.md#retrieval-ablation--section-level-n82) | Which pipeline stage actually earns its cost? | naive chunking ties the six-stage pipeline |
| [Adversarial rephrasing](evaluation/README.md#adversarial-rephrasing--the-same-82-questions-renter-voice) | Does it hold when renters don't speak statute? | full pipeline wins; the earlier tie was vocabulary leakage |
| [Hybrid BM25](evaluation/README.md#hybrid-retrieval-bm25--dense--measured-and-not-shipped) | Does a lexical channel help ask mode? | no — the apparent gain was vocabulary leakage |
| [Generation layer](evaluation/README.md#generation-layer-evaluation--is-the-final-answer-right-n82--2-configs) | Is the final answer right and grounded? | 82/82 consistent, 81/82 grounded |

**Three worth the click:**

- **[The no-retrieval baseline](evaluation/README.md#no-retrieval-baseline--the-same-leases-and-model-closed-book)** is the most persuasive number in the project. Same model, same leases, no pipeline: 14 of 18 violations found, but only **3 of 14 citations correct** — it invents section numbers. Retrieval is not decoration here; it is the difference between a claim and a checkable one.
- **[Three candidate retrieval fixes, none shipped](evaluation/README.md#three-candidate-fixes-none-shipped)** — including one that *closed* the defect it targeted (strict retrieval .492 → .984) and was rejected anyway, because it cost a false red on a compliant clause. The metric that found the defect turned out to be a proxy, not an outcome.
- **[Five real published leases](evaluation/README.md#the-same-question-against-documents-nobody-here-wrote)** broke two things no self-generated test could: six of seven real numbering conventions, and a silent 24,000-character truncation that invented two "missing protections" out of text it never read.

One caveat applies everywhere: a table is one run, `temperature=0` is not determinism, and a gap of one or two questions is a tie. The [full version](evaluation/README.md) sits with the write-ups.

## Roadmap

- **Stop the judge inventing exclusivity, which is the actual defect under both indexes** — the [labelled set](evaluation/README.md#the-labelled-set-said-ship-it-the-40-lease-set-said-no) got built rather than skipped, and running it plus the 40-lease set moved the target. The rejected enumerated split takes strict retrieval .492 → .984 and recall on the generated set 60/61 → **61/61**, and it still reads *"by check or electronic payment"* as electronic-**only**, on the compliant lease, with the judge's own explanation inserting a word the clause does not contain. The shipped index has the mirror defect: it reads the exculpation and arbitration-cost prohibitions as green with no citation at all, which 18/18 gold recall never covered because no labelled lease plants a clause of either shape. So the fix is neither index — it is the judge distinguishing a permitted list from a mandate — and it is now testable against `permissive_pairs.jsonl` for **$0.028** instead of the $0.6 the generated set costs.
- False-premise and unanswerable question sets — the remaining adversarial categories (testing premise correction and honest refusal, not just retrieval)
- Recalibrate the is-this-a-lease gate on boundary documents — the [real-document probe](evaluation/README.md#document-formats--real-published-leases-and-real-numbering) found it accepts a tenancy addendum and rejects a genuine WA university housing agreement. Needs a labelled set of boundary cases, and a decision on whether such agreements fall under RCW 59.18 at all, before touching the prompt. The *unrelated* end is now handled and is a different problem from this one: see [the gate](#the-gate--three-answers-that-used-to-be-two)
- Replace the 60-clause cap with a per-scan dollar budget — the cap now [degrades instead of refusing](#when-a-lease-is-longer-than-the-cap), but a clause count is a proxy for spend, and clauses vary in length by 10×. A budget would let a 270-clause agreement of short provisions finish where a cap of 60 stops it arbitrarily
- OCR for scanned/photo leases (Tesseract) — today a no-text-layer PDF is detected and refused with an explanation
- Session-scoped vector collection for full lease-text retrieval in ask mode
- Fairness grade for the whole lease — derived mechanically from verdict counts, never model-invented
- CA corpus (states are first-class metadata already)
