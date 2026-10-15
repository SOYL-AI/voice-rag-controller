# D1 `d1_open`: popularity-stratified open-domain QA

D1 is the public open-domain dataset of research v2. It is built so that
the need for retrieval varies **continuously** rather than being nearly
binary. PopQA questions about very obscure entities need retrieval, questions
about very popular entities usually do not, and the 5 log-popularity bins
cover the range between. A small HotpotQA multi-hop slice adds questions that
need two passages.

| | |
|---|---|
| Items | **5,300** (5,000 PopQA + 300 HotpotQA) |
| Splits | dev 1,566 / test 3,734 (CONTRACTS.md §7; test is sealed) |
| Corpus | **49,694** passages (46,752 Wikipedia from 21,622 pages + 2,942 HotpotQA paragraphs) |
| Build | `python datasets/v2/build_d1_open.py` (seed 20261014) |

The closed-book model for the empirical label is Mistral-7B-Instruct-v0.3,
the only model in the study. **No model was run to build this dataset.**

## Files

| File | Committed | Content |
|---|---|---|
| `items.jsonl` | yes | 5,300 items, schema CONTRACTS.md §2 |
| `corpus.jsonl.gz` | yes | deterministic gzip of `corpus.jsonl` (23.6 MB) |
| `corpus.jsonl` | no (72.7 MB, gitignored) | written by the build; schema §3 plus optional `meta` |
| `splits.json` | yes | dev/test ids, seed, method, per-stratum balance check |
| `stats.json` | yes | every number in this README, plus the full build report |
| `.cache/` | no | raw downloads and every Wikipedia API response |
| `index/` | no | FAISS index from `retrieval/build_index_v2.py` |

## Sources and licences

| Source | Used for | Licence |
|---|---|---|
| PopQA (Mallen et al., ACL 2023, *When Not to Trust Language Models*), `akariasai/PopQA` `test.tsv` at HF revision `098765c7` | 5,000 questions, answers, `s_pop` | MIT |
| HotpotQA distractor dev (Yang et al., EMNLP 2018), `hotpotqa/hotpot_qa` config `distractor`, split `validation` (7,405 rows) | 300 questions + their 10 context paragraphs each | CC BY-SA 4.0 |
| English Wikipedia, current revisions fetched 2026-10-15 via the MediaWiki Action API (TextExtracts plain text) | subject articles and hard negatives | CC BY-SA 4.0 |

Every corpus line carries its `licence`. Every item records its source
licence in `meta.licence`. Wikipedia passages record `meta.pageid` and
`meta.revid`, so each passage can be traced to an exact revision. The corpus
as a whole is CC BY-SA 4.0 (share-alike, with attribution to Wikipedia
contributors and the HotpotQA authors).

## Item fields specific to D1

- `item_type`: `popqa` or `multihop` (`multihop` was added to the
  CONTRACTS.md enum together with this dataset).
- `retrieval_needed_authored`: always `null`. For public knowledge, whether
  retrieval is needed depends on what the model memorised. D1 therefore
  relies only on the **empirical** label (closed-book correctness from E1-v2).
- `popularity`: PopQA `s_pop` (monthly Wikipedia page views of the subject).
  This is the covariate. It is `null` for `multihop`.
- `answer_aliases`: PopQA `possible_answers` (non-empty strings). For
  HotpotQA, the single gold answer.
- `gold_doc_ids`: for PopQA, every passage of the subject's article (first 20
  passages) that contains an answer alias. Matching ignores case and accents
  and respects word boundaries; aliases of 3 characters or fewer must match
  case-sensitively, so "US" never matches "us". For HotpotQA, the passages
  holding the supporting-fact sentences (always 2).
- `meta`: `relation`, `subject`, `s_wiki_title`, `resolved_wiki_title`,
  `pageid`, `revid`, `s_pop`, `o_pop`, `log10_pop`, `pop_bin` (0 = least
  popular), `primary_gold_doc_id` (first alias-bearing passage in article
  order, usually the lead), `bin_rank`, `multi_valued_answer` (true for
  `genre` and `occupation`, 777 items), Wikidata URIs. HotpotQA items carry
  `hotpot_type`, `level`, `supporting_titles`.

## Construction

### PopQA (5,000 items)

1. **Spoken-ambiguity filter (item level).** A spoken question loses quotes,
   italics and capitalisation. So an item is kept only if (a) its
   `s_wiki_title` has no parenthetical disambiguator, meaning the subject is
   the primary Wikipedia referent of its name ("Doomwatch (film)" is
   dropped), and (b) its question string occurs exactly once in PopQA
   (duplicated strings name different entities). The whole `color` relation
   (34 items) is dropped because its answers are multi-valued and
   ill-defined (flags have several colours).

   | Filter | Rows removed |
   |---|---|
   | subject needs a disambiguator | 5,658 |
   | duplicated question string | 2,080 |
   | relation `color` | 34 |
   | **eligible pool** | **6,495** of 14,267 |

   The rule removes 63–76% of the creative-work relations (director,
   screenwriter, producer, composer, genre, author) and 3–14% of the others.
   An earlier draft dropped those six relations wholesale. That left a pool
   of 3,890, too small for the 5,000 items the v2 power analysis needs, so
   the item-level rule replaced it.
2. **Popularity bins.** 5 quantile bins of log10 `s_pop` over the eligible
   pool. Edges at `s_pop` = 137, 390, 1,443 and 10,608.
3. **Equal counts per bin (1,000 each), balanced across relations.** Inside
   each bin, a per-relation quota is water-filled evenly, capped at 70% of
   each (bin, relation) cell to leave room for gold-check failures. Any
   shortfall is then filled round-robin over relations. Candidates in each
   cell are taken in seeded hash order, `sha256("20261014|popqa|<id>")`.
   Rare relations limit how even the balance can be: `mother` and
   `capital of` are almost absent from bin 0.
4. **Gold-passage verification.** For every candidate, the subject article
   is fetched (redirects followed) and chunked. The candidate is accepted
   only if at least one alias appears in its first 20 passages; otherwise
   the next candidate in the same cell is tried. 5,634 candidates were
   attempted and 634 dropped:

   | Drop reason | n |
   |---|---|
   | no alias in the subject article | 522 |
   | subject title resolves to a disambiguation page | 89 |
   | article missing | 23 |

   Drops by bin (0→4): 174, 132, 131, 101, 96.

### HotpotQA (300 items)

Distractor dev, filtered to non-yes/no answers (458 removed) and questions of
30 words or fewer (128 removed, too long to speak naturally). Then 150
`bridge` and 150 `comparison` items are taken in seeded hash order of the
HotpotQA `_id`, keeping only items whose answer appears in a supporting
paragraph (no item failed). Text is repaired for the UTF-8-as-cp1252
mojibake present in the source (`â€“` → `–`). Paragraphs over 350 words are
split on sentence boundaries.

### Retrieval corpus (49,694 passages)

| Role (a passage may have several) | Passages |
|---|---|
| PopQA subject articles (first ≤ 20 passages) | 28,574 |
| name-similar hard negatives | 17,354 |
| same-relation hard negatives | 21,603 |
| HotpotQA supporting paragraphs | 598 |
| HotpotQA distractor paragraphs | 2,344 |

- **Chunking.** Plain-text extracts are split into paragraphs and
  reference-type sections are skipped (References, See also, External links,
  Notes, …). Paragraphs are packed greedily into passages of at most 350
  words. A new passage starts at a section heading once the current one has
  120 or more words, and fragments under 40 words are folded into the
  previous passage when that fits. Passage words: mean 182, median 178,
  p90 319, max 350.
- **Name-similar negatives.** For each item, Wikipedia search on the subject
  name (top 6 hits). The lead sections of the first 3 usable hits are added
  (≤ 2 passages each), excluding the gold page, disambiguation pages and
  "List of" pages. Example: "George Rankin" gets "George Rankin (judge)" and
  "George Rankin Irwin".
- **Same-relation negatives.** Lead sections of 2 other PopQA subjects with
  the same relation, preferring the same object: other politicians for an
  occupation question, other births in the same city. Candidates are
  ordered by a seeded hash, and leads are fetched 20 titles per request. A
  negative that is another item's subject article is already in the corpus
  in full and is only tagged.
- **HotpotQA.** All 10 context paragraphs of each sampled item.

## Splits

`sha256(id) mod 1000 < 300 → dev`, exactly as CONTRACTS.md §7. The
pre-specified balance criterion: rebalance a stratum only if its dev
fraction is more than 2 binomial SE from 0.30, or if χ² homogeneity within
its family has p < 0.05. Nothing met it, so **no adjustment was made**.

| Stratum | dev | test | dev frac | z |
|---|---|---|---|---|
| popqa bin0 | 300 | 700 | 0.300 | 0.00 |
| popqa bin1 | 293 | 707 | 0.293 | −0.48 |
| popqa bin2 | 286 | 714 | 0.286 | −0.97 |
| popqa bin3 | 314 | 686 | 0.314 | 0.97 |
| popqa bin4 | 286 | 714 | 0.286 | −0.97 |
| multihop bridge | 41 | 109 | 0.273 | −0.71 |
| multihop comparison | 46 | 104 | 0.307 | 0.18 |

χ² homogeneity: PopQA bins χ²(4) = 2.64, p = 0.62; multihop χ²(1) = 0.41,
p = 0.52. The test split is sealed. Only the build script writes test items,
and no generation, signal extraction or retrieval evaluation touches them.

**Pilot subsets.** `--n-popqa N` rebuilds with N PopQA items (N/5 per bin)
from the same seed. Each item's `meta.bin_rank` is its rank inside its bin
in seeded hash order, so `bin_rank < k` picks a nested pilot of k items per
bin straight from this build.

## Statistics

PopQA items per relation and popularity bin (bin 0 = least popular):

| Relation | b0 | b1 | b2 | b3 | b4 | total |
|---|---|---|---|---|---|---|
| author | 85 | 103 | 118 | 78 | 41 | 425 |
| capital | 31 | 35 | 50 | 84 | 238 | 438 |
| capital of | 6 | 12 | 37 | 51 | 56 | 162 |
| composer | 18 | 35 | 52 | 57 | 54 | 216 |
| country | 230 | 128 | 91 | 72 | 33 | 554 |
| director | 55 | 89 | 87 | 86 | 56 | 373 |
| father | 31 | 69 | 79 | 94 | 101 | 374 |
| genre | 49 | 84 | 109 | 92 | 76 | 410 |
| mother | 1 | 13 | 26 | 42 | 59 | 141 |
| occupation | 113 | 93 | 68 | 55 | 38 | 367 |
| place of birth | 153 | 111 | 69 | 37 | 17 | 387 |
| producer | 17 | 47 | 56 | 74 | 67 | 261 |
| religion | 21 | 33 | 32 | 48 | 61 | 195 |
| screenwriter | 33 | 67 | 80 | 94 | 70 | 344 |
| sport | 157 | 81 | 46 | 36 | 33 | 353 |
| **total** | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 5,000 |

`s_pop` per bin (median [min, max]): b0 80 [3, 136]; b1 226 [137, 388];
b2 739 [390, 1,442]; b3 3,403 [1,443, 10,574]; b4 49,453 [10,608, 15,101,521].

| | PopQA | multihop |
|---|---|---|
| question words, mean (min–max) | 7.0 (4–18) | 13.9 (6–30) |
| gold passages per item, mean (median) | 2.48 (1) | 2.0 (2) |

## Retrieval check (dev split only)

Built with `retrieval/build_index_v2.py`, using the same stack as v1:
`BAAI/bge-small-en-v1.5`, one vector per passage, exact `IndexFlatL2`, no
query instruction. bge's pipeline L2-normalises, so this ranks the same as
cosine. Passages are embedded as `title\ntext`. The query is the full
written question. hit@k = at least one gold passage among the top-k
distinct passages.

| Group (dev) | n | hit@1 | hit@3 | hit@5 | MRR@50 | primary hit@1 / @3 / @5 |
|---|---|---|---|---|---|---|
| **all** | 1,566 | **0.799** | **0.932** | **0.949** | 0.867 | |
| popqa | 1,479 | 0.792 | 0.930 | 0.948 | 0.862 | 0.761 / 0.920 / 0.943 |
| popqa bin0 (least popular) | 300 | 0.867 | 0.943 | 0.950 | 0.906 | 0.860 / 0.943 / 0.950 |
| popqa bin1 | 293 | 0.792 | 0.925 | 0.942 | 0.859 | 0.778 / 0.918 / 0.942 |
| popqa bin2 | 286 | 0.825 | 0.951 | 0.972 | 0.887 | 0.787 / 0.948 / 0.969 |
| popqa bin3 | 314 | 0.752 | 0.911 | 0.936 | 0.837 | 0.723 / 0.895 / 0.927 |
| popqa bin4 (most popular) | 286 | 0.724 | 0.923 | 0.941 | 0.823 | 0.654 / 0.895 / 0.927 |
| multihop | 87 | 0.931 | 0.966 | 0.966 | 0.951 | both gold @3 / @5: 0.724 / 0.805 |
| multihop bridge | 41 | 0.878 | 0.927 | 0.927 | 0.908 | both gold @3 / @5: 0.463 / 0.610 |
| multihop comparison | 46 | 0.978 | 1.000 | 1.000 | 0.989 | both gold @3 / @5: 0.957 / 0.978 |

"Primary" counts only the first alias-bearing passage of the subject
article. "Both gold" requires both HotpotQA supporting passages in the top k.
Per-relation numbers are in `index/eval_dev.json` after a local build; the
hardest relation at hit@1 is `capital` (0.59, since city articles mention
many places).

Reading: retrieval is non-trivial but mostly successful (5% of PopQA dev
questions miss at k = 5). hit@1 **falls** with popularity (0.87 → 0.72):
popular subjects have long articles and many same-name or same-relation
neighbours, while obscure subjects have short, distinctive articles. So
retrieval quality and the closed-book prior move in opposite directions
along the popularity axis, which E1-v2/E3-v2 analyses should keep in mind.
Bridge questions are the hard multihop case: only 61% get both passages
in the top 5 from the full written question.

Index build: 49,694 vectors, 4,928 s of embedding on 8 CPU threads on a
shared machine (streamed in blocks of 2,048; peak RSS about 1 GB);
`index.faiss` 76 MB and `chunks.json` 62 MB, both gitignored.

## Rebuild

From the repo root, on Windows with `.venv-e2` (the build itself needs only
`requests`; the index needs faiss and sentence-transformers):

```bat
set PYTHONUTF8=1
rem 1. dataset (network: ~8.7k polite serial API requests, ~80 min; raw responses cached in .cache/)
.venv-e2\Scripts\python datasets\v2\build_d1_open.py
rem    re-run from cache only (byte-identical, ~5 min)
.venv-e2\Scripts\python datasets\v2\build_d1_open.py --offline
rem    pilot build with 1,000 PopQA items
.venv-e2\Scripts\python datasets\v2\build_d1_open.py --n-popqa 1000 --out <dir>
rem 2. FAISS index + dev retrieval check (~80 min on 8 CPU threads; streamed in blocks)
.venv-e2\Scripts\python retrieval\build_index_v2.py --corpus datasets\v2\d1_open\corpus.jsonl ^
    --eval-items datasets\v2\d1_open\items.jsonl --split dev
```

Reproducibility notes:

- With `.cache/` present, `--offline` reproduces `items.jsonl`,
  `corpus.jsonl` and `splits.json` byte for byte. This was verified:
  `corpus.jsonl` sha256 is `724624b9…` (full hash in `stats.json`).
- From an empty cache, the build refetches **current** Wikipedia revisions,
  so a few passages, and rarely an item's gold check, can differ. Each
  passage's `revid` records the exact revision used here. The committed
  files are the reference.
- The HF rows API serves HotpotQA as JSON, which avoids pyarrow; the dataset
  revision at build time was `1908d6af`. `--hotpot-json
  hotpot_dev_distractor_v1.json` uses the official file instead and yields
  the same sample, because sampling is keyed on HotpotQA `_id`.
- Requests use a descriptive User-Agent, run strictly serially with ≥ 0.2 s
  spacing, set `maxlag=5` and back off on 429/5xx.

## Known limitations

- Equal counts per popularity bin leave relations uneven across bins
  (`capital` concentrates in bin 4, `country`/`sport`/`place of birth` in
  bin 0). Analyses that condition on popularity should include relation as
  a covariate.
- `genre` and `occupation` answers are one of several valid values
  (`meta.multi_valued_answer`). Closed-book scoring against aliases will
  under-count correct answers for these items.
- The gold check requires the answer to appear in the article text. Facts
  that exist only in infoboxes (which TextExtracts omits) fail the check, so
  the sample slightly favours subjects whose article states the answer in
  prose.
- Hard negatives are lead sections only. Gold articles contribute body
  passages, but negatives do not, except where a negative is another item's
  subject article (common: 21,603 passages carry a negative role).
