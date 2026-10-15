# D2 `d2_support`: multi-domain customer-support voice benchmark

D2 holds the questions callers put to a customer-service or service-desk voice
agent, across 12 industries. For each question, whether the agent needs its
company knowledge base (KB) is hard to tell from the wording alone. D2 replaces
the v1 data. That data could be separated from surface form alone: every
retrieval-needed question named a fictional hotel, and every other question was
school-level trivia. As a result, the model's uncertainty was close to binary.

## Construction (stated plainly)

- **Synthetic, written for this benchmark.** Every organisation, KB document,
  question and answer was written for D2. Nothing was collected from real
  callers or copied from real companies. All 24 organisations, and their
  products, places and phone numbers, are fictional. Names were chosen so that
  no language model could have memorised them. Any resemblance to a real
  company is unintended.
- **Two halves, one spec.** Part A covers telecom, banking, insurance,
  e-commerce, airline and healthcare. Part B covers utilities, SaaS, food
  delivery, education, government and automotive. Both were written in
  parallel from the same specification. Part B's interpretation notes are in
  `raw/PART_B_NOTES.md`.
- **Human verification is pending.** `verification_sample.csv` is a stratified
  20% sample with blank columns for a reviewer: is the label correct, is the
  answer correct, and notes. Until that review is done, all labels count as
  *authored* labels.
- **Labels.** `retrieval_needed_authored` is the author's judgement.
  Experiment E1-v2 produces the empirical label (wrong when answered closed-book
  across draws, with Mistral-7B-Instruct-v0.3). That label is never set by
  hand (CONTRACTS §2).
- **Automated validation.** `build_d2_support.py` refuses to build when any
  check fails:
  - schema
  - every number, amount, date, time and capitalised name in a key fact
    appears verbatim in a gold document
  - pair-group integrity: exactly one general item per group
  - question lengths match within ±3 words inside each pair group
  - no organisation name in any question or dialogue history
  - no near-duplicate questions
  - document length of 120–350 words

  The current build has 0 errors and 46 advisory warnings, mostly general
  answers that mention a generic term also found in a KB. Each warning is
  listed in `validation_report.txt`.

## Domains and organisations

| domain | organisation (slug) | setting | docs |
|---|---|---|---:|
| telecom | Sarovan Telecom (`sarovan_telecom`) | India, mobile + home fibre | 15 |
| telecom | Tallowood Connect (`tallowood_connect`) | Australia, mobile + home internet | 15 |
| banking | Saffronleaf Bank (`saffronleaf_bank`) | India, retail bank | 14 |
| banking | Quaymark Money (`quaymark_money`) | UK, app-only bank | 14 |
| insurance | Tulsivan Health Insurance (`tulsivan_health`) | India, health insurer | 14 |
| insurance | Liwan Crest Insurance (`liwan_crest_insurance`) | UAE, motor/home/travel | 14 |
| ecommerce | Tamarind Lane (`tamarind_lane`) | India, fashion marketplace | 14 |
| ecommerce | Brackenfield Home (`brackenfield_home`) | UK, home and electricals retailer | 15 |
| airline | Saffron Kite Air (`saffron_kite_air`) | India, low-cost carrier | 15 |
| airline | Coralwind Airways (`coralwind_airways`) | Australia, full-service airline | 14 |
| healthcare | Varunika Hospitals (`varunika_hospitals`) | India, multi-specialty hospital | 15 |
| healthcare | Wrenfield Clinics (`wrenfield_clinics`) | UK, private GP and diagnostics | 15 |
| utilities | Vellora Energy Distribution Ltd (`vellora`) | India, electricity distributor | 14 |
| utilities | Harrowmere Water & Gas (`harrowmere`) | UK, water and gas supplier | 15 |
| saas | Quorvane (`quorvane`) | international, project-management SaaS | 15 |
| saas | Ledgerwick Books (`ledgerwick`) | India, accounting / GST invoicing SaaS | 15 |
| food_delivery | Tiffinora (`tiffinora`) | India, food delivery with tiffin plans | 15 |
| food_delivery | Brambleway Eats (`brambleway`) | UK, food and grocery delivery | 15 |
| education | Vidyanjali University (`vidyanjali`) | India, private university admissions | 15 |
| education | Ashcombe Polytechnic Institute (`ashcombe`) | Canada, polytechnic admissions | 15 |
| government | Kestrel Bay County Council (`kestrelbay`) | UK, county council | 15 |
| government | Kesarinagar Municipal Corporation (`kesarinagar`) | India, municipal corporation | 15 |
| automotive | Arunodaya Motors (`arunodaya`) | India, dealership and service | 15 |
| automotive | Corrigan & Pike Auto Group (`corriganpike`) | US, dealer group | 15 |

## Item types

| type | `retrieval_needed_authored` | definition |
|---|---|---|
| `org_specific` | true | The answer exists only in this organisation's KB, for example fees, timelines, plan limits, procedures, hours or eligibility. |
| `counter_prior` | true | The organisation's policy deliberately differs from the industry norm, so a model's confident prior answer is wrong. For example, refunds take 21 working days instead of 5–7. The norm is recorded in `meta.common_prior` (part A) or `meta.note` (part B). |
| `ambiguous` | true | General knowledge gives an answer that sounds right but is incomplete or wrong. The precise organisation-specific detail is needed. The tempting answer or rationale is recorded in `meta.plausible_wrong_answer` (part A) or `meta.note` (part B). |
| `general` | false | Domain knowledge that a competent agent answers correctly without the KB, asked as a caller would ask this organisation. The answer does not depend on any organisation-specific value. A few general items rest on country-wide rules that match the organisation's setting, such as a national cheque validity period or UK consumer law; these cases are noted in the raw notes. |

Per organisation the mix is either:
- 10 / 5 / 3 / 12 (part A), or
- 10 / 6 / 3 / 11 (part B),

for `org_specific` / `counter_prior` / `ambiguous` / `general`. Every item
carries `meta.key_facts`: the facts a correct answer must contain and must not
contradict. For retrieval-needed items these are copied from the gold document.

**Pair groups.** Each `pair_id` names one pair group, which holds:

- exactly one `general` item, and
- one or two retrieval-needed items from the same organisation, with the same
  intent or question template.

Each retrieval-needed question is within ±3 words of its group's general
question. Strict 1:1 pairs are impossible at roughly 62% retrieval-needed
items, so groups are the matched unit. Analyses can form (general,
retrieval-needed) pairs inside each group. Callers never name the
organisation, because in a real call the agent already knows which company it
represents. The `org` field carries the organisation instead.

**History.** 100 items (13.9%) carry a short `history` of earlier turns that
the question depends on. It is a list of
`{"role": "user" | "assistant", "content": ...}`, ending with an assistant
turn (CONTRACTS §2).

KB documents include realistic mess:

- superseded and current versions of a policy, with effective dates. The gold
  document is always the current one; for part A, `meta.superseded_doc_id`
  names the distractor.
- near-duplicate documents that differ in one key detail, such as two plan
  tiers or two branches.
- documents that no item uses. Retrieval runs over the whole 353-document
  corpus, so the other organisation in the same domain also acts as a
  distractor.

## System prompt template

Every model call on D2 uses this system prompt. `meta.system_prompt` stores
the filled-in version for each item.

```
You are a customer support voice agent for {org}, a {domain} company. Answer the caller briefly.
```

`{domain}` is the readable label stored in `meta.domain_label`, for example
"telecom (mobile and broadband)", "e-commerce and retail" or "government and
citizen services". For a multi-turn item, the `history` turns go between the
system prompt and the question as user and assistant messages.

## Files

| file | content |
|---|---|
| `raw/<domain>/<org_slug>/docs.jsonl` | KB documents written for one organisation (CONTRACTS §3) |
| `raw/<domain>/<org_slug>/items.jsonl` | items for one organisation, with local ids and local pair ids, no `split` |
| `raw/PART_B_NOTES.md` | part B's interpretation notes |
| `items.jsonl` | built items (CONTRACTS §2), with global ids `d2_support-DDOSS` and pair ids `d2-pair-DDOPP`; `meta` adds `local_id`, `local_pair_id`, `org_slug`, `domain_label`, `system_prompt`, `question_words` |
| `corpus.jsonl` | all KB documents (353) |
| `splits.json` | dev and test ids, plus the split method |
| `stats.json` | counts per domain, organisation, type and split; length statistics |
| `validation_report.txt` | output of the validator |
| `verification_sample.csv` | stratified sample for human review: 138 items, 19% |
| `leak_test.json` | final leak test on the built `items.jsonl` |
| `leak_test_history/` | leak-test results before and after each revision round |
| `revision_log.tsv` | every question rewording made to pass the leak test |

Global ids are stable. In `DDOSS`, DD is the domain's position in a fixed
domain order, O is the organisation's position within its domain (alphabetical
slug), and SS is the local item order. Adding domains or organisations never
renumbers existing items.

## Splits

The split is deterministic, about 30% dev and 70% test (CONTRACTS §7):

- **Unit of assignment.** The pair group, so all members always share a split.
- **Allocation.** Each organisation gets `round(0.3 · n_groups)` dev groups,
  spread across type strata by largest remainder. The stratum is the rarest
  retrieval-needed type in the group: ambiguous > counter_prior >
  org_specific.
- **Selection.** Within a stratum, the groups with the lowest unsalted
  `sha256(pair_id)` go to dev.

Allocation never looks outside an organisation, so merging new data never
changes existing assignments. The result is 222 dev and 498 test items (30.8%
dev); see `stats.json` for the split by type and by organisation. **The test
split is sealed**: no generation, signal extraction, tuning or judging until
the v2 pre-registration addendum is frozen.

## Surface-form leak test

`leak_test_d2.py` predicts `retrieval_needed_authored` from the question alone.

- **Features.** TF-IDF word 1–2-grams, TF-IDF char 2–5-grams, and length
  features (words, characters, has-history).
- **Model.** Logistic regression, with 5-fold cross-validation grouped by
  pair group. Vectorisers are fit inside each fold.
- **Reported AUC.** The pooled out-of-fold ROC AUC. The headline is the worst
  case over C ∈ {0.1, 1, 10}, averaged over three fold seeds.
- **Extra checks.** Word-only, char-only, length-only and question+history
  variants. A leave-one-domain-out (LODO) run measures cues that carry across
  domains.
- **Target.** AUC ≤ 0.65.

| stage | items | headline AUC [95% group-bootstrap CI] | LODO |
|---|---:|---|---:|
| part A, first draft | 360 | 0.688 [0.643, 0.743], fail | – |
| part A, after rewording round 1 | 360 | 0.625 [0.578, 0.687] | 0.702 |
| part A, after rewording round 2 | 360 | 0.582 [0.533, 0.638] | 0.641 |
| parts A+B combined, before combined revision | 720 | 0.678 [0.649, 0.722], fail | 0.693 |
| combined, after round C1 (raw files) | 720 | 0.597 [0.565, 0.641] | – |
| **final, built `items.jsonl`** | **720** | **0.576 [0.541, 0.617], pass** | **0.610** |

The last two rows contain the same items; they differ only in how the CV folds
fall. The worst single variant on the final build is question+history at
0.583.

**What leaked, and how it was fixed.** In the first drafts:
- general items over-used "why", "what does X mean", "difference between" and
  "what's a";
- retrieval-needed items over-used "how much", "can I", "cancel", "fee",
  "days" and "time";
- later, content framing leaked: retrieval items were about the caller's own
  transaction (cost, deposit, limit, this week), general items about concepts
  and third parties (a code, the bank, GST).

Questions were reworded, not re-targeted, so that each cue appears in both
classes at similar rates. Every item kept its type, pair group and intent, and
answers were adjusted only where the wording changed what was asked.
`revision_log.tsv` records all 279 rewordings with the old and new question,
whether the answer changed, and the round: 1 and 2 are part-A-only passes, and
C1 is the combined pass. Part B's own pre-merge revisions are summarised in
`raw/PART_B_NOTES.md`.

**Final per-domain AUC** (pooled out-of-fold scores):

| domain | AUC |
|---|---:|
| automotive | 0.52 |
| healthcare | 0.53 |
| telecom | 0.53 |
| government | 0.54 |
| airline | 0.56 |
| saas | 0.57 |
| ecommerce | 0.58 |
| utilities | 0.58 |
| food_delivery | 0.61 |
| banking | 0.63 |
| insurance | 0.66 |
| education | 0.67 |

Education and insurance are the weakest spots, with n = 60 each.

By item type against general:

| comparison | AUC |
|---|---:|
| org_specific vs general | 0.59 |
| counter_prior vs general | 0.56 |
| ambiguous vs general | 0.57 |

The test is not a proof of non-separability. It is a fixed, bag-of-n-grams
check, and the items were revised with knowledge of what it detects. Some
residual signal is semantic and arguably intrinsic: organisation-specific
questions are more often about the caller's own costs and deadlines.

## Statistics

- **Size.** 720 items, 24 organisations in 12 domains, 353 KB documents. The
  documents average 207 words (range 132–289).
- **Types.**

  | type | items |
  |---|---:|
  | `org_specific` | 240 |
  | `counter_prior` | 132 |
  | `ambiguous` | 72 |
  | `general` | 276 |

  That is 444 retrieval-needed items (61.7%) and 276 general items.
- **Pair groups.** 276 groups: 108 with 2 items and 168 with 3.
- **Question length.** Retrieval-needed: mean 15.0 words (sd 2.1). General:
  mean 15.1 words (sd 2.1).
- **Splits.**

  | type | dev | test |
  |---|---:|---:|
  | `org_specific` | 76 | 164 |
  | `counter_prior` | 38 | 94 |
  | `ambiguous` | 24 | 48 |
  | `general` | 84 | 192 |

  Each organisation has 7–12 dev items out of 30.

## Rebuild

From the repository root:

```bash
PYTHONUTF8=1 python datasets/v2/build_d2_support.py --check-only       # validate only
PYTHONUTF8=1 python datasets/v2/build_d2_support.py                    # validate and write all outputs
PYTHONUTF8=1 python datasets/v2/leak_test_d2.py                        # leak test on the built items -> leak_test.json
PYTHONUTF8=1 python -m pytest tests/test_build_d2_support.py           # builder unit tests
```

The builder uses only the standard library. The leak test needs scikit-learn
and scipy, and does not need pyarrow, so system Python 3.12 is fine; it pins
BLAS to one thread. The build is deterministic: the same raw files always give
byte-identical outputs.

## Licence

Proposed licence: Creative Commons Attribution 4.0 International (CC BY 4.0)
for all documents, items and annotations in this directory.
