# d2_support raw data, part B

Part B covers six of the twelve D2 domains. Part A covers telecom, banking &
fintech, insurance, e-commerce & retail, airline & travel and healthcare front
desk, and owns the builder (`datasets/v2/build_d2_support.py`) and the official
leak test.

## Domains and organisations

All organisations, places, products, plans and phone numbers are invented.
The organisation is never named by the caller (validated: no distinctive word
of the org name appears in any question or history turn).

| domain (`domain` field) | org_slug | org | setting |
|---|---|---|---|
| utilities | vellora | Vellora Energy Distribution Ltd | Indian electricity distributor (₹) |
| utilities | harrowmere | Harrowmere Water & Gas | UK-style water, wastewater and gas supplier (£) |
| saas | quorvane | Quorvane | international project-management SaaS ($) |
| saas | ledgerwick | Ledgerwick Books | Indian accounting / GST invoicing SaaS (₹) |
| food_delivery | tiffinora | Tiffinora | Indian food-delivery app with tiffin plans (₹) |
| food_delivery | brambleway | Brambleway Eats | UK food and grocery delivery (£) |
| education | vidyanjali | Vidyanjali University | Indian private university admissions desk (₹) |
| education | ashcombe | Ashcombe Polytechnic Institute | Canadian polytechnic in an invented Manitoba town, domestic and international admissions (CAD) |
| government | kestrelbay | Kestrel Bay County Council | UK-style county council (£) |
| government | kesarinagar | Kesarinagar Municipal Corporation | Indian municipal corporation (₹) |
| automotive | arunodaya | Arunodaya Motors | Indian dealership and service network (₹) |
| automotive | corriganpike | Corrigan & Pike Auto Group | US dealer group in two invented towns: sales, certified pre-owned, service, collision (USD) |

## Counts

Every org has 30 items: 10 `org_specific`, 6 `counter_prior`, 3 `ambiguous`,
11 `general`, in 11 pair groups.

| domain | org_slug | docs | items | with history |
|---|---|---:|---:|---:|
| utilities | vellora | 14 | 30 | 5 |
| utilities | harrowmere | 15 | 30 | 4 |
| saas | quorvane | 15 | 30 | 4 |
| saas | ledgerwick | 15 | 30 | 4 |
| food_delivery | tiffinora | 15 | 30 | 4 |
| food_delivery | brambleway | 15 | 30 | 5 |
| education | vidyanjali | 15 | 30 | 4 |
| education | ashcombe | 15 | 30 | 5 |
| government | kestrelbay | 15 | 30 | 4 |
| government | kesarinagar | 15 | 30 | 4 |
| automotive | arunodaya | 15 | 30 | 5 |
| automotive | corriganpike | 15 | 30 | 4 |
| **total** | 12 orgs | **179** | **360** | **52 (14%)** |

Part B totals by type: 120 `org_specific`, 72 `counter_prior`, 36 `ambiguous`,
132 `general`. That is 228 retrieval-needed and 132 not-needed items, in 132
pair groups.

## Files

```
raw/<domain>/<org_slug>/docs.jsonl    corpus schema (CONTRACTS §3)
raw/<domain>/<org_slug>/items.jsonl   item schema (CONTRACTS §2), no `split`
validate_raw_part_b.py                part-B validator
leak_check_part_b.py                  part-B surface-form leak check
```

## Spec interpretation decisions

1. **Type mix per org (30 items):** 10 `org_specific`, 6 `counter_prior`,
   3 `ambiguous`, 11 `general` (33 / 20 / 10 / 37 percent).
2. **Pairing.** With 19 retrieval-needed and 11 general items per org, strict
   1:1 pairs are impossible. Each `pair_id` is a group with exactly one
   `general` item and one or two needed items on the same intent/template
   (8 groups of 1+2 and 3 groups of 1+1 per org, so 11 groups per org).
   Every needed item is within ±3 words of its group's general item
   (word = `[A-Za-z0-9']+` token). The builder should keep a whole group in
   one split. `pair_id` = `<org_slug>-pair-NN`.
3. **Ids.** Local item ids are `<org_slug>-NNN`; doc ids are
   `d2-<domain>-<org_slug>-<topic>-NN`. The builder assigns the global
   `d2_support-NNNNNN` ids and `split`.
4. **History format.** `history` is a list of
   `{"role": "user" | "assistant", "content": "..."}` turns, ending with an
   assistant turn; the question depends on it. 4-5 items per org (13-17%)
   carry history, spread over both classes (at least 2 needed and 2 general).
5. **Key facts.** `meta.key_facts` for needed items are literal substrings
   (case- and whitespace-insensitive) of at least one gold doc, so the doc's
   own wording is used (e.g. "₹125", "within 6 hours"). For `general` items the
   key facts are short general facts and are not checked against the KB.
   `meta.note` (optional) records, for `counter_prior` items, the norm the org
   deviates from, and for `ambiguous` items, why a generic answer falls short;
   it also flags superseded/near-duplicate distractor docs.
6. **Gold docs** always point at the current doc, never the superseded one.
7. **Corpus fields.** `source` = "authored", `licence` = "CC-BY-4.0",
   `dataset` = "d2_support". Docs are 120-350 words of plain prose.
8. **KB messiness.** Each org has at least one superseded-vs-current pair
   with effective dates (the old doc says "superseded"/"replaces") and at
   least one near-duplicate distractor that differs in a key detail (e.g.
   domestic vs commercial connection charges, two recycling centres with
   different closing days, home vs office tiffin plans).
9. **Spoken style.** Questions and history have no digits, symbols or
   markdown (numbers are written as spoken); expected answers may use digits
   and currency symbols.
10. **General items** are answerable without the KB and do not contradict
    it. A few rely on country-standard facts that match the org's setting
    (e.g. UK household water cannot be disconnected for non-payment; Indian
    GST invoice numbers are at most 16 characters). These are noted here
    because they are jurisdiction-dependent.

11. **Counter-prior strength varies.** The deviations were written to be
    sharp (e.g. net billing instead of net metering, no overdue library
    fines, guests with edit rights billed as full seats). A few are softer,
    where the "norm" is common rather than universal (e.g. per-workspace API
    rate limits, refundability of a factory-order deposit). The empirical
    label from E1-v2 (Mistral-7B-Instruct-v0.3, closed-book) decides which
    items actually need retrieval; the authored label is only the prior.
12. **Uncovered docs.** Some KB docs have no item pointing at them (e.g. a
    marriage-registration doc, an allergen policy). They are kept on purpose
    as realistic retrieval noise.

## Validation

`python datasets/v2/d2_support/validate_raw_part_b.py` checks schema and field
types, per-org counts and type mix, doc length, gold-doc existence, literal
key-fact support in gold docs, pair-group integrity and the ±3-word length
match, that no org name appears in questions or history, spoken style (no
digits or markdown in questions), and exact and near-duplicate questions.
Final result: **0 errors, 0 warnings** over all 12 orgs.

## Surface-form leak check

`python datasets/v2/d2_support/leak_check_part_b.py` fits a logistic
regression on TF-IDF word 1-2 grams of the question plus length and history
features to predict `retrieval_needed_authored`. It is scored by
cross-validation grouped by `pair_id` (StratifiedGroupKFold, 5 folds, 10
shuffles; pooled out-of-fold ROC AUC). Target: ≤ 0.65. On Windows, set
`OPENBLAS_NUM_THREADS=1` if numpy runs out of memory.

| set | items | AUC (mean ± sd) |
|---|---:|---:|
| first draft, 8 orgs (before revision) | 240 | 0.765 ± 0.012 |
| same 8 orgs, final | 240 | 0.569 ± 0.021 |
| all 12 orgs, before the final pooled pass | 360 | 0.684 ± 0.014 |
| **all 12 orgs, final** | **360** | **0.634 ± 0.016** |

Final per-domain AUCs: utilities 0.50, saas 0.53, food_delivery 0.44,
education 0.36, government 0.56, automotive 0.36 (n = 60 each). Values below
0.5 come from small-sample over-correction within a domain, not from a real
inverse signal.

What leaked in the first draft, and how it was fixed: general items used
"should", "what does ... mean", "difference", "same thing as", "safe" and
"is it okay" far more than needed items, while needed items used "how much",
"get", "fee", "year", "back", "do you" and "should I expect". Questions were
rephrased (not re-targeted) so that each cue appears in both classes at
roughly the class ratio, and the items the model scored most confidently
were rewritten with the other class's framing. The strongest remaining
features are weak function and topic words (|coef| ≈ 0.6, e.g. "is the",
"week", "they", "gst").
