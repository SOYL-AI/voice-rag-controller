# Labelling guide: is the support agent's answer correct? (rubric r3)

**Your sheet:** `sheet_annotator_A.csv` or `sheet_annotator_B.csv`. Each sheet
has 300 rows, which takes about 2 hours. Label on your own: do not compare
notes with the other annotator, look up company documents, or use an AI tool.

Each row shows a caller's question to a customer-support voice agent at a
fictional company, with any earlier conversation shown first. It also shows:
- the **reference answer**;
- its **core facts**: the information that actually answers the question;
- its **supplementary facts**: useful extra detail;
- the agent's **candidate answer**.

Fill in **`verdict`** with exactly `correct` or `incorrect`. Use **`notes`**
for anything borderline; a few words is enough.

## The rule

The candidate is **correct** only if **both** of these hold:

1. **It states every core fact.** Any wording counts, as do equivalent units
   and formats and close approximations: "₹50" = "50 rupees" = "Rs 50" =
   "about ₹50"; "48 hours" = "two days"; "9 AM" = "09:00"; "13 amp" = "13 A".
2. **It contradicts nothing** in the reference or the supplementary facts.
   That means no different figure, time, name, price, place, condition,
   limit or exception.

**Supplementary facts may be left out.** They only count against the answer
when the answer contradicts them.

Otherwise it is **incorrect**. In particular:
- **Deferring** without stating the core facts is incorrect: "please contact
  us", "check your policy documents", "it depends on your plan", "our team
  will review it".
- **Declining**, "I don't know", or answering a different question is
  incorrect.
- **A vague or hedged answer** that leaves out a core fact is incorrect, even
  if nothing it says is false.
- **Extra detail** is fine if it is true or nothing in the reference
  contradicts it.
- **Wording, length, tone, politeness, apologies, and naming the company** do
  not matter.

Judge only what the answer **says**. Ignore whether the agent could have known
it, and ignore how good the reference is. If you think the reference itself is
wrong, still judge against it and say so in `notes`.

## Worked examples (fictional company "Kestrel Mobile")

| # | Core facts | Supplementary facts | Candidate answer | Verdict | Why |
|---|---|---|---|---|---|
| 1 | A replacement SIM costs ₹50 | Issued in-store with original ID | "It's about fifty rupees for a new SIM." | correct | Core fact stated as an approximation; supplementary fact left out. |
| 2 | Same as 1 | Same as 1 | "It's ₹50, and you can get it delivered to your home." | incorrect | Core fact stated, but home delivery contradicts "issued in-store". |
| 3 | Same as 1 | Same as 1 | "Please check our website for the current SIM replacement charges." | incorrect | Defers without the core fact. |
| 4 | Same as 1 | Same as 1 | "A replacement SIM costs ₹100." | incorrect | Contradicts the core fact. |
| 5 | Burst pipes are covered · an excess of AED 500 applies | Not covered if the home was empty more than 60 days | "Our policy typically covers burst pipes, so you should be fine." | incorrect | A hedge: a core fact (the excess) is missing. |
| 6 | Same as 5 | Same as 5 | "Yes, burst pipes are covered, with a 500-dirham excess." | correct | Both core facts stated; the exclusion may be left out. |
| 7 | A washing machine needs a 13 amp fuse | A fuse that blows again suggests a fault | "It should be a 13 A fuse." | correct | The core fact is stated; supplementary advice left out. |
| 8 | Same as 7 | Same as 7 | "Use a 5 amp fuse for that." | incorrect | Contradicts the core fact. |
| 9 | The 7-day roaming pack has 5 GB of data | Valid in 40 countries | (after earlier talk about roaming packs) "The seven-day pack gives you 5 GB." | correct | Read the earlier conversation to see which pack is meant. |

## Borderline cases

- **Ranges and approximations:** "about ₹50" for ₹50 is correct; "₹40–60" is
  incorrect, because it does not state the figure.
- **Hedges with the facts present:** "I believe it's ₹50" is correct; the
  hedge does not remove a stated fact.
- **Breadth:** a fact stated more broadly than the core fact ("bring ID"
  where the core fact says "original ID") is correct unless it changes the
  meaning. Note it.
- **"Why" questions with several reasons:** often only the main reason is a
  core fact, and the other reasons are supplementary. Mark the answer
  **correct only if it gives the core reason**. An answer that gives only
  other (true) reasons is **incorrect**, even when those reasons appear in
  the reference. If you disagree with which reason was made core, still
  follow this rule and write `core reason?` in `notes`. We measure whether
  this rule is contentious through the agreement between the two sheets.
- **Unsure:** pick your best verdict and write `unsure` in `notes`. Never
  leave `verdict` blank.

Disagreements between the two sheets go to a third person
(`adjudication_needed.csv` from `merge_sheets.py`).
