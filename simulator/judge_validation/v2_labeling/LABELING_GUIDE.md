# Labelling guide: is the support agent's answer correct? (rubric r2)

**Your sheet:** `sheet_annotator_A.csv` or `sheet_annotator_B.csv`. Each sheet
has 300 rows, which takes about 2 hours. Label on your own: do not compare
notes with the other annotator, look up company documents, or use an AI tool.
Each row shows a caller's question to a customer-support voice agent at a
fictional company. Where there was earlier conversation, it is shown first.
Each row also has the **reference answer**, its **key facts**, and the
agent's **candidate answer**.

Fill in **`verdict`** with exactly `correct` or `incorrect`. Use **`notes`**
for anything borderline; a few words is enough.

## The rule

The candidate is **correct** only if **both** of these hold:

1. **It states every key fact.** Any wording counts, as do equivalent units
   and formats: "₹50" = "50 rupees" = "Rs 50"; "48 hours" = "two days";
   "9 AM" = "09:00"; "13 amp" = "13 A".
2. **It contradicts nothing in the reference.** That means no different
   figure, time, name, price, place, condition, limit or exception.

Otherwise it is **incorrect**. In particular:

- **Deferring** without stating the key facts is incorrect: "please contact
  us", "check your policy documents", "it depends on your plan", "our team
  will review it".
- **Declining**, "I don't know", or answering a different question is
  incorrect.
- **A vague or hedged answer** that leaves out a key fact is incorrect, even
  if nothing it says is false.
- **Extra detail** is fine if it is true or the reference does not
  contradict it.
- **Wording, length, tone, politeness, apologies, and naming the company** do
  not matter.

Judge only what the answer **says**. Ignore whether the agent could have known
it, and ignore how good the reference is. If you think the reference itself is
wrong, still judge against it and say so in `notes`.

## Worked examples (fictional company "Kestrel Mobile")

| # | Key facts | Candidate answer | Verdict | Why |
|---|---|---|---|---|
| 1 | A replacement SIM costs ₹50 · issued in-store with original ID | "It's fifty rupees, and you'll need to bring your original ID to one of our stores." | correct | Both facts stated, in other words. |
| 2 | Same as 1 | "A replacement SIM is ₹50." | incorrect | The ID/in-store fact is missing. |
| 3 | Same as 1 | "Please visit our website or call us to check the current SIM replacement charges." | incorrect | Defers without stating any fact. |
| 4 | Same as 1 | "A replacement SIM costs ₹100 at any store with your ID." | incorrect | Contradicts the price. |
| 5 | Burst pipes are covered · excess of AED 500 · not covered if the home was empty more than 60 days | "Our policy typically covers burst pipes, so you should be fine." | incorrect | A hedge: the excess and the exclusion are missing. |
| 6 | A washing machine needs a 13 amp fuse · a fuse that blows again suggests a fault | "Use a 13 A fuse. If it blows again, stop using the machine, as it probably has a fault; and always switch off at the wall first." | correct | Both facts stated; the extra advice is true. |
| 7 | Same as 6 | "It should be a 13 amp fuse." | incorrect | The second key fact is missing. |
| 8 | Data allowance of the 7-day pack is 5 GB | (after earlier talk about roaming packs) "The seven-day pack gives you 5 GB of data to use abroad." | correct | Read the earlier conversation to see which pack is meant. |

## Borderline cases

- **Partly-matching numbers:** "about ₹50" against ₹50 is correct; "₹40–60"
  is incorrect, because it does not state the figure.
- **Hedges with the facts present:** "I believe it's ₹50 with your ID" is
  correct; the hedge does not remove a stated fact.
- **Breadth:** a fact stated more broadly than the key fact ("bring ID"
  where the key fact is "original ID") is correct unless it changes the
  meaning. Note it.
- **Unsure:** pick your best verdict and write `unsure` in `notes`. Never
  leave `verdict` blank.

Disagreements between the two sheets go to a third person
(`adjudication_needed.csv` from `merge_sheets.py`).
