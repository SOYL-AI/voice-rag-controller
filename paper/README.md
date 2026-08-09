# paper/ — Documentation & Results

**Owner: Yashmitha**  
This README covers Week 1. The heavier work (results compilation, reviewer responses) 
grows in later weeks and will be detailed here as we go.

## What this folder is

The written backbone of the project: the paper draft, the record of what we decided and 
why, and (later) the compiled results that go into the paper.

## Why it matters

This is the credibility role, not an admin role. The most important thing you own is the 
pre-registration document — a simple record of what each experiment will measure and what 
counts as success, written *before* we run it. That is exactly what makes reviewers trust 
our results: it proves we did not shape the goals after seeing the data. Without it, good 
results look suspicious.

Your load is lighter in Week 1 and grows later. In the final weeks, when the engineers are 
done building and just running experiments, you are the one compiling all the results, 
building the tables, and driving the paper. So Week 1 is your runway to get set up and 
comfortable.

## Week 1 goal (your deliverable)

1. A running progress log where the team's dated decisions are recorded (`lab_log.md`).
2. The pre-registration document, started (`preregistration.md`). The integration lead 
   will scaffold the first version with the experiment list; you take it over and keep it 
   current.
3. The empty paper skeleton with section headings (`paper.md` or a LaTeX file).

## What "pre-registration" actually means (in plain terms)

It is just writing down, before an experiment runs, two things:
- what we will measure (e.g. "does the model's uncertainty predict when it needs to look 
  something up?"), and
- what counts as success (e.g. "a score of 0.7 or higher").

That is it. No special format, no legal anything. "Signed off" just means the model/signals 
owner reads it and agrees "yes, this is what we are doing." You keep this document current 
as experiments get defined; the integration lead helps with the technical parts.

## How to start (step by step)

### Step 1 — Open the lab log (`lab_log.md`)

A running file. Each entry is a date and a decision, e.g.:2025-XX-XX — Chose flat FAISS index over IVF for Week 1 (small corpus). Decided by team.
Whenever the team makes a real decision (a scope cut, a tool choice, an experiment 
verdict), record it here. This becomes invaluable later when we write the paper and need to 
remember why we did things.

### Step 2 — Start the pre-registration doc (`preregistration.md`)

The integration lead will give you a first draft listing the experiments and their success 
criteria. Your job is to hold it, keep it readable, and update it as things get decided. Do 
not worry about filling in the technical details yourself — that comes from the integration 
lead and the model/signals owner.

### Step 3 — Create the paper skeleton

Make an empty paper file with the standard section headings in place: Abstract, 
Introduction, Related Work, Method, Experiments, Results, Discussion, Limitations, 
Conclusion. Just the headings for now — content comes later. (We will confirm the exact 
venue template once the model/signals owner tells us the target conference.)

## Optional Week 1 bonus (if you have spare time)

Read the two closest prior papers and write a short plain-English summary of each for the 
team: "Stream RAG" and "SpokenWOZ". This feeds directly into the Related Work section 
later, and helps everyone understand what has already been done. Ask the integration lead 
for links.

## Not in Week 1 (comes later)

- Results compilation and tables (Weeks 3–4, once experiments produce numbers).
- The reviewer-response table (Week 4).
- Citation verification (later).

## Definition of done (Week 1)

Lab log opened, pre-registration doc started (with the integration lead's scaffold), and 
the paper skeleton created with section headings.

## Coordinate with

The integration lead (scaffolds the pre-registration doc and provides paper links) and the 
model/signals owner (signs off on the pre-registration once experiments are defined).

---

## Phase 2 — Documentation and Results Compilation

In Phase 1 you set up the records (lab log, paper skeleton, pre-registration doc). In Phase 
2 this role ramps up: keep the records complete and compile the experiment results into 
clean, organized form so they are ready to use.

### Important: who writes the paper

Ryan writes the paper itself. This role does not draft the prose. Instead, it keeps the 
records and compiles and organizes the material so that when Ryan writes, everything he 
needs (results, references, figures, decisions) is ready in one place.

### Goal

Keep the pre-registration doc and lab log current, and compile experiment results into 
organized tables and summaries as they land.

### How to do it

- Keep the pre-registration document current: as each experiment's success criteria get 
  finalized, record them, and get Ryan to sign off before results exist. This matters because 
  pre-committing criteria is what makes the results credible.
- Compile results as Experiment 1 and 2 produce numbers: organize them into clean tables and 
  summaries, ready for Ryan to use. This is the main ramp-up of the role.
- Keep the lab log updated with every decision and experiment verdict (for example, the 
  Experiment 1 AUC result and whether it passed), so nothing is lost when it is time to write.
- Gather and organize supporting material Ryan will need: references (start with the two 
  closest papers, Stream RAG and SpokenWOZ), figures from the harness, and the key numbers, 
  all in one place.

### Definition of done

Pre-registration current and signed, lab log up to date, and results compiled into organized 
tables and summaries ready for the paper. Ryan writes the paper; this role supplies the 
organized material.

### Coordinate with

Ryan (he writes the paper and signs off the pre-registration), Shubhankit and Ryan (their 
experiment results are what you compile), and the whole team (record their decisions in the 
lab log).
