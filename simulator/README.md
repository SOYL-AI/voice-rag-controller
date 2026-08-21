# simulator/ — Fake-User Scenarios

**Owner: Lakshya (second priority, after the retrieval corpus)**

## What this folder is

The fake user: software that generates test conversations (guest goals, the questions they 
would ask) with a known correct answer for each. This lets us run many test conversations 
without needing live people every time.

## Status: not a Week 1 deliverable

The full simulator is needed for later experiments (Weeks 3–4), not Week 1. The detailed 
brief will be added here in Week 2.

For Week 1, focus is on the retrieval corpus (see `retrieval/README.md`). If there is spare 
time, a light head start is fine: draft a scenario format (a guest goal, the question, the 
expected answer, and whether a document lookup is needed) and a handful of example 
scenarios.

## Important when this starts

The answer key for each scenario must match what the hotel documents in `retrieval/corpus/` 
actually say. Because the same person owns both the corpus and the scenarios, this stays 
consistent.

---

## E3 tooling that lives here (owner: Renya)

Two Experiment 3 files sit in this folder because they operate on
`scenarios.json`, not because they belong to the simulator brief above:

- `e3_signals.py` — the frozen per-scenario closed-book belief-signal precompute every E3
  baseline is handed (so baselines differ only in decision logic, never in what they saw).
  It runs **Experiment 1's protocol**: one generation per E1 seed (`[11, 23, 37, 53, 71]`) at
  E1's temperature 0.2 / top_p 0.9 / max_tokens 80, read straight out of
  `llm/experiment1/config.json`, with every uncertainty signal averaged across the five draws.
  Fixed seeds make it exactly reproducible; matching E1's sampling distribution is what makes
  the averaged entropy the quantity `controller/baselines/e1_calibration.json` was fitted on.
  A single unseeded draw (the pre-v2 protocol) was neither.
- `e3_grid.py` — derives E3's λ_l price grid by tiling divergence bands across the VoI
  distribution of these 50 scenarios. One command regenerates it:

  ```powershell
  python -m simulator.e3_grid                   # prints the distribution + grid, writes configs/e3_price_grid.json
  python -m simulator.e3_grid --refresh-signals # after changing llm.backend/model
  ```

  It caches the per-scenario signals in `e3_scenario_signals.json` (committed) so grid
  regeneration needs no model call, and records the backend **and the precompute protocol**
  those signals came from, rebuilding automatically if either no longer matches — with
  `llm.backend: mock` they are synthetic and the grid must be regenerated against the real
  backend before anything is reported off it. See `controller/README.md` for the formula.
