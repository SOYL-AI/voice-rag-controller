# E3 Baseline Interface (FROZEN — the shared contract)

Do not change without notifying the whole team. Same rule as `eventlog/schema.md`: this is
the E3 equivalent of the frozen event-log schema. Every baseline (B0-B9 and Ours) implements
this same interface, so all of them plug into the one frontier sweep and Shubhankit's
analysis runs over them identically. Build your baseline against this contract and it will
combine cleanly with everyone else's.

**Change notice (2026-08-13):** E3 adds two fields to the existing `controller_action` event
payload — `baseline` (which baseline made the decision) and the price point `lambda_l` /
`lambda_c` (for B9 and Ours). This is additive and non-breaking: old events and code are
unaffected. Flagging here since it wasn't posted in the group — the frozen-schema rule is
notify before changing, so consider this the notice.

## The core idea

Every baseline is a controller: given the current state of a turn, it decides three things
(retrieve or not, when to launch, what to remember) and logs the same events. B0-B7 are
simple/partial controllers (some decisions fixed or trivial); B8/B9/Ours are the full ones.
The sweep runs each baseline over the same scenarios and records the same metrics, so their
frontiers are comparable.

## The interface every baseline implements

Each baseline is a class (or factory) exposing one decision entry point, callable from the
pipeline the same way our current controller is:

```python
class BaselineController:
    name: str                      # e.g. "B4_flare", "B8_independent_best", "ours"

    def decide(self, ctx) -> Decision:
        """Given the current turn context, return the control decision."""
        ...
```

The real implementation lives in `controller/baseline.py` (`Ctx`, `Decision`,
`BaselineController`, `register`) — that module is the code form of this contract; this
document is the narrative form. Copy `controller/baselines/b0_never_retrieve.py` as your
starting template.

`ctx` (read-only, provided by the pipeline) contains at least:

- `transcript_partial` / `transcript_final` — what's been said so far
- `signals` — the belief signals (`max_entropy_nats` etc. from E1; retrieval scores; tracker
  uncertainty), the same ones E1 validated
- `elapsed_ms`, `user_end_known` — timing context (for placement decisions)
- `turn_id`, `conversation_id`
- `prices` — `{lambda_l, lambda_c}`, the current latency/compute prices (used by B9 and Ours;
  ignored by baselines that don't price)

`Decision` returns:

- `retrieve: bool` — whether to retrieve this turn
- `timing: str` — when to launch: one of `sentence_end`, `immediate`, `fixed_early_proxy`
  (reuse the E2 timing policies; simple baselines use `sentence_end`)
- `memory_op: Optional[str]` — memory action if any (most baselines: `None`; B7/Ours: may act)
- `reason: str` — a short string logged for auditability (like the E2 mask reasons)

## The scalarization objective (B9 and Ours)

Per the handbook: each policy is scored by `quality − λ_l·latency − λ_c·compute`. Sweeping
`(λ_l, λ_c)` traces each policy's Pareto frontier. So B9 and Ours read `ctx.prices` and make
decisions that optimize that objective; the sweep varies the prices across a grid. B0-B7 do
NOT price (they're fixed heuristics) — they appear as single points, not frontiers, which is
the intended contrast.

- **B9 (decomposed-with-prices):** each component (trigger, timing, memory) optimizes the
  objective separately against the shared prices — no shared state between them. This is the
  strongest rival: coordinated by price, but not jointly reasoned.
- **Ours (unified):** one controller reasons about all three decisions together under the
  objective — the coupling hypothesis. The difference from B9 is the crux of H3, so implement
  the "decomposed vs unified" distinction faithfully.

**The quality term is a fitted probability, not a heuristic.** `estimate_voi()` in the shared
`controller/baselines/_objective.py` returns
`P(retrieval needed | max_entropy_nats) * utility_scale`, using the logistic fitted on
Experiment 1's 180 real Modal samples (`controller/baselines/e1_calibration.json`,
AUC 0.978). B9 and Ours import that one function, so a measured gap between them can never
be attributed to one side holding a better belief.

**The λ_l grid is derived, not chosen.** `python -m simulator.e3_grid` tiles the
retrieve/skip divergence bands edge-to-edge across the VoI distribution the 50 scenarios
produce and writes `configs/e3_price_grid.json`. A hand-picked grid risks placing every band
above or below the observed VoI values, where B9 and Ours cannot disagree — which returns a
null that is an artifact of the grid rather than a result about coupling.

## Logging (reuse the frozen schema, no new event types)

Every baseline logs through the shared `LogWriter` using the existing events
(`controller_action`, `retrieval_launched`, `retrieval_result`, `llm_final`,
`tts_first_audio`, etc.). Add the baseline name and, where relevant, the prices to the
`controller_action` payload so the analysis can group by baseline and price point:

```
controller_action payload: {action, timing, reason, baseline, lambda_l, lambda_c}
```

This is additive to the frozen schema (see the change notice above — notify the team, same
as any schema touch).

## Metrics the sweep records (handbook 29.4)

The harness already computes most of these; the frontier needs, per
`(baseline, price point, scenario, seed)`:

- **Quality:** task success + answer accuracy (vs gold) + hit@3 (retrieval)
- **Latency:** p50/p95 TTFA, dead-air rate
- **Compute:** retrievals per dialogue, wasted-retrieval rate (launched but `results_used`
  false), prompt tokens per turn, controller overhead ms per decision

## The fairness rule (handbook 29.3 — non-negotiable)

Every baseline gets the same tuning budget on the same dev split, and thresholds are swept,
not defaulted. A strawman baseline invalidates the whole headline. So each baseline owner
must sweep their baseline's thresholds (not hardcode a guess), tuned on the dev split, same
budget as everyone. This is how we make B8/B9 genuinely strong rivals.

## How your piece plugs in

1. Implement your baseline as a `BaselineController` with `decide(ctx)`.
2. Register it by name so the sweep can select it.
3. It reads `ctx`, returns a `Decision`, logs via the shared `LogWriter`.
4. The sweep runner (built centrally) runs all baselines × price grid × scenarios × seeds.
5. Shubhankit's frontier analysis reads the logs and builds the frontiers.

Build to this contract and your baseline drops into the sweep with no glue code.
