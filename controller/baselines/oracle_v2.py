"""
oracle_v2.py — NON-CAUSAL UPPER BOUND. Not a controller anyone could deploy.

Retrieves iff the item's EMPIRICAL closed-book answer is wrong (the E1-v2
label: closed-book wrong across draws), and dispatches at the earliest
partial whose query retrieves a gold passage — or on the final transcript if
no partial does. Both facts are hindsight: the label needs the answer graded
against the reference, and "this partial's query hits gold" needs the gold
passage ids. The evaluator supplies them in TurnCtx.oracle:

    {"closed_book_wrong": bool,
     "gold_hit_partial_indices": [partial indices whose retrieval hits gold],
     "label_source": "..."}

It overrides decide_turn() — the only v2 controller that does — because it
is by definition not causal. Every record it produces is labelled ORACLE.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.sequential_v2 import (  # noqa: E402
    SKIP,
    SequentialController,
    StepObs,
    StepTrace,
    TurnCtx,
    TurnDecision,
    build_steps,
)


class OracleV2(SequentialController):
    priced = False
    causal = False

    def step(self, obs: StepObs, memory: dict) -> Tuple[str, StepTrace]:  # pragma: no cover
        raise RuntimeError("oracle_v2 is not a causal controller; use decide_turn()")

    def decide_turn(self, ctx: TurnCtx) -> TurnDecision:
        oracle = ctx.oracle
        if oracle is None or oracle.get("closed_book_wrong") is None:
            raise ValueError(f"oracle_v2 needs TurnCtx.oracle labels for item {ctx.item_id}")
        steps = build_steps(ctx, self.cfg)
        if not oracle["closed_book_wrong"]:
            return TurnDecision(None, "skip", None,
                                "ORACLE(upper bound): closed-book answer is right -> skip",
                                [StepTrace(k=0, available_ms=steps[0].available_ms, action=SKIP)])
        hits = set(int(i) for i in (oracle.get("gold_hit_partial_indices") or []))
        for obs in steps:
            if obs.is_final or obs.partial_index in hits:
                label = "sentence_end" if obs.is_final else "oracle_partial"
                return TurnDecision(
                    dispatch_t_ms=obs.available_ms, timing_label=label, query_text=obs.text,
                    reason=(f"ORACLE(upper bound): closed-book wrong -> retrieve at "
                            f"{'final transcript' if obs.is_final else f'partial {obs.partial_index}'} "
                            f"(earliest gold hit; source={oracle.get('label_source', '?')})"),
                    trace=[StepTrace(k=obs.k, available_ms=obs.available_ms, action="retrieve")],
                    dispatch_step=obs.k, dispatch_partial_index=obs.partial_index,
                )
        raise RuntimeError("unreachable: build_steps always ends with a final step")
