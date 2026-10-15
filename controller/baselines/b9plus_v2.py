"""
b9plus_v2.py — B9+: the strongest decomposed rival.

Identical to b9_v2.B9V2 (same timing sub-optimizer, same trigger, same
retrieval sub-optimizer, same g+/g-/rho/waste/calibration) except for the
latency the retrieval sub-optimizer prices: not the full E[L] but the
EXPECTED EXPOSED latency of the policy the timing sub-optimizer chose, a
constant estimated on dev:

    priced_latency = Ebar_p + pi,  Ebar_p = mean over dev turns of
                     E_L[max(0, L - W_true)] at policy p's trigger,
                     W_true = end_of_turn - dispatch

This is the controller a reviewer argued would equal the joint one in v1 —
and under A1-A3 it does (constant overlap makes Ebar_p = L - o*, the very
price Ours uses). What it cannot do is price THIS turn's exposure, which
under C1 varies with the turn's remaining speech, nor trade waiting for
information (C2) against overlap. That gap — and only that gap — is what
ours_v2 is tested on against it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from controller.baselines._objective_v2 import priced_latency_ms  # noqa: E402
from controller.baselines.b9_v2 import B9V2  # noqa: E402


class B9PlusV2(B9V2):
    latency_pricing = "policy_expected_exposed"

    def _priced_latency_ms(self) -> float:
        row = self.prof.get(self.policy) or {}
        exposed = self.lat.mean_ms if row.get("exposed_ms") is None else float(row["exposed_ms"])
        # dev-mean exposure of the retrieval call + pi (identical for all)
        return priced_latency_ms(self.lat, exposed)
