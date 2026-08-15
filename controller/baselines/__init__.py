"""
controller/baselines/__init__.py — imports every baseline module so its
@register(...) decorator runs and the baseline becomes selectable by name
via controller.baseline.get_baseline().

Importing controller.baseline alone does NOT register any baseline — that
module only defines the contract (Ctx, Decision, BaselineController) and
the empty registry. A baseline module only registers itself once something
actually imports it, which is exactly what this file is for: `import
controller.baselines` (this package) triggers every line below, which
triggers every @register(...) in turn.

Add your new baseline's import here when you add the file — nothing else
in this file should need to change. Keep the list alphabetical by baseline
name so a merge conflict here is easy to resolve.
"""

from controller.baselines import b0_never_retrieve  # noqa: F401 -- import-for-side-effect: runs @register("b0")
from controller.baselines import b8_independent  # noqa: F401 -- import-for-side-effect: runs @register("b8")
from controller.baselines import b9_decomposed  # noqa: F401 -- import-for-side-effect: runs @register("b9")
from controller.baselines import ours  # noqa: F401 -- import-for-side-effect: runs @register("ours")
from controller.baselines import b4_flare          # noqa: F401
from controller.baselines import b5_adaptive_rag   # noqa: F401
from controller.baselines import b6_dragin         # noqa: F401

# B1-B7 land here as their modules are built, e.g.:
#   from controller.baselines import b1_threshold_tracker  # noqa: F401
#
# Note: controller/baselines/_objective.py is NOT imported here. It's the
# shared objective/cost model the price-aware baselines (ours, b9) import
# themselves — it registers no baseline, so it has no side effect to
# trigger and doesn't belong in this list.
