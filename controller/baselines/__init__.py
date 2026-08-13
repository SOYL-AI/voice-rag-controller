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

# B1-B9 / Ours land here as their modules are built, e.g.:
#   from controller.baselines import b1_threshold_tracker  # noqa: F401
#   from controller.baselines import b9_shared_price  # noqa: F401
#   from controller.baselines import ours  # noqa: F401
