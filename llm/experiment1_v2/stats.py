"""Statistics for E1-v2 analysis: AUC, cluster bootstrap, calibration.

Numpy only. The logistic fit is v1's unregularized IRLS
(llm/experiment1/fit_calibration.py:fit_logistic), reused so the v2 artifact
is the same estimator as the v1 one; isotonic is pool-adjacent-violators.
"""

from __future__ import annotations

import hashlib

import numpy as np

from llm.experiment1.fit_calibration import fit_logistic  # noqa: F401 -- re-exported


def auc(labels, scores):
    """ROC AUC with half credit for ties (Mann-Whitney via average ranks).
    Equal to llm/experiment1/run.py:roc_auc. None if a class is empty."""
    y = np.asarray(labels, dtype=float)
    s = np.asarray(scores, dtype=float)
    keep = ~np.isnan(s) & ~np.isnan(y)
    y, s = y[keep], s[keep]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)
    average_rank = (upper - counts + 1 + upper) / 2.0
    rank_sum = average_rank[inverse][y == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fit_logistic_safe(x, y, ridge=1e-3, max_iter=200):
    """v1's unregularized MLE when it exists. If the classes are separable
    the MLE diverges (v1's fit_logistic raises), so fall back to a lightly
    L2-penalised IRLS on the slope and say so. Returns
    (slope, intercept, info)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    try:
        slope, intercept, iterations = fit_logistic(x, y)
        if np.isfinite(slope) and np.isfinite(intercept):
            return slope, intercept, {"method": "mle", "iterations": iterations}
    except (RuntimeError, np.linalg.LinAlgError):
        pass
    design = np.column_stack([x, np.ones_like(x)])
    penalty = np.diag([ridge * len(x), 0.0])
    beta = np.zeros(2)
    for iteration in range(max_iter):
        p = sigmoid(design @ beta)
        w = np.clip(p * (1 - p), 1e-9, None)
        grad = design.T @ (y - p) - penalty @ beta
        hess = design.T @ (design * w[:, None]) + penalty + 1e-10 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[0]), float(beta[1]), {
        "method": f"l2-penalised (ridge {ridge} x n on slope): MLE does not exist, classes separable",
        "iterations": iteration + 1,
    }


def cluster_bootstrap(metric, clusters, n_boot=2000, seed=0, alpha=0.05):
    """Percentile CI of metric(index_array) resampling CLUSTERS with
    replacement (items, or pair_ids when items come in matched pairs).
    metric returns None for a degenerate resample, which is skipped."""
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    members = {c: np.flatnonzero(clusters == c) for c in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(n_boot)):
        picked = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([members[c] for c in picked])
        value = metric(index)
        if value is not None:
            values.append(value)
    if len(values) < max(10, n_boot // 10):
        return None
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def fit_isotonic(x, y):
    """Increasing step function by pool-adjacent-violators.
    Returns (thresholds, values): predict y for x as values at the last
    threshold <= x (first value below the range)."""
    order = np.argsort(x, kind="mergesort")
    xs, ys = np.asarray(x, float)[order], np.asarray(y, float)[order]
    # Merge tied x first so the fit is a function of x.
    ux, inv = np.unique(xs, return_inverse=True)
    sums = np.bincount(inv, weights=ys)
    counts = np.bincount(inv).astype(float)
    blocks = []  # [x_start, sum, count]
    for xv, sv, cv in zip(ux, sums, counts):
        blocks.append([xv, sv, cv])
        while len(blocks) > 1 and blocks[-2][1] / blocks[-2][2] > blocks[-1][1] / blocks[-1][2]:
            x0, s0, c0 = blocks[-2]
            _, s1, c1 = blocks[-1]
            blocks[-2:] = [[x0, s0 + s1, c0 + c1]]
    thresholds = np.array([b[0] for b in blocks])
    values = np.array([b[1] / b[2] for b in blocks])
    return thresholds, values


def predict_isotonic(model, x):
    thresholds, values = model
    index = np.searchsorted(thresholds, np.asarray(x, float), side="right") - 1
    return values[np.clip(index, 0, len(values) - 1)]


def ece(y, p, bins=10):
    """Expected calibration error, equal-width bins on [0, 1]."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def reliability(y, p, bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    out = []
    for b in range(bins):
        mask = index == b
        out.append({
            "bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]), "n": int(mask.sum()),
            "mean_predicted": float(p[mask].mean()) if mask.any() else None,
            "observed_rate": float(y[mask].mean()) if mask.any() else None,
        })
    return out


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def log_loss(y, p):
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def group_folds(groups, k):
    """Deterministic fold per group (sha256 of the group id), so pair
    partners and all seeds of an item always share a fold."""
    return np.array([int(hashlib.sha256(str(g).encode()).hexdigest(), 16) % k for g in groups])


def cross_validate(x, y, groups, k=5, bins=10):
    """Out-of-fold logistic and isotonic predictions on dev, grouped."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    folds = group_folds(groups, k)
    oof = {"logistic": np.full(len(x), np.nan), "isotonic": np.full(len(x), np.nan)}
    used = []
    for f in range(k):
        test = folds == f
        train = ~test
        if not test.any() or len(np.unique(y[train])) < 2:
            continue
        used.append(f)
        slope, intercept, _ = fit_logistic_safe(x[train], y[train])
        oof["logistic"][test] = sigmoid(slope * x[test] + intercept)
        oof["isotonic"][test] = predict_isotonic(fit_isotonic(x[train], y[train]), x[test])
    out = {"k": k, "folds_used": used}
    for name, p in oof.items():
        mask = ~np.isnan(p)
        if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
            out[name] = None
            continue
        out[name] = {
            "n": int(mask.sum()),
            "ece": ece(y[mask], p[mask], bins),
            "brier": brier(y[mask], p[mask]),
            "log_loss": log_loss(y[mask], p[mask]),
            "auc": auc(y[mask], p[mask]),
            "reliability": reliability(y[mask], p[mask], bins),
        }
    return out, oof
