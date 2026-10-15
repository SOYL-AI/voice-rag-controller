"""Signals for one generation: everything llm/signals.py computes, plus the
first-k entropy family that a controller could have BEFORE the answer ends.

first_k_mean_entropy_nats[k] is the mean per-token entropy (llm/client.py's
TokenLogprob.entropy_nats, the top-k-plus-residual lower bound) over the
first min(k, n) generated tokens. With a fixed per-request seed the first k
sampled tokens do not depend on max_tokens, so reading them off an
80-token generation gives the same value a k-token generation would; the
latency of obtaining it is the measured arrival time of token k
(t_first_k_ms), not the full generation time.
"""

from __future__ import annotations

from llm.signals import build_signal_snapshot


def first_k_mean_entropy(tokens, k: int):
    """Mean entropy over the first min(k, n) tokens that carry an entropy.
    None when no token has one (logprobs off or an empty answer)."""
    values = [t.entropy_nats for t in list(tokens)[:k]]
    values = [v for v in values if v is not None]
    return (sum(values) / len(values)) if values else None


def compute_signals(generation, first_k=(4, 8, 16), low_confidence_threshold=0.5) -> dict:
    snapshot = build_signal_snapshot(
        generation=generation,
        tracker_slots={},
        low_confidence_threshold=low_confidence_threshold,
    )
    signals = dict(snapshot["model"]["uncertainty"])
    signals["first_k_mean_entropy_nats"] = {
        str(k): first_k_mean_entropy(generation.tokens, int(k)) for k in first_k
    }
    return signals


def first_k_arrival_ms(token_arrival_ms, first_k=(4, 8, 16)) -> dict:
    """Request-start-relative arrival time of token k; None if fewer than k
    tokens were generated (the signal then saturates at n < k tokens and its
    latency is the full generation time, recorded separately as gen_ms)."""
    out = {}
    for k in first_k:
        k = int(k)
        out[str(k)] = token_arrival_ms[k - 1] if len(token_arrival_ms) >= k else None
    return out


def flatten_signals(signals: dict) -> dict:
    """{name: value} with nested dicts flattened as 'name@key', e.g.
    first_k_mean_entropy_nats@8. Non-numeric values are dropped."""
    flat = {}
    for name, value in (signals or {}).items():
        if isinstance(value, dict):
            for key, inner in value.items():
                if isinstance(inner, (int, float)) and not isinstance(inner, bool):
                    flat[f"{name}@{key}"] = float(inner)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[name] = float(value)
    return flat
