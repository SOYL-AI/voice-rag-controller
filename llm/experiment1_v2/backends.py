"""Async generation backends for E1-v2.

Both return a GenResult around llm/client.py's Generation, so every signal
is computed by the same TokenLogprob/Generation code v1 E1 used.

VLLMAsyncBackend  streams from the vLLM OpenAI-compatible endpoint with
                  logprobs + top_logprobs (top_k from configs/config.yaml, as
                  v1), stamping the arrival time of every token so TTFT and
                  the latency of each first-k signal are measured, not
                  inferred. Endpoint and key come from the environment via
                  llm/config.py; neither is ever written to a record, a log
                  line or an exception message (see _redact).
MockAsyncBackend  reuses llm/client.py's MockClient for the synthetic
                  logprobs (seeded on the messages + seed, so deterministic)
                  and picks a deterministic answer per (item, seed, prefix)
                  so the labelling path has both right and wrong answers to
                  grade. Its timings are synthetic. Plumbing only, never data.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import random
import time
from dataclasses import dataclass, field

from llm.client import AlternativeLogprob, Generation, MockClient, TokenLogprob
from llm.config import api_key


@dataclass
class GenResult:
    generation: Generation
    ttft_ms: float | None
    gen_ms: float | None
    token_arrival_ms: list = field(default_factory=list)
    finish_reason: str | None = None
    served_model: str | None = None
    attempts: int = 1
    timing_synthetic: bool = False


class GenerationError(RuntimeError):
    """A call that did not settle after retries. Message is redacted."""


_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_RETRYABLE_NAMES = {
    "APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError",
    "ConnectError", "ConnectTimeout", "ReadTimeout", "ReadError", "RemoteProtocolError",
    "TimeoutError", "PoolTimeout", "WriteTimeout",
}


def is_retryable(error: BaseException) -> bool:
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status in _RETRYABLE_STATUS:
        return True
    return type(error).__name__ in _RETRYABLE_NAMES


def _redact(text: str, secrets) -> str:
    text = str(text)
    for secret in secrets:
        if secret and len(secret) >= 4:
            text = text.replace(secret, "<redacted>")
    return text


class VLLMAsyncBackend:
    name = "vllm"

    def __init__(self, llm_cfg: dict, top_k: int, timeout_s: float = 120.0,
                 max_attempts: int = 6, base_delay_s: float = 1.0, max_delay_s: float = 30.0):
        from openai import AsyncOpenAI  # noqa: PLC0415 -- only needed for real runs

        server = llm_cfg.get("server") or {}
        self._base_url = server.get("base_url", "http://localhost:8000/v1")
        self._key = api_key(llm_cfg)
        self._secrets = [self._base_url, self._base_url.rstrip("/"), self._key]
        self.model = llm_cfg["model"]
        self.extra_body = llm_cfg.get("extra_body") or {}
        self.top_k = int(top_k)
        self.max_attempts = int(max_attempts)
        self.base_delay_s = float(base_delay_s)
        self.max_delay_s = float(max_delay_s)
        # max_retries=0: retries are ours, so attempts are counted and logged.
        self._client = AsyncOpenAI(
            base_url=self._base_url, api_key=self._key, timeout=timeout_s, max_retries=0
        )

    def redact(self, text) -> str:
        return _redact(text, self._secrets)

    async def served_models(self) -> list:
        try:
            listing = await self._client.models.list()
        except Exception as error:  # noqa: BLE001
            raise GenerationError(
                f"cannot reach the LLM server: {type(error).__name__}: {self.redact(error)}"
            ) from None
        return [m.id for m in listing.data]

    async def close(self) -> None:
        await self._client.close()

    async def generate(self, messages, *, seed, temperature, top_p, max_tokens,
                       mock_hint=None) -> GenResult:
        delay = self.base_delay_s
        last = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = await self._once(messages, seed, temperature, top_p, max_tokens)
                result.attempts = attempt
                return result
            except Exception as error:  # noqa: BLE001 -- classified below
                last = error
                if not is_retryable(error) or attempt == self.max_attempts:
                    break
                await asyncio.sleep(min(self.max_delay_s, delay) * (0.5 + random.random()))
                delay *= 2
        raise GenerationError(
            f"{type(last).__name__}: {self.redact(last)}"[:600]
        ) from None

    async def _once(self, messages, seed, temperature, top_p, max_tokens) -> GenResult:
        request = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "logprobs": True,
            "top_logprobs": self.top_k,
            "seed": int(seed),
        }
        if self.extra_body:
            request["extra_body"] = self.extra_body
        gen = Generation(text="", model=self.model)
        arrivals, pieces = [], []
        finish_reason = served = None
        ttft = None
        start = time.perf_counter()
        stream = await self._client.chat.completions.create(**request)
        async for chunk in stream:
            served = served or getattr(chunk, "model", None)
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            now = (time.perf_counter() - start) * 1000.0
            delta = getattr(choice.delta, "content", None) or ""
            lp = getattr(choice, "logprobs", None)
            entries = list(getattr(lp, "content", None) or []) if lp is not None else []
            for entry in entries:
                alternatives = [
                    AlternativeLogprob(item.token, item.logprob)
                    for item in (getattr(entry, "top_logprobs", None) or [])
                ]
                gen.tokens.append(TokenLogprob(entry.token, entry.logprob, top_logprobs=alternatives))
                arrivals.append(now)
            if ttft is None and (delta or entries):
                ttft = now
            if delta:
                pieces.append(delta)
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
        gen_ms = (time.perf_counter() - start) * 1000.0
        gen.text = "".join(pieces)
        return GenResult(
            generation=gen, ttft_ms=ttft, gen_ms=gen_ms, token_arrival_ms=arrivals,
            finish_reason=finish_reason, served_model=served,
        )


class MockAsyncBackend:
    """Deterministic fake signals via MockClient. NOT DATA."""

    name = "mock"
    # Synthetic timing so latency plots have something to draw; flagged on
    # every record (timing_synthetic=True).
    MOCK_TTFT_MS = 120.0
    MOCK_PER_TOKEN_MS = 30.0

    def __init__(self, llm_cfg: dict, top_k: int, fail_first_n_calls: int = 0, **_):
        cfg = copy.deepcopy(llm_cfg)
        cfg.setdefault("logprobs", {})["top_k"] = int(top_k)
        cfg["mock"] = {"ttft_s": 0.0, "per_token_s": 0.0}
        self._client = MockClient(cfg)
        self.model = llm_cfg["model"]
        self.calls = 0
        self._fail_remaining = int(fail_first_n_calls)

    def redact(self, text) -> str:
        return str(text)

    async def served_models(self) -> list:
        return [self.model]

    async def close(self) -> None:
        return None

    @staticmethod
    def _pick_answer(messages, seed, mock_hint) -> str:
        hint = mock_hint or {}
        item = hint.get("item") or {}
        material = f"{item.get('id')}|{seed}|{messages[-1]['content']}"
        u = int(hashlib.sha256(material.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        expected = item.get("expected_answer") or "I don't know."
        if hint.get("grounded"):
            p_right = 0.9
        elif item.get("retrieval_needed_authored"):
            p_right = 0.2
        else:
            p_right = 0.85
        return expected if u < p_right else "I'm not sure about that, sorry."

    async def generate(self, messages, *, seed, temperature, top_p, max_tokens,
                       mock_hint=None) -> GenResult:
        self.calls += 1
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise GenerationError("mock: injected failure")
        await asyncio.sleep(0)
        answer = self._pick_answer(messages, seed, mock_hint)
        gen = self._client.generate(
            messages, seed=seed, temperature=temperature, top_p=top_p,
            max_tokens=max_tokens, mock_answer=answer,
        )
        n = len(gen.tokens)
        arrivals = [self.MOCK_TTFT_MS + i * self.MOCK_PER_TOKEN_MS for i in range(n)]
        return GenResult(
            generation=gen, ttft_ms=arrivals[0] if arrivals else None,
            gen_ms=(arrivals[-1] + self.MOCK_PER_TOKEN_MS) if arrivals else self.MOCK_TTFT_MS,
            token_arrival_ms=arrivals, finish_reason="stop", served_model=gen.model,
            timing_synthetic=True,
        )


def make_backend(kind: str, llm_cfg: dict, top_k: int, run_cfg: dict, **kwargs):
    if kind == "mock":
        return MockAsyncBackend(llm_cfg, top_k, **kwargs)
    if kind == "vllm":
        return VLLMAsyncBackend(
            llm_cfg, top_k,
            timeout_s=run_cfg.get("request_timeout_s", 120),
            max_attempts=run_cfg.get("max_attempts", 6),
            base_delay_s=run_cfg.get("base_delay_s", 1.0),
            max_delay_s=run_cfg.get("max_delay_s", 30.0),
        )
    raise ValueError(f"unknown backend {kind!r} (expected 'vllm' or 'mock')")
