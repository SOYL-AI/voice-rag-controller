"""Answer judges for E1-v2: a primary and a secondary LLM judge on Azure OpenAI.

  primary    the v2 judge of record: a GPT-6-family deployment, chosen at
             pre-registration freeze. Deployment, model id and API version
             come from VOICE_RAG_JUDGE_DEPLOYMENT / _MODEL / _API_VERSION.
  secondary  the validated v4 "reasoned + strip" judge that graded the real
             E3 sweep (gpt-4o), from VOICE_RAG_JUDGE2_* or the AZURE_OPENAI_*
             variables the E3 runner reads.

Both grade with the SAME prompt, so their disagreement measures the judges,
not the prompts. The prompt is E3's v4 instrument (simulator/judge_core.py:
reason first, list only direct conflicts, final 'VERDICT: YES|NO', answer
stripped before both the cache key and the prompt) generalised off the hotel:
the system prompt names the dataset's setting instead of "hotel concierge",
"naming the hotel" reads "naming the company or organisation", and two
optional blocks carry answer aliases (d1_open) and key facts (d2_support).
With key facts the judge is also told that an answer stating none of them is
incorrect, because a generic deflection ("please check your account") would
otherwise pass v4's omission rule while answering nothing. That rule is new
in v2 and is what judge validation against the team's human labels has to
confirm. The prompt version string (judges.json) changes whenever any of
this does.

Request parameters are per judge (judges.json, overridable by env). The
primary (gpt-6.1-sol, api-version 2025-04-01-preview) is a reasoning model:
it rejects `max_tokens` and any temperature but the default, so it is sent
max_completion_tokens (a budget that also covers reasoning tokens), a pinned
reasoning_effort and a seed, and never a temperature. Because it therefore
samples stochastically, each answer is graded by k samples (seeds seed,
seed+1, ...; k=3 by default) and the verdict is their majority; every sample
is kept so self-consistency can be reported. The secondary keeps v4's
temperature 0, max_tokens 300, one sample. If a deployment still rejects a
parameter the judge adapts once (as v4 does). Every verdict records the
parameters ACTUALLY sent, the model version the server reported, and the
token usage (including reasoning tokens); the request config (effort, seed,
k) is part of the cache key.

Credentials and endpoints are read from the environment only and never
written to a verdict, a cache file or a log line.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from llm.experiment1_v2 import common
from llm.experiment1_v2.rubrics import get_rubric, rubric_sha256
from simulator.judge_core import JUDGE_INSTRUCTIONS, _is_retryable, parse_verdict


class JudgeConfigError(RuntimeError):
    """A judge is not configured. Never silently fall back to another."""


class JudgeCallError(RuntimeError):
    """A configured judge did not produce a reply after retries."""


_HOTEL_PHRASE = "naming the hotel"
assert _HOTEL_PHRASE in JUDGE_INSTRUCTIONS, "v4 judge instructions changed; re-derive the v2 prompt"
V2_INSTRUCTIONS = JUDGE_INSTRUCTIONS.replace(_HOTEL_PHRASE, "naming the company or organisation")
KEY_FACTS_RULE = (
    "- KEY FACTS are listed below the reference. The candidate is also "
    "INCORRECT if it states none of the key facts: a generic or evasive "
    "reply does not answer the question.\n"
)


def system_prompt(context: str) -> str:
    article = "an" if context[:1].lower() in "aeiou" else "a"
    return (
        f"You are a strict grader for {article} {context} question-answering "
        "system. You compare a candidate answer against a reference answer "
        "and judge FACTUAL CORRECTNESS only."
    )


def user_prompt(question, reference, answer, key_facts=None, aliases=None) -> str:
    instructions = V2_INSTRUCTIONS
    if key_facts:
        # Inserted just before the conflict-list instruction paragraph.
        head, sep, tail = instructions.partition("\n\nList every DIRECT CONFLICT")
        instructions = head + "\n" + KEY_FACTS_RULE.rstrip("\n") + sep + tail
    parts = [instructions, f"Question:\n{question}", f"Reference answer:\n{reference}"]
    if aliases:
        parts.append("Equivalent forms of the reference answer:\n"
                     + "\n".join(f"- {a}" for a in aliases))
    if key_facts:
        parts.append("Key facts:\n" + "\n".join(f"- {f}" for f in key_facts))
    parts.append(f"Candidate answer:\n{answer}")
    return "\n\n".join(parts) + "\n"


# Fingerprint of the prompt as rendered from its templates (PREREG_V2
# section 2.2 fixes "prompt version ID and sha256" at freeze). Part of every
# judge identity, so any edit to the wording changes the cache namespace.
PROMPT_SHA256 = common.sha256_text(
    system_prompt("{context}"),
    user_prompt("{question}", "{reference}", "{answer}", ["{key_fact}"], ["{alias}"]),
    user_prompt("{question}", "{reference}", "{answer}"),
)


# Fingerprint of the full prompt template (system line, instructions, the
# key-facts rule and the block layout). Part of every judge identity and so
# of every cache key; the pre-registration freezes it with prompt_version.
PROMPT_SHA256 = common.sha256_text(
    system_prompt("{context}"),
    user_prompt("{question}", "{reference}", "{answer}", ["{fact}"], ["{alias}"]),
    user_prompt("{question}", "{reference}", "{answer}"),
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class JudgeSpec:
    role: str
    label: str
    provider: str
    deployment: str
    model_id: str
    api_version: str
    prompt_version: str
    request: dict
    endpoint: str = field(default="", repr=False)
    api_key: str = field(default="", repr=False)

    def identity(self) -> dict:
        """Public identity: what graded. No endpoint, no key."""
        return {
            "role": self.role, "label": self.label, "provider": self.provider,
            "deployment": self.deployment, "model_id": self.model_id,
            "api_version": self.api_version, "prompt_version": self.prompt_version,
            "prompt_sha256": rubric_sha256(self.prompt_version),
        }

    def identity_string(self) -> str:
        return (f"{self.provider}/{self.deployment}[{self.model_id}]@{self.api_version}"
                f"#{self.prompt_version}")

    def slug(self) -> str:
        """Cache directory name: the label plus a hash of the full identity.
        Short on purpose (Windows MAX_PATH); the cache KEY carries the
        identity in full, this only keeps judges in separate directories."""
        digest = common.sha256_text(json.dumps(self.identity(), sort_keys=True))[:10]
        label = "".join(c if c.isalnum() or c in "-._" else "-" for c in self.label)[:24]
        return f"{label}-{digest}"


def _first_env(names, env) -> tuple:
    for name in names if isinstance(names, (list, tuple)) else [names]:
        value = (env.get(name) or "").strip()
        if value:
            return name, value
    return None, None


def _override(value: str):
    if value.lower() in ("none", "null", ""):
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def load_judges_config(path: Path = None) -> dict:
    return common.load_json(path or common.DEFAULT_JUDGES_CONFIG)


def resolve_spec(role: str, judges_cfg: dict, env=None) -> JudgeSpec:
    env = os.environ if env is None else env
    entry = judges_cfg["judges"][role]
    found, missing = {}, []
    for field_name in ("deployment", "api_version", "model_id", "endpoint", "api_key"):
        names = entry["env"][field_name]
        _, value = _first_env(names, env)
        if value is None and field_name == "model_id":
            value = entry.get("default_model_id")
        if value is None and field_name != "model_id":
            missing.append(" or ".join(names) if isinstance(names, list) else names)
        found[field_name] = value
    if missing:
        raise JudgeConfigError(
            f"The {role} judge ({entry['label']}) is not configured: set "
            + "; ".join(missing)
            + ". There is no fallback judge: an unconfigured judge stops the run."
        )
    if not found["model_id"]:
        raise JudgeConfigError(
            f"The {role} judge needs its model id/version: set "
            f"{' or '.join(entry['env']['model_id'])} (e.g. the GPT-6 model version "
            "named in the pre-registration)."
        )
    request = dict(entry["request"])
    prefix = entry.get("env_overrides_prefix")
    if prefix:
        for key, suffix in (("temperature", "TEMPERATURE"), ("token_param", "TOKEN_PARAM"),
                            ("max_output_tokens", "MAX_TOKENS"),
                            ("reasoning_effort", "REASONING_EFFORT"), ("seed", "SEED"),
                            ("samples", "SAMPLES"), ("max_extra_samples", "MAX_EXTRA_SAMPLES")):
            raw = env.get(f"{prefix}_{suffix}")
            if raw is not None:
                request[key] = _override(raw.strip())
    request["samples"] = int(request.get("samples") or 1)
    if request["samples"] < 1:
        raise JudgeConfigError(f"{role} judge samples must be >= 1")
    if request.get("token_param") not in ("max_tokens", "max_completion_tokens"):
        raise JudgeConfigError(f"{role} judge token_param must be max_tokens or "
                               f"max_completion_tokens, got {request.get('token_param')!r}")
    return JudgeSpec(
        role=role, label=entry["label"], provider=entry["provider"],
        deployment=found["deployment"], model_id=found["model_id"],
        api_version=found["api_version"], prompt_version=judges_cfg["prompt_version"],
        request=request, endpoint=found["endpoint"].rstrip("/"), api_key=found["api_key"],
    )


# ---------------------------------------------------------------------------
# Judges
# ---------------------------------------------------------------------------
def _sdk_transport(spec: JudgeSpec, timeout_s: float = 120.0):
    from openai import AzureOpenAI, DefaultHttpxClient  # noqa: PLC0415 -- real runs only

    # The JSON body the SDK actually puts on the wire is captured by an
    # httpx request hook (per thread), so the record holds the exact request
    # including anything the SDK adds. Only the body: never the URL or the
    # headers, which carry the endpoint and the key.
    captured = threading.local()

    def on_request(request):
        captured.body = request.content

    client = AzureOpenAI(api_key=spec.api_key, azure_endpoint=spec.endpoint,
                         api_version=spec.api_version, max_retries=0, timeout=timeout_s,
                         http_client=DefaultHttpxClient(event_hooks={"request": [on_request]}))

    def call(messages, params):
        # All sampling/limit parameters go through extra_body so the JSON
        # sent is exactly `params` plus model and messages, whatever this SDK
        # version knows about (openai 1.52 has no reasoning_effort argument).
        captured.body = None
        response = client.chat.completions.create(
            model=spec.deployment, messages=messages, extra_body=params
        )
        body = json.loads(captured.body) if captured.body else None
        choice = response.choices[0]
        return {
            "text": choice.message.content or "",
            "model": getattr(response, "model", None),
            "usage": response.usage.model_dump() if getattr(response, "usage", None) else None,
            "response_id": getattr(response, "id", None),
            "finish_reason": getattr(choice, "finish_reason", None),
            "request_body": body,
        }

    return call


class AzureChatJudge:
    def __init__(self, spec: JudgeSpec, transport=None, sleep=time.sleep):
        self.spec = spec
        self._transport = transport or _sdk_transport(spec)
        self._sleep = sleep
        self._lock = threading.Lock()
        self._temperature = spec.request.get("temperature")
        self._token_param = spec.request.get("token_param", "max_tokens")
        self.adaptations = []

    @property
    def identity(self) -> dict:
        return self.spec.identity()

    def _params(self, seed=None) -> dict:
        params = {self._token_param: int(self.spec.request.get("max_output_tokens") or 300)}
        if self._temperature is not None:
            params["temperature"] = self._temperature
        if self.spec.request.get("reasoning_effort"):
            params["reasoning_effort"] = self.spec.request["reasoning_effort"]
        if seed is not None:
            params["seed"] = int(seed)
        return params

    def _redact(self, text) -> str:
        text = str(text)
        for secret in (self.spec.endpoint, self.spec.api_key):
            if secret and len(secret) >= 4:
                text = text.replace(secret, "<redacted>")
        return text

    def _adapt(self, error) -> bool:
        text = str(error).lower()
        with self._lock:
            if "temperature" in text and self._temperature is not None:
                self._temperature = None
                self.adaptations.append("dropped temperature (rejected by deployment)")
            elif "max_tokens" in text and "max_completion_tokens" in text \
                    and self._token_param == "max_tokens":
                self._token_param = "max_completion_tokens"
                self.adaptations.append("max_tokens -> max_completion_tokens")
            elif "max_completion_tokens" in text and self._token_param == "max_completion_tokens":
                self._token_param = "max_tokens"
                self.adaptations.append("max_completion_tokens -> max_tokens")
            elif "max_tokens" in text and self._token_param == "max_tokens":
                self._token_param = "max_completion_tokens"
                self.adaptations.append("max_tokens -> max_completion_tokens")
            else:
                return False
        print(f"[judge:{self.spec.label}] request adapted: {self.adaptations[-1]}")
        return True

    def complete(self, system: str, user: str, seed=None) -> dict:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        transient = adapted = 0
        while True:
            params = self._params(seed)
            try:
                reply = self._transport(messages, params)
                if not isinstance(reply, dict):  # (text, model[, usage]) test transports
                    reply = {"text": reply[0], "model": reply[1],
                             "usage": reply[2] if len(reply) > 2 else None}
                body = reply.get("request_body")
                return {
                    "text": (reply.get("text") or "").strip(),
                    "sent_params": params,
                    "request_body": body if body is not None else
                    {"model": self.spec.deployment, "messages": messages, **params},
                    "request_body_captured": body is not None,
                    "response_model": reply.get("model"),
                    "response_id": reply.get("response_id"),
                    "finish_reason": reply.get("finish_reason"),
                    "usage": reply.get("usage"),
                    "attempts": transient + adapted + 1,
                    "adaptations_so_far": list(self.adaptations),
                }
            except Exception as error:  # noqa: BLE001 -- classified below
                if adapted < 3 and self._adapt(error):
                    adapted += 1
                    continue
                transient += 1
                if not _is_retryable(error) or transient > 10:
                    raise JudgeCallError(
                        f"{self.spec.label}: {type(error).__name__}: {self._redact(error)}"[:600]
                    ) from None
                self._sleep(min(60.0, 2 ** (transient - 1)))


class MockJudge:
    """Deterministic stand-in for tests and --judge-backend mock. NOT DATA.
    Says YES when the reference (or an alias, or any key fact) appears in
    the answer; `flip_rate` flips a deterministic fraction so two mock
    judges can disagree; the flip depends on the sample seed, so a k-sample
    mock judge is not perfectly self-consistent either."""

    def __init__(self, role: str, label: str, prompt_version: str, flip_rate: float = 0.0,
                 samples: int = 1, seed=None):
        self.spec = JudgeSpec(role=role, label=label, provider="mock", deployment=f"mock-{label}",
                              model_id=f"mock-{label}", api_version="none",
                              prompt_version=prompt_version,
                              request={"mock": True, "samples": int(samples), "seed": seed})
        self.flip_rate = float(flip_rate)

    @property
    def identity(self) -> dict:
        return self.spec.identity()

    def complete(self, system: str, user: str, seed=None) -> dict:
        def block(name):
            if f"{name}:\n" not in user:
                return ""
            return user.split(f"{name}:\n", 1)[1].split("\n\n", 1)[0]

        answer = _norm(block("Candidate answer"))
        targets = [block("Reference answer")]
        targets += [line[2:] for line in block("Equivalent forms of the reference answer").splitlines()]
        targets += [line[2:] for line in block("Key facts").splitlines()]
        ok = any(_norm(t) and _norm(t) in answer for t in targets)
        material = f"{self.spec.label}|{block('Question')}|{answer}|{seed}"
        u = int(hashlib.sha256(material.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        if u < self.flip_rate:
            ok = not ok
        text = "CONFLICT: none\nVERDICT: YES" if ok else \
            "CONFLICT: reference says X, candidate says Y\nVERDICT: NO"
        return {"text": text, "sent_params": {"mock": True, "seed": seed}, "response_model": "mock",
                "usage": None, "attempts": 1}


def _norm(text: str) -> str:
    import re  # noqa: PLC0415

    return " ".join(re.sub(r"[^a-z0-9 ]", " ", (text or "").lower()).split())


def make_judges(roles, backend: str, judges_cfg: dict = None, env=None) -> dict:
    judges_cfg = judges_cfg or load_judges_config()
    out = {}
    for role in roles:
        if backend == "mock":
            entry = judges_cfg["judges"][role]
            request = entry["request"]
            out[role] = MockJudge(role, entry["label"], judges_cfg["prompt_version"],
                                  flip_rate=0.05 if role == "primary" else 0.1,
                                  samples=int(request.get("samples") or 1), seed=request.get("seed"))
        elif backend == "azure":
            out[role] = AzureChatJudge(resolve_spec(role, judges_cfg, env))
        else:
            raise ValueError(f"unknown judge backend {backend!r}")
    return out


# ---------------------------------------------------------------------------
# Grading with a cache
# ---------------------------------------------------------------------------
def cache_key(judge, *, item_id, question, reference, answer, key_facts, aliases, context,
              dataset=None, history=None) -> str:
    parts = [
        json.dumps(judge.identity, sort_keys=True),
        json.dumps(judge.spec.request, sort_keys=True),
        item_id, question, reference, json.dumps(key_facts or []), json.dumps(aliases or []),
        context, answer,
    ]
    if get_rubric(judge.spec.prompt_version) is not None:
        # File rubrics also see the dataset (which rubric text) and the
        # history; r1 keys stay byte-identical so its cache remains valid.
        parts += [dataset or "", json.dumps(history or [], sort_keys=True)]
    return common.sha256_text(*parts)


def render_prompt(judge, *, question, reference, answer, key_facts=None, aliases=None,
                  context="general", dataset=None, history=None) -> tuple:
    """(system, user) for this judge's rubric version."""
    rubric = get_rubric(judge.spec.prompt_version)
    if rubric is None:
        return system_prompt(context), user_prompt(question, reference, answer, key_facts, aliases)
    return system_prompt(context), rubric.user_prompt(
        dataset=dataset, question=question, reference=reference, answer=answer,
        key_facts=key_facts, aliases=aliases, history=history)


def grade(judge, *, item_id, question, reference, answer, key_facts=None, aliases=None,
          context="general", cache_dir: Path = None, dataset=None, history=None) -> dict:
    """One verdict, cached by (judge identity incl. rubric hash + request
    config + item + reference + answer [+ dataset, history for file
    rubrics]). The answer is stripped before both the key and the prompt
    (the v4 'strip' fix)."""
    answer = (answer or "").strip()
    key = cache_key(judge, item_id=item_id, question=question, reference=reference, answer=answer,
                    key_facts=key_facts, aliases=aliases, context=context, dataset=dataset,
                    history=history)
    system, user = render_prompt(judge, question=question, reference=reference, answer=answer,
                                 key_facts=key_facts, aliases=aliases, context=context,
                                 dataset=dataset, history=history)
    # Identical answers across seeds share a key; serialise them so the
    # judge is called once and nobody reads a file mid-replace (Windows).
    with _key_lock(key):
        return _grade_locked(judge, key, item_id, system, user, cache_dir)


_LOCKS: dict = {}
_LOCKS_GUARD = threading.Lock()


def _key_lock(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


def sample_seeds(request: dict) -> list:
    """Seeds for the k samples: seed, seed+1, ... (None when no seed is
    pinned, as for the temperature-0 v4 judge)."""
    k = int(request.get("samples") or 1)
    base = request.get("seed")
    return [None if base is None else int(base) + i for i in range(k)]


def majority(votes, min_parsed: int = 1) -> tuple:
    """(verdict, parsed_ok, self_consistency) over parsed sample verdicts.
    Unresolved (verdict False, parsed_ok False) when fewer than `min_parsed`
    samples parsed or the parsed ones tie, so it is counted, never silently
    scored."""
    parsed = [v for v in votes if v is not None]
    if not parsed:
        return False, False, None
    yes = sum(parsed)
    no = len(parsed) - yes
    if yes == no:
        return False, False, 0.5
    share = max(yes, no) / len(parsed)
    if len(parsed) < min_parsed:
        return False, False, share
    return yes > no, True, share


def _grade_locked(judge, key, item_id, system, user, cache_dir) -> dict:
    """k independent samples (each cached under its own key, which includes
    its seed), then a majority vote. Every sample is kept on the record."""
    request = judge.spec.request
    k = int(request.get("samples") or 1)
    need = k // 2 + 1  # a valid majority needs this many parsed samples (2 of 3)
    max_extra = int(request.get("max_extra_samples") or 0)
    base = request.get("seed")
    samples = []
    for index in range(k + max_extra):
        if index >= k:
            # PREREG_V2 section 9: fewer than 2 of 3 parsed (or a tie) ->
            # draw up to 3 more samples until a valid majority exists.
            if majority([s["answer_correct"] for s in samples], need)[1]:
                break
        seed = None if base is None else int(base) + index
        sample_key = common.sha256_text(key, index, seed)
        path = None
        if cache_dir is not None:
            path = Path(cache_dir) / judge.spec.slug() / f"{sample_key[:40]}.json"
            if path.exists():
                sample = common.load_json(path)
                sample["cache_hit"] = True
                samples.append(sample)
                continue
        reply = judge.complete(system, user, seed=seed)
        correct, parsed_ok = parse_verdict(reply["text"])
        sample = {
            "sample_index": index,
            "seed": seed,
            "answer_correct": correct if parsed_ok else None,
            "parsed_ok": parsed_ok,
            "judge_raw": reply["text"],
            "sample_key": sample_key,
            "extra_sample": index >= k,
            "response_model": reply.get("response_model"),
            "response_id": reply.get("response_id"),
            "finish_reason": reply.get("finish_reason"),
            "sent_params": reply.get("sent_params"),
            "request_body": reply.get("request_body"),
            "request_body_captured": reply.get("request_body_captured"),
            "usage": reply.get("usage"),
            "graded_at_utc": common.now_utc(),
        }
        if path is not None:
            common.atomic_write_json(path, sample)
        sample["cache_hit"] = False
        samples.append(sample)
    verdict, parsed_ok, consistency = majority([s["answer_correct"] for s in samples], need)
    first_k = [s["answer_correct"] for s in samples[:k]]
    record = {
        "answer_correct": verdict,
        "parsed_ok": parsed_ok,
        "n_samples": len(samples),
        "n_extra_samples": max(0, len(samples) - k),
        "self_consistency": consistency,
        # Unanimity over the k planned samples (the gate's definition).
        "unanimous": k > 1 and None not in first_k and len(set(first_k)) == 1,
        "judge_key": key,
        "judge": judge.identity,
        "request_config": judge.spec.request,
        "response_model": samples[0].get("response_model"),
        "sent_params": samples[0].get("sent_params"),
        "item_id": item_id,
        "samples": samples,
        "cache_hit": all(s["cache_hit"] for s in samples),
        "judge_raw": samples[0]["judge_raw"],
    }
    if isinstance(judge, MockJudge):
        record["judge_note"] = "MOCK JUDGE: deterministic plumbing verdict. Not data."
    return record
