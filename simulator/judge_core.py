"""The validated v4 answer judge, shared by E3 and E1-v2.

Moved verbatim out of simulator/e3_full_sweep.py (which re-exports every
name below) so that the instrument can be imported without pulling in the
E3 runner's pipeline, controllers and FAISS index. Nothing in the prompt, the
verdict parser, the cache version or the Azure transport changed in the move;
see e3_full_sweep.py's docstring and simulator/judge_validation/ for why each
choice was made and how it was measured.
"""

import hashlib
import os
import time


JUDGE_SYSTEM_PROMPT = (
    "You are a strict grader for a hotel-concierge question-answering "
    "system. You compare a candidate answer against a reference answer "
    "and judge FACTUAL CORRECTNESS only."
)

# The judge REASONS before it answers, and the omission rule binds the
# conflict LIST rather than only the verdict.
#
# Both of those are measured choices, not style. Scored against the 19
# hand labels in simulator/judge_validation/labels.json (run
# `python -m simulator.judge_validation.score_judges`):
#
#     terse, answer unstripped   14/19   74%   <- what the first sweep used
#     terse, answer stripped     15/19   79%
#     reasoned, stripped         18/19   95%   <- this
#
# The terse form got 8 tokens and no room to think, which left it deciding
# borderline cases on surface form: an answer identical to the reference
# except for the words "The Grand Horizon Hotel & Suites offers" at the
# front scored NO five times out of five, and YES five out of five with
# that phrase removed. Reasoning first removes that sensitivity.
#
# The omission rule has to constrain the list. Told only that omission is
# correct, the model still WROTE "CONFLICT: the reference says menu items
# may contain allergens, but the candidate does not mention this" and then
# voted NO on its own list. So the instruction is placed on what may be
# listed as a conflict at all.
#
# Every error made by every variant was in one direction — marking a
# correct answer incorrect, never the reverse. That direction matters: it
# under-credits retrieval, and so penalises whichever baseline retrieves
# most. The remaining miss at 18/19 is s018/retrieve, which is labelled
# BORDERLINE in labels.json by the person who labelled it.
JUDGE_INSTRUCTIONS = (
    "Decide whether the candidate answer is factually correct with respect "
    "to the reference answer.\n"
    "Only FACTUAL CONFLICT counts. Specifically:\n"
    "- Wording, length, style, politeness, and naming the hotel do NOT "
    "matter.\n"
    "- Restating the reference in different words is CORRECT.\n"
    "- Omitting some of the reference's details is CORRECT.\n"
    "- Adding detail the reference does not mention is CORRECT, unless it "
    "states a specific figure, time, name, price or location that the "
    "reference gives differently.\n"
    "- It is INCORRECT only if it contradicts a specific fact in the "
    "reference, declines to answer, or answers a different question.\n\n"
    "List every DIRECT CONFLICT, one per line, as "
    "'CONFLICT: reference says X, candidate says Y'.\n"
    "A missing detail is NOT a conflict — do not list it. A conflict "
    "requires the candidate to STATE something incompatible with the "
    "reference. If there are none, write 'CONFLICT: none'.\n"
    "Then on the final line write exactly 'VERDICT: YES' or 'VERDICT: NO'."
)

# Enough room to list conflicts and still reach the verdict line. The terse
# judge ran at 8, which is what made reasoning impossible.
JUDGE_MAX_TOKENS = 300


def _sha256(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        # Length-prefixed separator: without it, ("ab", "c") and ("a",
        # "bc") hash identically, and this key is what decides whether two
        # answers get the same verdict.
        digest.update(b"\x00")
    return digest.hexdigest()


class JudgeConfigError(RuntimeError):
    """Judge configuration is missing or unusable. Never fall back to a
    weaker judge on this — a silently downgraded instrument produces
    numbers that look fine and are not."""


class JudgeCallError(RuntimeError):
    """The judge was configured but the call did not produce a verdict."""


# Read from the environment, never from a config file and never a literal
# in this repo. The key in particular must not reach disk, logs, cached
# verdicts or git; see _judge_record() for what a verdict is allowed to
# carry.
AZURE_ENV_VARS = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_API_VERSION",
)

# Bumped whenever the judge changes in a way that invalidates verdicts.
# Part of the cache key, so old verdicts cannot be read back. v1 was the
# Mistral/mock era.
JUDGE_CACHE_VERSION = "v4-azure-reasoned"


def _slugify(text: str) -> str:
    """Filesystem-safe identity fragment. Purely cosmetic — the cache key
    is the hash; this only decides which directory a verdict lands in."""
    return "".join(ch if ch.isalnum() or ch in "-._" else "-" for ch in text)[:64]


class AzureOpenaiJudge:
    """
    The Azure OpenAI judge, at temperature 0.

    Configuration is read from the environment ONLY (AZURE_ENV_VARS). A
    missing variable is a hard error naming the variable; there is no
    fallback to the system's Mistral client and no fallback to a mock,
    because both of those failure modes are invisible downstream — the
    sweep would complete, write a full results file, and be wrong.

    Transport is the openai SDK's AzureOpenAI client when it is installed,
    and a direct httpx POST to the deployment's chat/completions endpoint
    when it is not. Same request either way.
    """

    def __init__(self, api_key, endpoint, deployment, api_version):
        # Held in memory for the process lifetime and never written
        # anywhere. Not stored on any record, not printed, not logged.
        self._api_key = api_key
        self._endpoint = endpoint.rstrip("/")
        self.deployment = deployment
        self.api_version = api_version
        # Sticky request-shape flags, resolved on first use: newer Azure
        # deployments (o-series / reasoning models) reject `max_tokens` in
        # favour of `max_completion_tokens`, and some reject a temperature
        # override entirely. Discovering that once per process beats
        # failing the whole sweep on the first call.
        self._token_param = "max_tokens"
        self.temperature_supported = True

        try:
            from openai import AzureOpenAI  # noqa: PLC0415 -- optional dep, probed at construction
        except ImportError:
            self._sdk = None
            self.transport = "httpx"
        else:
            self._sdk = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=self._endpoint,
                api_version=api_version,
            )
            self.transport = "openai-sdk"

    @classmethod
    def from_env(cls) -> "AzureOpenaiJudge":
        missing = [name for name in AZURE_ENV_VARS if not (os.environ.get(name) or "").strip()]
        if missing:
            raise JudgeConfigError(
                "The E3 judge is Azure OpenAI and is not configured: "
                f"{', '.join(missing)} " + ("is" if len(missing) == 1 else "are")
                + " not set in the environment.\n"
                "Set every one of "
                f"{', '.join(AZURE_ENV_VARS)} and re-run, e.g.\n\n"
                "  export AZURE_OPENAI_API_KEY=...        # never commit this\n"
                "  export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com\n"
                "  export AZURE_OPENAI_DEPLOYMENT=<deployment-name>\n"
                "  export AZURE_OPENAI_API_VERSION=<api-version>\n\n"
                "There is deliberately no fallback judge: grading these "
                "answers with a weak model (or a mock) scores wrong answers "
                "correct, which makes skipping retrieval look free and hands "
                "the frontier to B0 by construction."
            )
        return cls(
            api_key=os.environ["AZURE_OPENAI_API_KEY"].strip(),
            endpoint=os.environ["AZURE_OPENAI_ENDPOINT"].strip(),
            deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"].strip(),
            api_version=os.environ["AZURE_OPENAI_API_VERSION"].strip(),
        )

    @property
    def identity(self) -> str:
        """What this judge IS, for the cache key and the verdict record.

        Deployment name and API version only. The endpoint is left out
        (it names the Azure resource) and the key obviously so. Two
        different deployments therefore get two different cache
        namespaces, which is the whole point."""
        return f"azure-openai/{self.deployment}@{self.api_version}"

    @property
    def slug(self) -> str:
        return _slugify(f"{JUDGE_CACHE_VERSION}-{self.deployment}-{self.api_version}")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """One graded call at temperature 0. Returns the raw text.

        Bounded retries on transient failures (rate limit, 5xx, dropped
        connection) because the sweep makes hundreds of judge calls over
        hours and a single 429 must not end it. Anything else is re-raised:
        a judge that is failing for a real reason should stop the run, not
        quietly leave answers ungraded."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        transient = 0
        for _ in range(6):  # >= 3 transient retries, plus room for shape adaptation
            try:
                return self._call(messages)
            except Exception as error:  # noqa: BLE001 -- classified below, then retried or re-raised
                if self._adapt_request_shape(error):
                    continue
                transient += 1
                if not _is_retryable(error) or transient > 3:
                    raise
                time.sleep(2 ** (transient - 1))
        raise JudgeCallError("The Azure judge call did not settle after retries.")

    def _call(self, messages) -> str:
        kwargs = {self._token_param: JUDGE_MAX_TOKENS}
        if self.temperature_supported:
            kwargs["temperature"] = 0
        if self._sdk is not None:
            response = self._sdk.chat.completions.create(
                model=self.deployment, messages=messages, **kwargs
            )
            return (response.choices[0].message.content or "").strip()
        return self._call_httpx(messages, kwargs)

    def _call_httpx(self, messages, kwargs) -> str:
        import httpx  # noqa: PLC0415 -- only needed when the openai SDK is absent

        url = (
            f"{self._endpoint}/openai/deployments/{self.deployment}"
            f"/chat/completions?api-version={self.api_version}"
        )
        response = httpx.post(
            url,
            # The key travels in the header and nowhere else. Do not add it
            # to the URL: URLs end up in exception strings and proxy logs.
            headers={"api-key": self._api_key, "Content-Type": "application/json"},
            json={"messages": messages, **kwargs},
            timeout=60.0,
        )
        if response.status_code >= 400:
            raise _HttpJudgeError(response.status_code, response.text)
        payload = response.json()
        return (payload["choices"][0]["message"].get("content") or "").strip()

    def _adapt_request_shape(self, error) -> bool:
        """Some deployments reject `max_tokens` (want `max_completion_tokens`)
        or reject any temperature override. Adapt once and retry, rather
        than failing a multi-hour sweep on the request shape. Returns True
        if something was changed and the call is worth retrying."""
        text = str(error).lower()
        if "max_tokens" in text and self._token_param == "max_tokens":
            self._token_param = "max_completion_tokens"
            print("[judge] deployment rejects max_tokens; retrying with max_completion_tokens")
            return True
        if "temperature" in text and self.temperature_supported:
            self.temperature_supported = False
            print(
                "[judge] deployment rejects a temperature override; retrying at its "
                "default. NOTE: verdicts from this deployment are not "
                "temperature-0 and are recorded as such."
            )
            return True
        return False


class _HttpJudgeError(RuntimeError):
    def __init__(self, status, body):
        self.status = status
        # Azure error bodies echo the request, never the api-key header.
        super().__init__(f"HTTP {status}: {body[:500]}")


def _is_retryable(error) -> bool:
    status = getattr(error, "status", None) or getattr(error, "status_code", None)
    if status in (408, 409, 429, 500, 502, 503, 504):
        return True
    name = type(error).__name__
    return name in (
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "RemoteProtocolError",
    )


def parse_verdict(raw: str) -> tuple:
    """(answer_correct, parsed_ok) from the judge's raw reply.

    The judge now reasons before it answers, so the verdict is the LAST
    line containing "VERDICT:" rather than the first token. Last, not
    first: the conflict list above it quotes the reference and the
    candidate, and either can contain the word "verdict" or a stray YES.

    The bare-token form is still accepted as a fallback so a verdict cached
    under the pre-reasoning prompt stays readable.
    """
    text = (raw or "").strip()
    lines = [line for line in text.splitlines() if "VERDICT" in line.upper()]
    if lines:
        tail = lines[-1].upper().split("VERDICT:")[-1].strip().lstrip("*\"' \n\t.:-")
        if tail.startswith("YES"):
            return True, True
        if tail.startswith("NO"):
            return False, True
        return False, False

    token = text.upper().lstrip("*\"' \n\t.:-")[:3]
    if token.startswith("YES"):
        return True, True
    if token.startswith("NO"):
        return False, True
    return False, False
