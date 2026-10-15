"""Versioned judge rubrics for E1-v2.

  e1v2-reasoned-strip-r1  E3's v4 judge generalised off the hotel (built in
                          judges.py from simulator/judge_core.py). Kept so the
                          first d2_support dev grading stays reproducible.
  e1v2-r2                 text in rubrics/e1v2-r2/<dataset>.txt (default.txt
                          for any other dataset, which is r1's text):
                          d2_support is CORRECT iff the answer states EVERY
                          key fact (paraphrase and unit/format variants
                          allowed) and contradicts nothing in the reference.
                          Deferring, declining and key-fact-free hedges are
                          INCORRECT. The judge also sees the dialogue history.

  e1v2-r3                 the second and final dev revision (PREREG_V2 2.4):
                          d2_support is CORRECT iff the answer states every
                          CORE fact (meta.core_facts, a verbatim subset of
                          key_facts; paraphrase, unit/format variants and
                          close approximations allowed) and contradicts
                          nothing in the reference or the supplementary
                          facts, which may be omitted. Without core_facts
                          the first key fact stands in, with a warning.

Why r3: r2's "every key fact" failed correct general answers that omitted a
secondary fact (d2_support dev: general items needed 65/84 under r2 vs 26/84
under r1, empirical need 201/222).

Why r2: under r1 the inherited rule "omitting reference detail is correct"
credited non-answers on support questions. gpt-6.1-sol passed vague hedges
("our policy typically covers burst pipes"), gpt-4o failed correct general
answers, and the two agreed at kappa 0.52 on d2_support dev.

The rubric's sha256 (file texts with line endings normalised, plus the
prompt layout version) is the judge identity's prompt_sha256, so it enters
every cache key: editing a rubric file can never reuse old verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llm.experiment1_v2 import common

RUBRIC_DIR = Path(__file__).resolve().parent / "rubrics"
R1 = "e1v2-reasoned-strip-r1"
LAYOUT_VERSION = "layout-2"  # history block + key facts always shown when the rubric needs them


def _history_lines(history) -> str:
    lines = []
    for index, turn in enumerate(history or []):
        if isinstance(turn, str):
            role, text = ("Caller" if index % 2 == 0 else "Agent"), turn
        else:
            raw = str(turn.get("role") or turn.get("speaker") or "").lower()
            role = "Caller" if raw in ("user", "caller", "customer", "human") else "Agent"
            text = turn.get("content", turn.get("text", ""))
        lines.append(f"{role}: {' '.join(str(text).split())}")
    return "\n".join(lines)


def core_facts_or_fallback(core_facts, key_facts, reference) -> list:
    """The core facts to grade against (r3). Without an annotation, the
    first key fact stands in (callers log that); without key facts either,
    the reference itself."""
    if core_facts:
        return list(core_facts)
    if key_facts:
        return [key_facts[0]]
    return [reference]


@dataclass(frozen=True)
class FileRubric:
    version: str
    texts: dict           # {dataset or "default": instructions}
    sha256: str

    def instructions(self, dataset) -> str:
        return self.texts.get(dataset) or self.texts["default"]

    def requires_key_facts(self, dataset) -> bool:
        return dataset in self.texts and "KEY FACT" in self.texts[dataset]

    def uses_core_facts(self, dataset) -> bool:
        return dataset in self.texts and "CORE FACT" in self.texts[dataset]

    def user_prompt(self, *, dataset, question, reference, answer, key_facts=None, aliases=None,
                    history=None, core_facts=None) -> str:
        from llm.experiment1_v2.judges import KEY_FACTS_RULE  # noqa: PLC0415 -- avoid a cycle

        instructions = self.instructions(dataset).rstrip("\n")
        facts = list(key_facts or [])
        if self.uses_core_facts(dataset):
            # r3: core facts must all be stated; the other key facts are
            # supplementary (may be omitted, must not be contradicted).
            core = core_facts_or_fallback(core_facts, facts, reference)
            supplementary = [f for f in facts if f not in core]
            parts = [instructions]
            if history:
                parts.append("Conversation before the question:\n" + _history_lines(history))
            parts += [f"Question:\n{question}", f"Reference answer:\n{reference}"]
            if aliases:
                parts.append("Equivalent forms of the reference answer:\n"
                             + "\n".join(f"- {a}" for a in aliases))
            parts.append("Core facts:\n" + "\n".join(f"- {f}" for f in core))
            if supplementary:
                parts.append("Supplementary facts (may be omitted, must not be contradicted):\n"
                             + "\n".join(f"- {f}" for f in supplementary))
            parts.append(f"Candidate answer:\n{answer}")
            return "\n\n".join(parts) + "\n"
        if self.requires_key_facts(dataset) and not facts:
            facts = [reference]  # no listed facts: the reference itself is the fact
        elif facts and not self.requires_key_facts(dataset):
            # r1-style rubric with key facts: same insertion as r1.
            head, sep, tail = instructions.partition("\n\nList every DIRECT CONFLICT")
            instructions = head + "\n" + KEY_FACTS_RULE.rstrip("\n") + sep + tail
        parts = [instructions]
        if history:
            parts.append("Conversation before the question:\n" + _history_lines(history))
        parts += [f"Question:\n{question}", f"Reference answer:\n{reference}"]
        if aliases:
            parts.append("Equivalent forms of the reference answer:\n"
                         + "\n".join(f"- {a}" for a in aliases))
        if facts:
            parts.append("Key facts:\n" + "\n".join(f"- {f}" for f in facts))
        parts.append(f"Candidate answer:\n{answer}")
        return "\n\n".join(parts) + "\n"


def load_file_rubric(version: str) -> FileRubric:
    folder = RUBRIC_DIR / version
    if not folder.is_dir():
        raise ValueError(f"unknown rubric version {version!r}: no {folder}")
    texts = {}
    for path in sorted(folder.glob("*.txt")):
        # Bytes, normalised here: the hash must not depend on how git or the
        # OS translated line endings on checkout.
        texts[path.stem] = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
    if "default" not in texts:
        raise ValueError(f"rubric {version} has no default.txt")
    from llm.experiment1_v2.judges import system_prompt  # noqa: PLC0415

    digest = common.sha256_text(version, LAYOUT_VERSION, system_prompt("{context}"),
                                *[f"{name}\n{texts[name]}" for name in sorted(texts)])
    return FileRubric(version, texts, digest)


_CACHE: dict = {}


def get_rubric(version: str):
    """The rubric object for a prompt version: None means r1 (built in code)."""
    if version == R1:
        return None
    if version not in _CACHE:
        _CACHE[version] = load_file_rubric(version)
    return _CACHE[version]


def rubric_sha256(version: str) -> str:
    from llm.experiment1_v2.judges import PROMPT_SHA256  # noqa: PLC0415

    rubric = get_rubric(version)
    return PROMPT_SHA256 if rubric is None else rubric.sha256
