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


@dataclass(frozen=True)
class FileRubric:
    version: str
    texts: dict           # {dataset or "default": instructions}
    sha256: str

    def instructions(self, dataset) -> str:
        return self.texts.get(dataset) or self.texts["default"]

    def requires_key_facts(self, dataset) -> bool:
        return dataset in self.texts and "KEY FACT" in self.texts[dataset]

    def user_prompt(self, *, dataset, question, reference, answer, key_facts=None, aliases=None,
                    history=None) -> str:
        from llm.experiment1_v2.judges import KEY_FACTS_RULE  # noqa: PLC0415 -- avoid a cycle

        instructions = self.instructions(dataset).rstrip("\n")
        facts = list(key_facts or [])
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
