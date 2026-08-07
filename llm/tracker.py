"""
tracker.py — the state tracker.

Each turn, this writes down what the user wants as a small structured form:

    {"intent": "book_spa", "room": "unknown", "time": "4pm",
     "people": 2, "confirmed": false}

The slot names and their defaults live in configs/config.yaml under
llm.tracker.slots, not in this file.

Two rules from llm/README.md that shape everything here:

1. Fixed schema. Every output has exactly the configured keys, always, so
   downstream code can index into it without defensive checks.
2. "unknown" is a correct value, not a failure. Most slots are unknown early
   in a call. A tracker that guesses to avoid saying "unknown" is worse than
   useless — the controller would act on invented state.

On logging: the tracker does NOT write an event. The frozen schema
(eventlog/schema.md) has no state-tracker event type, and inventing one
locally is exactly what the freeze exists to prevent. Its internal model call
is also deliberately made without a log_writer, so it doesn't emit
llm_first_token / llm_final and pollute the conversation timeline with a call
that isn't the assistant speaking.

  ACTION FOR WEEK 2 (Renya): to record tracker output in the timeline we need
  a `state_update` event type added to eventlog/schema.md first.

Run it:
    python llm/tracker.py
"""

import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm.client import make_client
from llm.config import load_config


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------
# Strict, short, and it states the "unknown" rule twice, because the failure
# mode we actually see is a model filling a slot with a plausible guess rather
# than admitting it wasn't said.
_INSTRUCTION = """You extract structured state from a hotel guest conversation.

Output ONLY a single JSON object. No explanation, no markdown fences, no text
before or after it.

The object must have exactly these keys: {keys}

Rules:
- Use only information the guest actually stated. Never infer or guess.
- If a value was not stated, its value is "unknown". That is the correct
  answer, not a failure.
- "confirmed" is true only if the guest explicitly agreed to a specific
  booking or change. Otherwise false.
- Keep values short: "4pm", "spa", 2. Numbers as numbers, not words."""


def _slot_defaults(cfg: dict) -> dict:
    """The configured schema, as a fresh dict of default values."""
    return dict((cfg.get("tracker") or {}).get("slots") or {})


def build_messages(turns, cfg: dict) -> list:
    """
    Assemble the tracker prompt.

    `turns` is the conversation so far: either a list of {"role", "content"}
    dicts, or a list of plain strings taken to be user utterances in order.
    """
    defaults = _slot_defaults(cfg)
    keys = ", ".join(defaults)

    lines = []
    for turn in turns:
        if isinstance(turn, dict):
            role = turn.get("role", "user")
            lines.append(f"{role}: {turn.get('content', '')}")
        else:
            lines.append(f"user: {turn}")
    transcript = "\n".join(lines) if lines else "(nothing said yet)"

    return [
        {"role": "system", "content": _INSTRUCTION.format(keys=keys)},
        {"role": "user", "content": f"Conversation so far:\n{transcript}\n\nJSON:"},
    ]


# ---------------------------------------------------------------------------
# Parsing what came back
# ---------------------------------------------------------------------------
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _extract_json_object(raw: str):
    """
    Pull the first complete JSON object out of a model response.

    Models wrap JSON in markdown fences or add a sentence of preamble even
    when told not to, so we do not hand `raw` straight to json.loads. We strip
    fences, then scan for the first balanced {...}, ignoring braces that sit
    inside string literals.

    Returns the parsed dict, or None if there wasn't a usable object.
    """
    if not raw:
        return None
    text = _FENCE.sub("", raw.strip())

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def _coerce_to_schema(parsed, defaults: dict, numeric_slots=()) -> dict:
    """
    Force whatever the model produced into the fixed schema.

    Missing keys get their default. Extra keys the model invented are dropped
    — the schema is fixed, and letting a stray key through would break the
    "downstream code can index without checking" guarantee. Empty strings and
    null are treated as unknown, because a model writing "" means the same
    thing as it writing "unknown".
    """
    slots = dict(defaults)
    if not isinstance(parsed, dict):
        return slots

    for key, default in defaults.items():
        if key not in parsed:
            continue
        value = parsed[key]

        if value is None or (isinstance(value, str) and not value.strip()):
            continue  # keep the default

        if isinstance(default, bool):
            # A boolean slot must end up boolean. Models often return the
            # string "true" or "yes" instead of a JSON bool.
            if isinstance(value, bool):
                slots[key] = value
            elif isinstance(value, str):
                slots[key] = value.strip().lower() in {"true", "yes", "y", "1"}
            else:
                slots[key] = bool(value)
            continue

        if isinstance(value, str):
            value = value.strip()
            # Count slots often come back quoted: "2" -> 2. Only the slots
            # config names as numeric, so room "412A" survives intact.
            if key in numeric_slots and value.isdigit():
                value = int(value)
        slots[key] = value

    return slots


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def track(turns, client=None, cfg=None) -> dict:
    """
    Return the user's goal as a dict matching the configured slot schema.

    Always returns a complete, valid dict — including when the model returns
    something unparseable, in which case every slot is its default. The
    pipeline can never be handed a half-formed state object.
    """
    slots, _raw, _ok = track_detailed(turns, client=client, cfg=cfg)
    return slots


def track_detailed(turns, client=None, cfg=None):
    """
    Same as track(), but also returns the raw model text and whether it
    parsed. Use this when debugging prompt behaviour or measuring how often
    the model breaks format.

    Returns (slots, raw_text, parsed_ok).
    """
    cfg = cfg or load_config()
    client = client or make_client(cfg)
    defaults = _slot_defaults(cfg)
    tracker_cfg = cfg.get("tracker") or {}

    messages = build_messages(turns, cfg)

    # No log_writer on purpose — see the module docstring. Temperature 0:
    # this is extraction, and we want the same conversation to produce the
    # same state every time it's replayed.
    gen = client.generate(
        messages,
        temperature=tracker_cfg.get("temperature", 0.0),
        max_tokens=tracker_cfg.get("max_tokens", 200),
        mock_answer=_mock_json(turns, defaults),
    )

    parsed = _extract_json_object(gen.text)
    if parsed is None:
        print(
            f"[tracker] WARNING: could not parse JSON from model output - "
            f"returning all defaults. Raw was: {gen.text[:200]!r}"
        )
        return dict(defaults), gen.text, False

    numeric_slots = set(tracker_cfg.get("numeric_slots") or ())
    return _coerce_to_schema(parsed, defaults, numeric_slots), gen.text, True


def uncertainty(slots: dict, cfg=None, parsed_ok: bool = True) -> dict:
    """Return the tracker's Phase 2 uncertainty signal.

    The tracker does not produce calibrated probabilities. Its honest signal
    is the fraction of configured slots that remain ``"unknown"``. Boolean
    slots such as ``confirmed`` are excluded because ``False`` is a real
    state, not missing information. Higher values mean more uncertainty.

    This function is side-effect free so the integration owner can add the
    returned payload to the shared belief/event once the event schema has an
    approved home for it.
    """
    cfg = cfg or load_config()
    defaults = _slot_defaults(cfg)
    measured_slots = [
        key for key, default in defaults.items() if not isinstance(default, bool)
    ]
    values = slots or {}
    unknown_slots = [
        key for key in measured_slots if values.get(key, "unknown") == "unknown"
    ]
    fraction = len(unknown_slots) / len(measured_slots) if measured_slots else 0.0
    return {
        "unknown_slots": unknown_slots,
        "unknown_count": len(unknown_slots),
        "measured_slot_count": len(measured_slots),
        "unknown_fraction": fraction,
        "intent_unknown": values.get("intent", "unknown") == "unknown",
        "parsed_ok": bool(parsed_ok),
    }


def _mock_json(turns, defaults: dict) -> str:
    """
    What MockClient should 'say' when the tracker calls it.

    Ignored entirely by VLLMClient (it's swallowed as an unused override), so
    it costs nothing in real runs. It exists so the parsing, coercion, and
    schema-enforcement paths above can be exercised end to end without a GPU.
    """
    slots = dict(defaults)
    text = " ".join(
        (t.get("content", "") if isinstance(t, dict) else str(t)) for t in turns
    ).lower()

    if "spa" in text:
        slots["intent"] = "book_spa"
    elif "pool" in text:
        slots["intent"] = "pool_hours"
    elif "check" in text:
        slots["intent"] = "checkout_time"

    match = re.search(r"\b(\d{1,2})\s*(am|pm)\b", text)
    if match:
        slots["time"] = f"{match.group(1)}{match.group(2)}"

    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
    match = re.search(r"\b(?:for|party of)\s+(\d+|" + "|".join(words) + r")\b", text)
    if match:
        found = match.group(1)
        slots["people"] = int(found) if found.isdigit() else words[found]

    match = re.search(r"\broom\s+(\d+)\b", text)
    if match:
        slots["room"] = match.group(1)

    if any(w in text for w in ("yes please", "book it", "that works", "confirm")):
        slots["confirmed"] = True

    # Wrapped in a fence deliberately: the real models do this even when told
    # not to, so the mock should too, or _extract_json_object never gets tested.
    return "```json\n" + json.dumps(slots) + "\n```"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = load_config()
    client = make_client(cfg)

    print(f"backend: {cfg['backend']}   model: {cfg['model']}")
    print(f"slots  : {list(_slot_defaults(cfg))}\n")

    conversations = [
        ["hi, what time does the pool close"],
        ["I'd like to book the spa", "for two people at 4pm"],
        [
            "can I get a late checkout for room 412",
            "yes please, book it",
        ],
        ["uh, hello?"],  # nothing stated: everything should stay unknown
    ]

    all_valid = True
    for turns in conversations:
        slots, raw, ok = track_detailed(turns, client=client, cfg=cfg)

        # The deliverable is "outputs valid, parseable JSON each turn", so we
        # check the output actually survives a round trip, not just that it
        # looks like a dict.
        try:
            json.loads(json.dumps(slots))
            valid = ok and set(slots) == set(_slot_defaults(cfg))
        except (TypeError, ValueError):
            valid = False
        all_valid = all_valid and valid

        print(f"  turns : {turns}")
        print(f"  slots : {json.dumps(slots)}")
        print(f"  parsed: {ok}   schema ok: {valid}\n")

    print("all turns produced valid, schema-conforming JSON" if all_valid else "FAILED")
