"""Chat messages for E1-v2 closed-book and grounded generations.

Templates live in config.json ("prompts", "history"); this module only fills
them. d2_support items get the support-agent system prompt with the item's
org/domain and their `history` turns; d1_open gets a neutral assistant
prompt. The prefix text replaces the question in the LAST user turn only.
"""

from __future__ import annotations

_USER_ROLES = {"user", "caller", "customer", "human"}
_ASSISTANT_ROLES = {"assistant", "agent", "bot", "system_agent"}


def system_prompt(item: dict, prompts_cfg: dict) -> str:
    entry = prompts_cfg.get(item.get("dataset")) or prompts_cfg["default"]
    return entry["system"].format(
        org=item.get("org") or "the company",
        domain=item.get("domain") or "service",
    )


def _turn(turn, index: int) -> tuple:
    """(role, text) from a history turn. Accepts {"role","content"|"text"}
    dicts or bare strings (alternating, starting with the caller)."""
    if isinstance(turn, str):
        return ("user" if index % 2 == 0 else "assistant"), turn
    role = str(turn.get("role") or turn.get("speaker") or "").lower()
    text = turn.get("content", turn.get("text", ""))
    if role in _USER_ROLES:
        return "user", text
    if role in _ASSISTANT_ROLES:
        return "assistant", text
    raise ValueError(f"history turn has unknown role {role!r}: {turn!r}")


def normalize_history(history, history_cfg: dict) -> tuple:
    """(system_suffix, [messages]) with strict user/assistant alternation
    starting at user. Same-role neighbours are merged; a leading assistant
    turn (a greeting) is folded into the system prompt."""
    turns = [_turn(t, i) for i, t in enumerate(history or [])]
    turns = [(r, (t or "").strip()) for r, t in turns if (t or "").strip()]
    suffix = ""
    while turns and turns[0][0] == "assistant":
        suffix += history_cfg.get(
            "leading_assistant_template", '\n\nEarlier in this call you said: "{text}"'
        ).format(text=turns[0][1])
        turns = turns[1:]
    merged = []
    for role, text in turns:
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] += "\n" + text
        else:
            merged.append({"role": role, "content": text})
    return suffix, merged


def build_messages(item: dict, prefix_text: str, cfg: dict, passages=None) -> list:
    prompts_cfg = cfg["prompts"]
    system = system_prompt(item, prompts_cfg)
    suffix, history = normalize_history(item.get("history"), cfg.get("history") or {})
    system += suffix
    user = prefix_text
    if passages is not None:
        system += prompts_cfg.get("grounded_system_suffix", "")
        rendered = "\n".join(
            prompts_cfg.get("passage_template", "- {text}").format(
                text=" ".join(str(p.get("text", "")).split()), doc_id=p.get("doc_id")
            )
            for p in passages
        )
        user = prompts_cfg["grounded_user_template"].format(passages=rendered, question=prefix_text)
    messages = [{"role": "system", "content": system}] + history
    if messages[-1]["role"] == "user":
        # History ended on the caller: the question continues that turn.
        messages[-1] = {"role": "user", "content": messages[-1]["content"] + "\n" + user}
    else:
        messages.append({"role": "user", "content": user})
    return messages
