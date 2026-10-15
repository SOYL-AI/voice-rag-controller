"""Measure the token budget of the skip vs retrieve paths for the real prompts.

Retrieval's LLM-side cost is the extra PREFILL from the passages it adds to
the prompt (and any change in answer length). This script measures both on
the project's own data with the production model's own tokenizer:

* prompts: llm/client.py `build_messages(question, passages)` for each of the
  50 scenarios in simulator/scenarios.json, with passages = [] (skip) and the
  top-k (k=5, retrieval/search.py default) bge-small-en-v1.5 + FAISS
  IndexFlatL2 results over retrieval/corpus (retrieve);
* answers: the 500 cached Mistral-7B answers in simulator/e3_sweep_cache/
  answers/ (50 scenarios x 5 seeds x {skip, retrieve}).

Tokenizer: Mistral-7B-Instruct-v0.3's v3 SentencePiece tokenizer as shipped
in `mistral_common` (mistral_instruct_tokenizer_240323.model.v3), with the
instruct chat encoding (system prompt folded into the user turn as the v0.3
template does). The HF repo is gated, so this is the ungated exact copy.

    PYTHONUTF8=1 .venv-energy/Scripts/python.exe -m energy.tokens
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from energy.paths import model_dir  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "token_budget.json"
K = 5


def mistral_tokenizer():
    import mistral_common
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    path = (Path(mistral_common.__file__).parent / "data"
            / "mistral_instruct_tokenizer_240323.model.v3")
    return MistralTokenizer.from_file(str(path))


def chat_prompt_tokens(tok, messages) -> int:
    """Token count of the instruct-encoded chat (BOS [INST] system

user [/INST])."""
    from mistral_common.protocol.instruct.messages import SystemMessage, UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest

    msgs = [SystemMessage(content=m["content"]) if m["role"] == "system"
            else UserMessage(content=m["content"]) for m in messages]
    return len(tok.encode_chat_completion(ChatCompletionRequest(messages=msgs)).tokens)


def text_tokens(tok, text: str) -> int:
    """Generated-answer tokens, counting the EOS the model emits to stop."""
    return len(tok.instruct_tokenizer.tokenizer.encode(text.strip(), bos=False, eos=False)) + 1


def main():
    import faiss
    from sentence_transformers import SentenceTransformer

    # llm/client.py is importable without network/GPU (mock backend default)
    from llm.client import build_messages

    tok = mistral_tokenizer()
    corpus_dir = REPO / "retrieval" / "corpus"
    docs = [(p.name, p.read_text(encoding="utf-8")) for p in sorted(corpus_dir.glob("*.txt"))]
    emb = SentenceTransformer(str(model_dir("BAAI/bge-small-en-v1.5")), device="cpu")
    doc_vecs = emb.encode([d[1] for d in docs], convert_to_numpy=True)
    index = faiss.IndexFlatL2(doc_vecs.shape[1])
    index.add(doc_vecs)

    scenarios = json.loads((REPO / "simulator" / "scenarios.json").read_text(encoding="utf-8"))
    rows = []
    for sc in scenarios:
        q = sc["question"]
        _, idx = index.search(emb.encode([q], convert_to_numpy=True), K)
        passages = [docs[i][1] for i in idx[0]]
        skip_t = chat_prompt_tokens(tok, build_messages(q, []))
        ret_t = chat_prompt_tokens(tok, build_messages(q, passages))
        rows.append({"id": sc["id"], "retrieval_needed": sc.get("retrieval_needed"),
                     "prompt_skip": skip_t, "prompt_retrieve": ret_t,
                     "passage_tokens": [text_tokens(tok, p) - 1 for p in passages]})

    ans_dir = REPO / "simulator" / "e3_sweep_cache" / "answers"
    gen = {"skip": [], "retrieve": []}
    paired = []
    by_key = {}
    n_chars = n_toks = 0
    for f in sorted(ans_dir.glob("*.json")):
        a = json.loads(f.read_text(encoding="utf-8"))
        n = text_tokens(tok, a["answer_text"])
        gen[a["path"]].append(n)
        n_chars += len(a["answer_text"].strip())
        n_toks += n
        by_key[(a["scenario_id"], a["seed"], a["path"])] = n
    for (sid, seed, path), n in by_key.items():
        if path == "retrieve" and (sid, seed, "skip") in by_key:
            paired.append(n - by_key[(sid, seed, "skip")])

    def summ(xs):
        xs = np.asarray(xs, dtype=float)
        return {"n": int(len(xs)), "mean": float(xs.mean()), "sd": float(xs.std(ddof=1)),
                "median": float(np.median(xs)), "p10": float(np.percentile(xs, 10)),
                "p90": float(np.percentile(xs, 90)), "min": float(xs.min()), "max": float(xs.max())}

    extra = [r["prompt_retrieve"] - r["prompt_skip"] for r in rows]
    out = {
        "tokenizer": "mistral_common mistral_instruct_tokenizer_240323.model.v3 (Mistral-7B-Instruct-v0.3)",
        "k_passages": K,
        "corpus": "retrieval/corpus (60 hotel passages)",
        "prompt_skip": summ([r["prompt_skip"] for r in rows]),
        "prompt_retrieve": summ([r["prompt_retrieve"] for r in rows]),
        "extra_prefill_tokens": summ(extra),
        "passage_tokens": summ([t for r in rows for t in r["passage_tokens"]]),
        "gen_skip": summ(gen["skip"]),
        "gen_retrieve": summ(gen["retrieve"]),
        "gen_retrieve_minus_skip_paired": summ(paired),
        "chars_per_gen_token": n_chars / n_toks,
        "per_scenario": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    for k in ("prompt_skip", "prompt_retrieve", "extra_prefill_tokens", "passage_tokens",
              "gen_skip", "gen_retrieve", "gen_retrieve_minus_skip_paired"):
        print(f"{k:32s} mean {out[k]['mean']:7.1f}  median {out[k]['median']:7.1f}  "
              f"p10 {out[k]['p10']:6.1f}  p90 {out[k]['p90']:6.1f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
