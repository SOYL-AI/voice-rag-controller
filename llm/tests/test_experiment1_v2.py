"""Tests for llm/experiment1_v2 (E1-v2). Mock backend and mock judges only."""

import asyncio
import json
import math
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from llm.client import AlternativeLogprob, TokenLogprob
from llm.experiment1.run import roc_auc
from llm.experiment1_v2 import analyze, common, convert_v1, judges, label, prefixes, prompts, run
from llm.experiment1_v2 import stats
from llm.experiment1_v2.backends import MockAsyncBackend
from llm.experiment1_v2.signals_v2 import first_k_arrival_ms, first_k_mean_entropy
from simulator.judge_validation import score_judges

SENTINEL_URL = "http://sentinel-endpoint.invalid/v1"
SENTINEL_KEY = "sk-sentinel-e1v2-0000"


def _token(probabilities):
    """A TokenLogprob whose returned top-k is exactly `probabilities`."""
    chosen = math.log(probabilities[0])
    alternatives = [AlternativeLogprob(f"t{i}", math.log(p)) for i, p in enumerate(probabilities)]
    return TokenLogprob("t0", chosen, alternatives)


def _make_dataset(tmp: Path, traces=True) -> Path:
    convert_v1.main(["--out", str(tmp / "data"), "--all-dev"] + (["--plumbing-traces"] if traces else []))
    return tmp / "data" / "items.jsonl"


# ---------------------------------------------------------------------------
class PrefixTests(unittest.TestCase):
    def test_word_fraction_counts_floor_with_minimum_one(self):
        self.assertEqual(prefixes.n_prefix_words(8, 0.25), 2)
        self.assertEqual(prefixes.n_prefix_words(8, 0.5), 4)
        self.assertEqual(prefixes.n_prefix_words(7, 0.75), 5)
        self.assertEqual(prefixes.n_prefix_words(3, 0.25), 1)
        self.assertEqual(prefixes.n_prefix_words(0, 0.5), 0)

    def test_text_prefixes(self):
        out = prefixes.text_prefixes("What  is the  late fee?", (0.25, 0.5, 0.75))
        self.assertEqual(out[0]["kind"], "full")
        self.assertEqual(out[0]["text"], "What is the late fee?")
        self.assertEqual([p["text"] for p in out[1:]], ["What", "What is", "What is the"])
        self.assertTrue(all(p["source"] == "text" for p in out))

    def test_identical_texts_share_one_call_but_keep_their_conditions(self):
        item = {"id": "x", "question": "Hi there"}
        conds = prefixes.item_prefixes(item, None, (0.25, 0.5, 0.75), asr=False)
        groups = prefixes.group_by_text(conds)
        self.assertEqual(len(conds), 4)
        self.assertEqual(len(groups), 2)  # "Hi there" and "Hi" (x3 fractions)
        hi = dict(groups)["Hi"]
        self.assertEqual(sorted(p["word_frac"] for p in hi), [0.25, 0.5, 0.75])
        self.assertEqual(len({p["prefix_id"] for p in conds}), 4)

    def test_asr_prefixes_skip_empty_and_dedup_repeats(self):
        trace = {
            "item_id": "x", "voice_id": "v", "noise": "clean", "vad_user_end_ms": 1500,
            "words": [{"w": "what", "start_ms": 0, "end_ms": 400},
                      {"w": "time", "start_ms": 400, "end_ms": 800}],
            "partials": [{"t_ms": 300, "text": ""}, {"t_ms": 600, "text": "what"},
                         {"t_ms": 900, "text": "what", "available_ms": 1000},
                         {"t_ms": 1200, "text": "what time"}],
            "asr_final": {"t_ms": 1600, "text": "What time?"},
        }
        out = prefixes.asr_prefixes(trace)
        self.assertEqual([p["text"] for p in out], ["what", "what", "what time", "What time?"])
        self.assertEqual(out[0]["available_ms"], 600)  # defaults to t_ms
        self.assertEqual(out[1]["available_ms"], 1000)
        self.assertEqual(out[0]["word_frac"], 0.5)
        self.assertEqual(out[-1]["kind"], "full")
        conds = prefixes.item_prefixes({"id": "x", "question": "what time"}, [trace], (), text=False)
        self.assertEqual(len(prefixes.group_by_text(conds)), 3)


class PromptTests(unittest.TestCase):
    CFG = {"prompts": {"d2_support": {"system": "Agent for {org}, a {domain} company."},
                       "default": {"system": "Neutral."},
                       "grounded_system_suffix": " Use passages.",
                       "grounded_user_template": "P:\n{passages}\n\n{question}"},
           "history": {"leading_assistant_template": " [you said: {text}]"}}

    def test_support_prompt_with_history_alternates(self):
        item = {"dataset": "d2_support", "org": "Northwind", "domain": "telecom",
                "history": [{"role": "agent", "content": "Hello!"},
                            {"role": "caller", "content": "Hi"},
                            {"role": "caller", "content": "I have a bill question"},
                            {"role": "agent", "content": "Sure"}]}
        messages = prompts.build_messages(item, "What's the late fee?", self.CFG)
        roles = [m["role"] for m in messages]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        self.assertIn("Northwind, a telecom company", messages[0]["content"])
        self.assertIn("[you said: Hello!]", messages[0]["content"])
        self.assertEqual(messages[1]["content"], "Hi\nI have a bill question")
        self.assertEqual(messages[-1]["content"], "What's the late fee?")

    def test_open_domain_and_grounded(self):
        messages = prompts.build_messages({"dataset": "d1_open"}, "Who?", self.CFG,
                                          passages=[{"text": "A  b", "doc_id": "d"}])
        self.assertEqual(messages[0]["content"], "Neutral. Use passages.")
        self.assertEqual(messages[1]["content"], "P:\n- A b\n\nWho?")


class FirstKEntropyTests(unittest.TestCase):
    def test_first_k_mean_entropy(self):
        tokens = [_token([0.5, 0.5]), _token([1.0]), _token([0.25] * 4), _token([1.0])]
        self.assertAlmostEqual(tokens[0].entropy_nats, math.log(2))
        self.assertAlmostEqual(tokens[1].entropy_nats, 0.0)
        self.assertAlmostEqual(first_k_mean_entropy(tokens, 1), math.log(2))
        self.assertAlmostEqual(first_k_mean_entropy(tokens, 2), math.log(2) / 2)
        self.assertAlmostEqual(first_k_mean_entropy(tokens, 3), (math.log(2) + math.log(4)) / 3)
        # k beyond the answer length averages what exists
        self.assertAlmostEqual(first_k_mean_entropy(tokens, 16), (math.log(2) + math.log(4)) / 4)
        self.assertIsNone(first_k_mean_entropy([], 4))
        self.assertIsNone(first_k_mean_entropy([TokenLogprob("x", -0.1)], 4))  # no top-k

    def test_first_k_arrival(self):
        out = first_k_arrival_ms([10.0, 20.0, 30.0, 40.0, 50.0], (4, 8))
        self.assertEqual(out, {"4": 40.0, "8": None})


# ---------------------------------------------------------------------------
class SealingTests(unittest.TestCase):
    def test_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(common.SealedSplitError):
                common.check_split_allowed("test", Path(tmp))
            common.check_split_allowed("dev", Path(tmp))
            marker = Path(tmp) / common.PREREG_FROZEN_RELPATH
            marker.parent.mkdir(parents=True)
            marker.write_text("frozen")
            common.check_split_allowed("test", Path(tmp))
            with self.assertRaises(ValueError):
                common.check_split_allowed("train", Path(tmp))

    def test_entry_points_refuse_test_before_doing_anything(self):
        self.assertFalse((common.REPO_ROOT / common.PREREG_FROZEN_RELPATH).exists())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            # A non-existent items path proves nothing is loaded first.
            missing = str(Path(tmp) / "nope" / "items.jsonl")
            self.assertEqual(run.main(["--items", missing, "--split", "test", "--backend", "mock",
                                       "--out-dir", str(out)]), 3)
            self.assertEqual(label.main(["--items", missing, "--split", "test", "--model", "m"]), 3)
            self.assertEqual(analyze.main(["--items", missing, "--split", "test"]), 3)
            self.assertFalse(out.exists())

    def test_split_selection_excludes_test_and_checks_consistency(self):
        items = [{"id": "a", "split": "dev"}, {"id": "b", "split": "test"}]
        self.assertEqual([i["id"] for i in common.select_split(items, {"dev": ["a"], "test": ["b"]}, "dev")], ["a"])
        with self.assertRaises(ValueError):
            common.select_split(items, {"dev": ["a", "b"], "test": []}, "dev")

    def test_contract_split_rule_keeps_pairs_together(self):
        self.assertEqual(common.contract_split("x-1", "pair-9"), common.contract_split("y-2", "pair-9"))
        share = np.mean([common.contract_split(f"id-{i}") == "dev" for i in range(4000)])
        self.assertAlmostEqual(share, 0.3, delta=0.03)

    def test_synthetic_traces_refused_for_real_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            items = _make_dataset(Path(tmp))
            code = run.main(["--items", str(items), "--split", "dev", "--backend", "vllm",
                             "--out-dir", str(Path(tmp) / "o"), "--limit", "2"])
            self.assertEqual(code, 1)


# ---------------------------------------------------------------------------
class ResumabilityTests(unittest.TestCase):
    def _run(self, items, out, extra=()):
        argv = ["--items", str(items), "--split", "dev", "--backend", "mock", "--out-dir", str(out),
                "--seeds", "11,23", "--limit", "4", *extra]
        self.assertEqual(run.main(argv), 0)
        manifest = common.load_json(out / "run_manifest_dev.json")
        return manifest["runs"][-1]["stats"]

    def test_second_run_makes_no_calls_and_truncation_is_repaired(self):
        with tempfile.TemporaryDirectory() as tmp:
            items, out = _make_dataset(Path(tmp)), Path(tmp) / "out"
            first = self._run(items, out)["closed_book"]
            self.assertGreater(first["calls"], 0)
            path = out / "signals_dev.jsonl"
            records = common.read_jsonl(path)
            self.assertEqual(len(records), first["records"])
            self.assertGreater(first["records"], first["calls"])  # dedup saved calls
            self.assertEqual(self._run(items, out)["closed_book"]["calls"], 0)

            # Lose 3 records and leave a half-written line, as a crash would.
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(lines[:-3]) + "\n" + lines[-3][:40], encoding="utf-8")
            again = self._run(items, out)["closed_book"]
            self.assertLessEqual(again["calls"], 3)
            restored = common.read_jsonl(path)
            keys = [(r["item_id"], r["draw_seed"], r["prefix_id"]) for r in restored]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertEqual(len(keys), first["records"])

    def test_missing_dedup_copy_is_rebuilt_without_a_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            items, out = _make_dataset(Path(tmp)), Path(tmp) / "out"
            self._run(items, out)
            path = out / "signals_dev.jsonl"
            records = common.read_jsonl(path)
            copy_index = next(i for i, r in enumerate(records) if r["reused_generation"])
            kept = [r for i, r in enumerate(records) if i != copy_index]
            path.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
            stats_ = self._run(items, out)["closed_book"]
            self.assertEqual(stats_["calls"], 0)
            self.assertEqual(stats_["reused"], 1)
            self.assertEqual(len(common.read_jsonl(path)), len(records))

    def test_failed_calls_are_logged_and_retried_next_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            items_path = _make_dataset(Path(tmp), traces=False)
            cfg = common.load_config()
            item_list = common.load_items(items_path)[:2]
            units = run.build_units(item_list, {}, cfg, [11], "m", use_asr=False)
            llm_cfg = {"model": "m", "logprobs": {"top_k": 5}}
            out = Path(tmp) / "s.jsonl"

            def go(backend):
                done, by_call = run.load_existing(out)
                with common.JsonlAppender(out) as w, common.JsonlAppender(Path(tmp) / "e.jsonl") as e:
                    return asyncio.run(run.execute(units, backend, w, cfg, model="m", model_key="k",
                                                   split="dev", done_keys=done, existing_by_call=by_call,
                                                   concurrency=4, error_writer=e))

            first = go(MockAsyncBackend(llm_cfg, 5, fail_first_n_calls=3))
            self.assertEqual(first["failed"], 3)
            self.assertEqual(len(common.read_jsonl(Path(tmp) / "e.jsonl")), 3)
            second = go(MockAsyncBackend(llm_cfg, 5))
            self.assertEqual(second["calls"], 3)
            self.assertEqual(second["failed"], 0)


# ---------------------------------------------------------------------------
class LabelTests(unittest.TestCase):
    def test_alias_match_is_normalized_and_token_bounded(self):
        self.assertTrue(label.alias_match("It was in 1999.", ["1999"]))
        self.assertFalse(label.alias_match("It was 19990", ["1999"]))
        self.assertTrue(label.alias_match("The Beatles, of course", ["Beatles"]))
        self.assertTrue(label.alias_match("Rs. 150 late fee", ["Rs 150"]))
        self.assertFalse(label.alias_match("anything", ["", "  "]))

    def test_p_wrong_and_threshold(self):
        self.assertEqual(label.p_wrong([True, False, None, False])["p_wrong"], 2 / 3)
        self.assertTrue(label.empirical_need(0.5, 0.5))
        self.assertFalse(label.empirical_need(0.4, 0.5))
        self.assertIsNone(label.empirical_need(None, 0.5))

    def test_build_labels_with_gain(self):
        items = [{"id": "a", "dataset": "d1_open", "retrieval_needed_authored": True}]
        rows = [{"item_id": "a", "draw_seed": s, "condition": c, "alias_correct": ok, "judges": {}}
                for s, c, ok in [(1, "closed_book", False), (2, "closed_book", False),
                                 (3, "closed_book", True), (1, "retrieval", True),
                                 (2, "retrieval", True), (3, "retrieval", False)]]
        out = label.build_labels(items, rows, {}, lambda _ds: "alias", 0.5, [])["a"]
        self.assertAlmostEqual(out["closed_book"]["p_wrong"], 2 / 3)
        self.assertTrue(out["empirical_need"])
        self.assertAlmostEqual(out["gain"], 2 / 3 - 1 / 3)


class AgreementTests(unittest.TestCase):
    def test_kappa_known_values(self):
        stats_ = score_judges.agreement([1, 1, 0, 0], [1, 0, 0, 0])
        self.assertAlmostEqual(stats_["percent_agreement"], 0.75)
        self.assertAlmostEqual(stats_["cohen_kappa"], 0.5)
        self.assertEqual(stats_["a_yes_b_no"], 1)
        self.assertIsNone(score_judges.agreement([1, 1], [1, 1])["cohen_kappa"])
        self.assertEqual(score_judges.agreement([1, None], [None, 0])["n"], 0)


class JudgeTests(unittest.TestCase):
    CFG = judges.load_judges_config()
    ENV = {
        "VOICE_RAG_JUDGE_DEPLOYMENT": "gpt6-sol", "VOICE_RAG_JUDGE_API_VERSION": "2026-09-01",
        "VOICE_RAG_JUDGE_MODEL": "gpt-6-sol-2026-08", "AZURE_OPENAI_ENDPOINT": SENTINEL_URL,
        "AZURE_OPENAI_API_KEY": SENTINEL_KEY, "AZURE_OPENAI_DEPLOYMENT": "gpt-4o",
        "AZURE_OPENAI_API_VERSION": "2025-01-01-preview",
    }

    def test_resolution_and_missing_env(self):
        primary = judges.resolve_spec("primary", self.CFG, self.ENV)
        self.assertEqual(primary.deployment, "gpt6-sol")
        self.assertIsNone(primary.request["temperature"])
        self.assertEqual(primary.request["token_param"], "max_completion_tokens")
        secondary = judges.resolve_spec("secondary", self.CFG, self.ENV)
        self.assertEqual((secondary.deployment, secondary.model_id), ("gpt-4o", "gpt-4o"))
        self.assertEqual(secondary.request["temperature"], 0)
        with self.assertRaises(judges.JudgeConfigError) as ctx:
            judges.resolve_spec("primary", self.CFG, {"AZURE_OPENAI_API_KEY": SENTINEL_KEY})
        self.assertIn("VOICE_RAG_JUDGE_DEPLOYMENT", str(ctx.exception))
        self.assertNotIn(SENTINEL_KEY, str(ctx.exception))
        env = dict(self.ENV, VOICE_RAG_JUDGE_TEMPERATURE="0", VOICE_RAG_JUDGE_MAX_TOKENS="999")
        over = judges.resolve_spec("primary", self.CFG, env)
        self.assertEqual((over.request["temperature"], over.request["max_output_tokens"]), (0, 999))

    def test_cache_key_includes_judge_identity(self):
        a = judges.AzureChatJudge(judges.resolve_spec("primary", self.CFG, self.ENV),
                                  transport=lambda m, p: ("VERDICT: YES", "x"))
        b = judges.AzureChatJudge(judges.resolve_spec(
            "primary", self.CFG, dict(self.ENV, VOICE_RAG_JUDGE_MODEL="gpt-6-sol-2026-10")),
            transport=lambda m, p: ("VERDICT: YES", "x"))
        kw = dict(item_id="i", question="q", reference="r", answer="a", key_facts=[], aliases=[],
                  context="general")
        self.assertNotEqual(judges.cache_key(a, **kw), judges.cache_key(b, **kw))
        self.assertNotEqual(a.spec.slug(), b.spec.slug())

    def test_reasoning_model_param_adaptation_is_recorded(self):
        sent = []

        def transport(messages, params):
            sent.append(dict(params))
            if "temperature" in params:
                raise RuntimeError("Unsupported value: 'temperature' does not support 0 with this model")
            if "max_tokens" in params:
                raise RuntimeError("Unsupported parameter: 'max_tokens' is not supported with this "
                                   "model. Use 'max_completion_tokens' instead.")
            return "CONFLICT: none\nVERDICT: YES", "gpt-6-sol-2026-08"

        env = dict(self.ENV, VOICE_RAG_JUDGE_TEMPERATURE="0", VOICE_RAG_JUDGE_TOKEN_PARAM="max_tokens")
        judge = judges.AzureChatJudge(judges.resolve_spec("primary", self.CFG, env), transport=transport)
        with tempfile.TemporaryDirectory() as tmp:
            verdict = judges.grade(judge, item_id="i", question="q", reference="r", answer="  a  ",
                                   cache_dir=Path(tmp))
            self.assertTrue(verdict["answer_correct"])
            self.assertEqual(verdict["sent_params"], {"max_completion_tokens": 4000})
            self.assertEqual(verdict["response_model"], "gpt-6-sol-2026-08")
            self.assertEqual(verdict["judge"]["model_id"], "gpt-6-sol-2026-08")
            self.assertEqual(len(judge.adaptations), 2)
            on_disk = "".join(p.read_text(encoding="utf-8") for p in Path(tmp).rglob("*.json"))
            self.assertNotIn(SENTINEL_KEY, on_disk)
            self.assertNotIn("sentinel-endpoint", on_disk)
            # second call is a cache hit: no new request
            n = len(sent)
            self.assertTrue(judges.grade(judge, item_id="i", question="q", reference="r", answer="a",
                                         cache_dir=Path(tmp))["cache_hit"])
            self.assertEqual(len(sent), n)

    def test_errors_are_redacted(self):
        def transport(messages, params):
            raise RuntimeError(f"401 for {SENTINEL_URL} with key {SENTINEL_KEY}")

        judge = judges.AzureChatJudge(judges.resolve_spec("secondary", self.CFG, self.ENV),
                                      transport=transport)
        with self.assertRaises(judges.JudgeCallError) as ctx:
            judge.complete("s", "u")
        self.assertNotIn(SENTINEL_KEY, str(ctx.exception))
        self.assertNotIn(SENTINEL_URL, str(ctx.exception))

    def test_v2_prompt_derives_from_v4(self):
        text = judges.user_prompt("Q?", "Ref", "Ans", key_facts=["fact one"], aliases=["alt"])
        self.assertIn("naming the company or organisation", text)
        self.assertNotIn("hotel", text)
        self.assertIn("KEY FACTS", text)
        self.assertTrue(text.rstrip().endswith("Candidate answer:\nAns"))
        self.assertIn("an open-domain", judges.system_prompt("open-domain"))


# ---------------------------------------------------------------------------
class StatsTests(unittest.TestCase):
    def test_auc_matches_v1(self):
        rng = random.Random(3)
        y = [rng.random() < 0.4 for _ in range(200)]
        s = [round(rng.random(), 1) for _ in range(200)]  # many ties
        self.assertAlmostEqual(stats.auc(y, s), roc_auc([int(v) for v in y], s))

    def test_isotonic_and_ece(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        y = np.array([0, 1, 0, 1, 1, 1.0])
        model = stats.fit_isotonic(x, y)
        pred = stats.predict_isotonic(model, x)
        self.assertTrue(np.all(np.diff(pred) >= 0))
        self.assertAlmostEqual(stats.ece([0, 1], [0.0, 1.0]), 0.0)
        self.assertAlmostEqual(stats.ece([1, 1], [0.5, 0.5]), 0.5)

    def test_logistic_safe_on_separable_data(self):
        slope, intercept, info = stats.fit_logistic_safe([0, 1, 2, 3], [0, 0, 1, 1])
        self.assertGreater(slope, 0)
        self.assertIn("separable", info["method"])

    def test_artifact_loads_in_controller(self):
        from controller.baselines._objective import load_voi_calibration

        rng = np.random.default_rng(0)
        items = [{"id": f"i{n}", "retrieval_needed_authored": bool(n % 2)} for n in range(40)]
        records = []
        for item in items:
            for seed in (11, 23):
                value = float(rng.normal(0.6 if item["retrieval_needed_authored"] else 0.3, 0.15))
                records.append({"item_id": item["id"], "draw_seed": seed,
                                "prefix": {"source": "text", "word_frac": 1.0, "kind": "full"},
                                "_flat": {"mean_entropy_nats": value}})
        lm = {i["id"]: i["retrieval_needed_authored"] for i in items}
        cal = analyze.calibration_block(records, "mean_entropy_nats", lm, {},
                                        {"cv_folds": 5, "ece_bins": 10})
        self.assertGreater(cal["logistic"]["slope"], 0)
        self.assertIsNotNone(cal["cross_validation"]["logistic"]["ece"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cal.json"
            analyze.write_artifact(path, cal, {"backend": "vllm", "model": "m", "model_key": "k"},
                                   "authored", {"signals_sha256": "0"}, "d", "dev")
            loaded = load_voi_calibration({"voi": {"calibration": {
                "from_artifact": str(path), "signal": "mean_entropy_nats"}}})
            self.assertTrue(loaded.calibrated)
            self.assertAlmostEqual(loaded.slope, cal["logistic"]["slope"])


# ---------------------------------------------------------------------------
class EndToEndMockSmoke(unittest.TestCase):
    """v1 dataset -> v2 schema -> run (mock, text + plumbing ASR prefixes,
    lexical retrieval) -> label (mock judges) -> analyze -> judge validation."""

    def test_pipeline(self):
        env = {"VOICE_RAG_LLM_BASE_URL": SENTINEL_URL, "VLLM_API_KEY": SENTINEL_KEY,
               "MPLBACKEND": "Agg"}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, env):
            tmp = Path(tmp)
            items = _make_dataset(tmp)
            root = tmp / "results"
            out = root / "mistral7b_v03"
            self.assertEqual(run.main([
                "--items", str(items), "--split", "dev", "--backend", "mock", "--model",
                "mistral7b_v03", "--out-dir", str(out), "--seeds", "11,23", "--limit", "12",
                "--with-retrieval", "--retriever", "lexical"]), 0)
            records = common.read_jsonl(out / "signals_dev.jsonl")
            sources = {(r["prefix"]["kind"], r["prefix"]["source"]) for r in records}
            self.assertEqual(sources, {("full", "text"), ("partial", "text"), ("partial", "asr"),
                                       ("full", "asr")})
            r0 = records[0]
            for key in ("item_id", "model", "draw_seed", "prefix", "signals", "closed_book_answer",
                        "ttft_ms", "gen_ms", "n_tokens"):
                self.assertIn(key, r0)
            self.assertEqual(set(r0["signals"]["first_k_mean_entropy_nats"]), {"4", "8", "16"})
            self.assertIn("mean_entropy_nats", r0["signals"])
            self.assertTrue(r0["timing_synthetic"])
            retrieval = common.read_jsonl(out / "retrieval_dev.jsonl")
            self.assertEqual(len(retrieval), 12 * 2)
            self.assertTrue(any(r["retrieval"]["gold_hit"] for r in retrieval))

            self.assertEqual(label.main([
                "--items", str(items), "--split", "dev", "--model", "mistral7b_v03",
                "--results-dir", str(out), "--judge-backend", "mock",
                "--cache-dir", str(tmp / "jc")]), 0)
            labels = common.load_json(out / "labels_dev.json")
            self.assertEqual(len(labels["items"]), 12)
            some = next(iter(labels["items"].values()))
            self.assertIn("p_wrong", some["closed_book"])
            self.assertIsNotNone(some["gain"])
            self.assertIn("judge:primary vs judge:secondary", labels["agreement"]["closed_book"])
            self.assertEqual(labels["meta"]["judges"]["primary"]["provider"], "mock")

            self.assertEqual(analyze.main([
                "--items", str(items), "--split", "dev", "--results-root", str(root),
                "--bootstrap", "50", "--controller-artifact"]), 0)
            analysis = common.load_json(root / "analysis_dev" / "analysis_dev.json")
            model = analysis["models"]["mistral7b_v03"]
            self.assertIn("empirical", model["auc_full"]["mean_entropy_nats"])
            self.assertEqual([p["word_frac"] for p in model["prefix_curve_text"]["mean_entropy_nats"]],
                             [0.25, 0.5, 0.75, 1.0])
            self.assertTrue(model["asr_curves"]["mean_entropy_nats"])
            self.assertEqual([r["k"] for r in model["first_k"]], [4, 8, 16, "all"])
            self.assertTrue(analysis["figures"])
            for paths in analysis["artifacts"].values():  # never into controller/ for mock
                self.assertTrue(all("controller/baselines" not in p for p in paths))

            blob = "".join(p.read_text(encoding="utf-8", errors="ignore")
                           for p in tmp.rglob("*") if p.suffix in (".json", ".jsonl"))
            self.assertNotIn(SENTINEL_URL, blob)
            self.assertNotIn(SENTINEL_KEY, blob)

            cases = tmp / "human.csv"
            cases.write_text(
                "case_id,dataset,item_id,question,reference,answer,label_a,label_b\n"
                "c1,d1_open,i1,Capital of France?,Paris,It is Paris.,1,1\n"
                "c2,d1_open,i2,Capital of Peru?,Lima,Maybe Cusco.,0,0\n"
                "c3,d2_support,i3,Late fee?,Rs 150,Rs 150 after 7 days,1,0\n", encoding="utf-8")
            report = score_judges.validate_v2(cases, ("primary", "secondary"), "mock", tmp / "vc", 2)
            self.assertEqual(report["n_with_reference_label"], 2)
            self.assertEqual(report["vs_reference"]["primary"]["percent_agreement"], 1.0)
            self.assertIn("judge_vs_judge", report)


if __name__ == "__main__":
    unittest.main()
