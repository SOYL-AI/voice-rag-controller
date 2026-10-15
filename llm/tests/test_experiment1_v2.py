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


class SelfAssessmentTests(unittest.TestCase):
    ITEM = {"id": "d2_support-1", "dataset": "d2_support", "org": "Sarovan Telecom",
            "domain": "telecom", "question": "q",
            "history": [{"role": "user", "content": "Hi, roaming?"},
                        {"role": "assistant", "content": "Two packs."}]}

    def test_prompt_is_the_frozen_probe_wording(self):
        from llm.experiment1_v2 import self_assess

        sa = common.load_config()["self_assessment"]
        messages = self_assess.build_messages(self.ITEM, "And the seven-day", sa)
        # Reconstructed exactly as the exploratory probe built it.
        it = self.ITEM
        hist = "".join(f"{t['role'].capitalize()}: {t['content']}\n" for t in it["history"])
        probe_system = (f"You decide whether a customer-support voice agent for {it['org']}, a "
                        f"{it['domain']} company, must look up the company's own documents before "
                        "answering a caller.")
        probe_user = (f"{hist}Caller (may still be mid-sentence): \"And the seven-day\"\n"
                      f"Can this be answered correctly from general knowledge alone, without "
                      f"{it['org']}'s own policies, prices, plans or records? Answer with one word: "
                      "Yes or No.")
        self.assertEqual(messages, [{"role": "system", "content": probe_system},
                                    {"role": "user", "content": probe_user}])
        d1 = self_assess.build_messages({"id": "x", "dataset": "d1_open", "question": "q"}, "Who", sa)
        self.assertIn("from your own knowledge, without looking anything up?", d1[1]["content"])
        self.assertEqual(sa["request"], {"max_tokens": 1, "temperature": 0, "top_logprobs": 20})

    def test_p_no_math(self):
        from llm.experiment1_v2 import self_assess

        tops = [AlternativeLogprob("No", math.log(0.6)), AlternativeLogprob(" Yes", math.log(0.2)),
                AlternativeLogprob("Not", math.log(0.1)), AlternativeLogprob("Maybe", math.log(0.05))]
        out = self_assess.p_no_from_top(tops)
        self.assertAlmostEqual(out["p_no"], 0.7 / 0.9)
        self.assertAlmostEqual(out["yes_mass"], 0.2)
        self.assertIsNone(self_assess.p_no_from_top([AlternativeLogprob("I", 0.0)])["p_no"])


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

    def test_p_wrong_and_three_of_five_rule(self):
        self.assertEqual(label.p_wrong([True, False, None, False])["p_wrong"], 2 / 3)
        rule = {"min_wrong": 3, "of_draws": 5}
        self.assertTrue(label.empirical_need(3, 5, 5, rule))
        self.assertFalse(label.empirical_need(2, 5, 5, rule))
        self.assertTrue(label.empirical_need(3, 4, 5, rule))     # decided despite 1 unscored
        self.assertFalse(label.empirical_need(0, 4, 5, rule))    # cannot reach 3
        self.assertIsNone(label.empirical_need(2, 4, 5, rule))   # the unscored draw decides
        self.assertTrue(label.empirical_need(2, 2, 2, rule))     # 2-seed pilot: ceil(2*3/5) = 2
        self.assertFalse(label.empirical_need(1, 2, 2, rule))

    def test_build_labels_with_gain(self):
        items = [{"id": "a", "dataset": "d1_open", "retrieval_needed_authored": True}]
        rows = [{"item_id": "a", "draw_seed": s, "condition": c, "alias_correct": ok, "judges": {}}
                for s, c, ok in [(1, "closed_book", False), (2, "closed_book", False),
                                 (3, "closed_book", True), (1, "retrieval", True),
                                 (2, "retrieval", True), (3, "retrieval", False)]]
        out = label.build_labels(items, rows, {}, lambda _ds: "alias",
                                 {"min_wrong": 3, "of_draws": 5}, [])["a"]
        self.assertAlmostEqual(out["closed_book"]["p_wrong"], 2 / 3)
        self.assertTrue(out["empirical_need"])  # 3 draws: ceil(3*3/5) = 2 wrong needed
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
        "VOICE_RAG_JUDGE_DEPLOYMENT": "gpt-6.1-sol", "VOICE_RAG_JUDGE_API_VERSION": "2025-04-01-preview",
        "VOICE_RAG_JUDGE_MODEL": "gpt-6.1-sol-2026-09-29", "AZURE_OPENAI_ENDPOINT": SENTINEL_URL,
        "AZURE_OPENAI_API_KEY": SENTINEL_KEY, "AZURE_OPENAI_DEPLOYMENT": "gpt-4o",
        "AZURE_OPENAI_API_VERSION": "2025-01-01-preview",
    }

    def test_resolution_and_missing_env(self):
        primary = judges.resolve_spec("primary", self.CFG, self.ENV)
        self.assertEqual(primary.deployment, "gpt-6.1-sol")
        self.assertIsNone(primary.request["temperature"])
        self.assertEqual(primary.request["token_param"], "max_completion_tokens")
        self.assertEqual((primary.request["reasoning_effort"], primary.request["samples"]), ("medium", 3))
        params = judges.AzureChatJudge(primary, transport=lambda m, p: None)._params(seed=1)
        self.assertNotIn("temperature", params)
        self.assertNotIn("max_tokens", params)
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
            return "CONFLICT: none\nVERDICT: YES", "gpt-6.1-sol-2026-09-29"

        env = dict(self.ENV, VOICE_RAG_JUDGE_TEMPERATURE="0", VOICE_RAG_JUDGE_TOKEN_PARAM="max_tokens")
        judge = judges.AzureChatJudge(judges.resolve_spec("primary", self.CFG, env), transport=transport)
        with tempfile.TemporaryDirectory() as tmp:
            verdict = judges.grade(judge, item_id="i", question="q", reference="r", answer="  a  ",
                                   cache_dir=Path(tmp))
            self.assertTrue(verdict["answer_correct"])
            self.assertEqual(verdict["sent_params"], {"max_completion_tokens": 2000,
                                                      "reasoning_effort": "medium",
                                                      "seed": 20261015})
            self.assertEqual([s["sent_params"]["seed"] for s in verdict["samples"]],
                             [20261015, 20261016, 20261017])
            self.assertEqual(verdict["response_model"], "gpt-6.1-sol-2026-09-29")
            self.assertEqual(verdict["judge"]["model_id"], "gpt-6.1-sol-2026-09-29")
            self.assertEqual(len(judge.adaptations), 2)
            on_disk = "".join(p.read_text(encoding="utf-8") for p in Path(tmp).rglob("*.json"))
            self.assertNotIn(SENTINEL_KEY, on_disk)
            self.assertNotIn("sentinel-endpoint", on_disk)
            # second call is a cache hit: no new request
            n = len(sent)
            self.assertTrue(judges.grade(judge, item_id="i", question="q", reference="r", answer="a",
                                         cache_dir=Path(tmp))["cache_hit"])
            self.assertEqual(len(sent), n)

    def test_k_sample_majority_records_every_sample_and_usage(self):
        replies = iter(["VERDICT: YES", "VERDICT: NO", "reasoning...\nVERDICT: YES"])
        usage = {"completion_tokens": 50, "completion_tokens_details": {"reasoning_tokens": 40}}

        def transport(messages, params):
            return next(replies), "gpt-6.1-sol-2026-09-29", usage

        judge = judges.AzureChatJudge(judges.resolve_spec("primary", self.CFG, self.ENV), transport=transport)
        with tempfile.TemporaryDirectory() as tmp:
            verdict = judges.grade(judge, item_id="i", question="q", reference="r", answer="a",
                                   cache_dir=Path(tmp))
            self.assertTrue(verdict["answer_correct"])
            self.assertEqual(verdict["n_samples"], 3)
            self.assertAlmostEqual(verdict["self_consistency"], 2 / 3)
            self.assertFalse(verdict["unanimous"])
            self.assertEqual([s["answer_correct"] for s in verdict["samples"]], [True, False, True])
            self.assertEqual(verdict["samples"][0]["usage"]["completion_tokens_details"]["reasoning_tokens"], 40)
            self.assertEqual(len(list(Path(tmp).rglob("*.json"))), 3)  # one cache file per sample
        rows = [{"judges": {"primary": {**verdict, "samples": verdict["samples"]}}}]
        sc = label.self_consistency(rows, "primary")
        self.assertEqual(sc["k"], 3)
        self.assertEqual(sc["usage"]["reasoning_tokens_total"], 120)
        self.assertEqual(sc["pairwise_samples"]["0-2"]["percent_agreement"], 1.0)

    def test_majority_ties_are_unresolved(self):
        self.assertEqual(judges.majority([True, None, False]), (False, False, 0.5))
        self.assertEqual(judges.majority([None, None]), (False, False, None))
        self.assertEqual(judges.majority([False, False, True])[:2], (False, True))
        self.assertEqual(judges.majority([True, None, None], min_parsed=2)[:2], (False, False))

    def test_parse_failures_draw_extra_samples(self):
        replies = iter(["garbled", "no verdict here", "VERDICT: YES", "VERDICT: YES", "VERDICT: NO"])
        seeds = []

        def transport(messages, params):
            seeds.append(params["seed"])
            return {"text": next(replies), "model": "gpt-6.1-sol-2026-09-29", "usage": None, "response_id": f"r{len(seeds)}",
                    "finish_reason": "stop", "request_body": {"model": "x", **params}}

        judge = judges.AzureChatJudge(judges.resolve_spec("primary", self.CFG, self.ENV), transport=transport)
        verdict = judges.grade(judge, item_id="i", question="q", reference="r", answer="a")
        # 3 planned (1 parsed) -> extra draws until 2 parse and agree: stops after the 4th.
        self.assertEqual(seeds, [20261015, 20261016, 20261017, 20261018])
        self.assertTrue(verdict["answer_correct"])
        self.assertTrue(verdict["parsed_ok"])
        self.assertEqual(verdict["n_extra_samples"], 1)
        self.assertFalse(verdict["unanimous"])
        self.assertEqual(verdict["samples"][3]["response_id"], "r4")
        self.assertTrue(verdict["samples"][3]["request_body_captured"])
        self.assertEqual(len(verdict["judge"]["prompt_sha256"]), 64)

    def test_validation_gate_on_mock_judges(self):
        rows = ["case_id,dataset,item_id,question,reference,answer,label_a,label_b,path"]
        for n in range(90):
            right = n % 2 == 0
            answer = "It is Paris." if right else "Maybe Lyon."
            rows.append(f"c{n},d1_open,i{n},Capital of France number {n}?,Paris,{answer},"
                        f"{int(right)},{int(right)},{'retrieve' if n % 3 else 'skip'}")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "h.csv"
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            report = score_judges.validate_v2(path, ("primary", "secondary"), "mock", Path(tmp) / "c", 2)
        gate = report["gate"]
        self.assertEqual(gate["n_prompt_dev"] + gate["n_holdout_labelled"], 90)
        self.assertGreater(gate["n_holdout_labelled"], 45)
        self.assertTrue(gate["human_kappa_ok"])
        primary = gate["judges"]["primary"]
        self.assertIsNotNone(primary["fleiss_kappa"])
        self.assertIsNotNone(primary["path_error_gap"])
        self.assertIsNone(gate["judges"]["secondary"]["checks"]["fleiss"])  # single sample
        self.assertEqual(set(primary["checks"]),
                         {"kappa", "kappa_lower95", "accuracy", "path_error_gap", "unanimous", "fleiss"})

    def test_fleiss_kappa(self):
        perfect = score_judges.fleiss_kappa([[True] * 3, [False] * 3, [True] * 3])
        self.assertAlmostEqual(perfect["fleiss_kappa"], 1.0)
        mixed = score_judges.fleiss_kappa([[True, True, False], [False, False, True]])
        self.assertLess(mixed["fleiss_kappa"], 0.0)
        self.assertEqual(score_judges.fleiss_kappa([[True, None, True]])["n"], 0)

    def test_seed_and_effort_change_the_cache_key(self):
        kw = dict(item_id="i", question="q", reference="r", answer="a", key_facts=[], aliases=[],
                  context="general")
        base = judges.AzureChatJudge(judges.resolve_spec("primary", self.CFG, self.ENV),
                                     transport=lambda m, p: ("", ""))
        for override in ({"VOICE_RAG_JUDGE_SEED": "7"}, {"VOICE_RAG_JUDGE_REASONING_EFFORT": "high"},
                         {"VOICE_RAG_JUDGE_SAMPLES": "5"}):
            other = judges.AzureChatJudge(judges.resolve_spec("primary", self.CFG, dict(self.ENV, **override)),
                                          transport=lambda m, p: ("", ""))
            self.assertNotEqual(judges.cache_key(base, **kw), judges.cache_key(other, **kw))

    def test_secondary_keeps_v4_settings(self):
        spec = judges.resolve_spec("secondary", self.CFG, self.ENV)
        self.assertEqual(judges.AzureChatJudge(spec, transport=lambda m, p: None)._params(),
                         {"max_tokens": 300, "temperature": 0})
        self.assertEqual(judges.sample_seeds(spec.request), [None])

    # -- deployment pools --------------------------------------------------
    POOL_ENV = dict(ENV, VOICE_RAG_JUDGE_DEPLOYMENTS="gpt-6.1-sol,gpt-6.1-sol-dz")
    MODEL = "gpt-6.1-sol-2026-09-29"
    KW = dict(item_id="i", question="q", reference="r", answer="a", key_facts=["f"], aliases=[],
              context="customer-support", dataset="d2_support", core_facts=["f"])

    def test_pool_resolution_and_single_deployment_compat(self):
        pool = judges.resolve_spec("primary", self.CFG, self.POOL_ENV)
        self.assertEqual(pool.pool, ("gpt-6.1-sol", "gpt-6.1-sol-dz"))
        single = judges.resolve_spec("primary", self.CFG, self.ENV)  # VOICE_RAG_JUDGE_DEPLOYMENT only
        self.assertEqual(single.pool, ("gpt-6.1-sol",))
        self.assertNotIn("deployment", pool.identity())

    def test_cache_key_is_invariant_across_deployments(self):
        def judge(env):
            return judges.AzureChatJudge(judges.resolve_spec("primary", self.CFG, env),
                                         transport=lambda m, p: None)

        keys = {judges.cache_key(judge(env), **self.KW) for env in (
            self.ENV, self.POOL_ENV, dict(self.ENV, VOICE_RAG_JUDGE_DEPLOYMENT="gpt-6.1-sol-dz"),
            dict(self.ENV, VOICE_RAG_JUDGE_DEPLOYMENTS="gpt-6.1-sol-dz,gpt-6.1-sol"))}
        self.assertEqual(len(keys), 1)
        self.assertEqual(judge(self.ENV).spec.slug(), judge(self.POOL_ENV).spec.slug())
        # ... while the model version still splits it
        other = judge(dict(self.POOL_ENV, VOICE_RAG_JUDGE_MODEL="gpt-6.1-sol-2026-12-01"))
        self.assertNotIn(judges.cache_key(other, **self.KW), keys)

    def test_pre_pool_cache_entries_still_hit(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            # A verdict cached before pools, under the deployment-named identity.
            old = judges.AzureChatJudge(judges.resolve_spec("primary", self.CFG, self.ENV),
                                        transport=lambda m, p: None)
            legacy_key, legacy_slug = judges.legacy_cache_keys(old, **dict(self.KW, answer="a"))[0]
            for index, seed in enumerate((20261015, 20261016, 20261017)):
                sk = common.sha256_text(legacy_key, index, seed)
                path = Path(tmp) / legacy_slug / f"{sk[:40]}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"sample_index": index, "seed": seed, "answer_correct": True,
                                            "parsed_ok": True, "judge_raw": "VERDICT: YES"}),
                                encoding="utf-8")
            pooled = judges.AzureChatJudge(judges.resolve_spec("primary", self.CFG, self.POOL_ENV),
                                           transport=lambda m, p: calls.append(1))
            verdict = judges.grade(pooled, cache_dir=Path(tmp), **self.KW)
            self.assertEqual(calls, [])  # served from the legacy cache, no API call
            self.assertTrue(verdict["cache_hit"] and verdict["answer_correct"])

    def test_model_version_mismatch_is_refused(self):
        spec = judges.resolve_spec("primary", self.CFG, self.POOL_ENV)
        transports = {"gpt-6.1-sol": lambda m, p: ("VERDICT: YES", self.MODEL),
                      "gpt-6.1-sol-dz": lambda m, p: ("VERDICT: YES", "gpt-6.1-sol-2026-12-01")}
        judge = judges.AzureChatJudge(spec, transport=transports)
        with self.assertRaises(judges.JudgeModelMismatch):
            for _ in range(4):  # rotation reaches the mismatched deployment
                judge.complete("s", "u", seed=1)
        self.assertTrue(judges.model_matches("gpt-4o", "gpt-4o-2024-11-20"))  # unversioned id + version
        self.assertTrue(judges.model_matches(self.MODEL, self.MODEL))
        self.assertFalse(judges.model_matches(self.MODEL, "gpt-6.1-sol-2026-12-01"))

    def test_pool_fails_over_on_rate_limit(self):
        class RateLimited(Exception):
            status_code = 429

        now = [0.0]
        sleeps = []
        hits = {"gpt-6.1-sol": 0, "gpt-6.1-sol-dz": 0}

        def limited(messages, params):
            hits["gpt-6.1-sol"] += 1
            raise RateLimited("429 Too Many Requests")

        def ok(messages, params):
            hits["gpt-6.1-sol-dz"] += 1
            return "CONFLICT: none\nVERDICT: YES", self.MODEL

        spec = judges.resolve_spec("primary", self.CFG, self.POOL_ENV)
        judge = judges.AzureChatJudge(spec, transport={"gpt-6.1-sol": limited, "gpt-6.1-sol-dz": ok},
                                      sleep=lambda s: (sleeps.append(s), now.__setitem__(0, now[0] + s)),
                                      clock=lambda: now[0])
        replies = [judge.complete("s", "u", seed=i) for i in range(5)]
        self.assertTrue(all(r["deployment"] == "gpt-6.1-sol-dz" for r in replies))
        self.assertEqual(hits["gpt-6.1-sol-dz"], 5)
        self.assertLessEqual(hits["gpt-6.1-sol"], 2)  # cooled down after its 429s, traffic moved
        self.assertEqual(sleeps, [])                  # never had to wait: the other deployment was free
        self.assertGreaterEqual(judge.pool_stats()["gpt-6.1-sol"]["rate_limited_streak"], 1)

        # Both deployments rate-limited: waits for the earliest cooldown, then succeeds.
        flaky = {"n": 0}

        def flaky_ok(messages, params):
            flaky["n"] += 1
            if flaky["n"] <= 2:
                raise RateLimited("429")
            return "VERDICT: NO", self.MODEL

        judge2 = judges.AzureChatJudge(spec, transport={"gpt-6.1-sol": flaky_ok, "gpt-6.1-sol-dz": flaky_ok},
                                       sleep=lambda s: (sleeps.append(s), now.__setitem__(0, now[0] + s)),
                                       clock=lambda: now[0])
        reply = judge2.complete("s", "u", seed=1)
        self.assertEqual(reply["rate_limited_retries"], 2)
        self.assertTrue(sleeps)  # had to wait once both were cooling down

    # -- spend cap -----------------------------------------------------------
    def test_budget_config_baseline_and_cap_override(self):
        from llm.experiment1_v2 import spend

        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.json"
            budget = spend.budget_from_config("primary", self.CFG, env={}, ledger_path=ledger)
            status = budget.status()
            self.assertAlmostEqual(status["spent_usd"], 88.85, places=2)  # the Azure-metrics baseline
            self.assertAlmostEqual(status["spent_inr"], 88.85 * 88, delta=1)
            self.assertEqual(status["cap_inr"], 20000)
            capped = spend.budget_from_config("primary", self.CFG, env={"VOICE_RAG_JUDGE_CAP_INR": "9000"},
                                              ledger_path=ledger)
            self.assertEqual(capped.cap_inr, 9000)
            gpt4o = spend.budget_from_config("secondary", self.CFG, env={}, ledger_path=ledger)
            self.assertIsNone(gpt4o.cap_inr)
            self.assertEqual(gpt4o.price, {"input": 2.5, "output": 10.0})
            self.assertEqual(spend.main(["--ledger", str(ledger)]), 0)

    def test_cap_trips_before_the_call_and_grading_resumes_from_cache(self):
        from llm.experiment1_v2 import spend

        calls = []
        usage = {"prompt_tokens": 400, "completion_tokens": 100}  # $0.0018 per call

        def transport(messages, params):
            calls.append(params["seed"])
            return {"text": "VERDICT: YES", "model": self.MODEL, "usage": usage}

        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.json"
            # Baseline 0; INR 88/USD. Estimate per call ~ (chars/3 + 50) in + 2000 out ~ $0.0216 = INR 1.9.
            budget = spend.SpendBudget("gpt-6.1-sol", {"input": 2.0, "output": 10.0}, 88.0, cap_inr=5.0,
                                       ledger_path=ledger)
            judge = judges.AzureChatJudge(judges.resolve_spec("primary", self.CFG, self.ENV),
                                          transport=transport, budget=budget)
            cache = Path(tmp) / "cache"
            with self.assertRaises(spend.JudgeBudgetExceeded) as ctx:
                for n in range(10):
                    judges.grade(judge, cache_dir=cache, **dict(self.KW, item_id=f"i{n}", answer=f"a{n}"))
            self.assertIn("No call was made", str(ctx.exception))
            made = len(calls)
            self.assertGreater(made, 0)
            status = budget.status()
            self.assertLessEqual(status["spent_inr"], 5.0)  # never over the cap
            self.assertEqual(status["calls"], made)
            self.assertAlmostEqual(status["spent_usd"], made * 0.0018, places=6)
            # Raise the cap: grading resumes, already-graded samples come from the cache.
            budget.cap_inr = 1000.0
            before = len(calls)
            for n in range(10):
                judges.grade(judge, cache_dir=cache, **dict(self.KW, item_id=f"i{n}", answer=f"a{n}"))
            self.assertEqual(len(calls) - before, 30 - made)  # only the missing samples

    def test_ledger_is_safe_under_concurrent_processes_and_threads(self):
        import subprocess
        import sys
        from concurrent.futures import ThreadPoolExecutor

        from llm.experiment1_v2 import spend

        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.json"
            code = ("from llm.experiment1_v2.spend import bump_for_test; "
                    f"bump_for_test(r'{ledger}', 20, 1000, 100)")
            procs = [subprocess.Popen([sys.executable, "-c", code], cwd=str(common.REPO_ROOT))
                     for _ in range(4)]
            with ThreadPoolExecutor(4) as pool:
                list(pool.map(lambda _: spend.bump_for_test(str(ledger), 10, 1000, 100), range(4)))
            for p in procs:
                self.assertEqual(p.wait(timeout=300), 0)
            data = json.loads(ledger.read_text(encoding="utf-8"))
            tracked = data["entries"]["concurrency-test"]["tracked"]
            self.assertEqual(tracked["calls"], 4 * 20 + 4 * 10)
            self.assertEqual(tracked["input_tokens"], 120 * 1000)
            self.assertEqual(tracked["output_tokens"], 120 * 100)
            self.assertEqual(data["entries"]["concurrency-test"]["reservations"], {})

    def test_injected_transport_has_no_budget_by_default(self):
        judge = judges.AzureChatJudge(judges.resolve_spec("primary", self.CFG, self.ENV),
                                      transport=lambda m, p: ("VERDICT: YES", self.MODEL))
        self.assertIsNone(judge.budget)

    def test_errors_are_redacted(self):
        def transport(messages, params):
            raise RuntimeError(f"401 for {SENTINEL_URL} with key {SENTINEL_KEY}")

        judge = judges.AzureChatJudge(judges.resolve_spec("secondary", self.CFG, self.ENV),
                                      transport=transport)
        with self.assertRaises(judges.JudgeCallError) as ctx:
            judge.complete("s", "u")
        self.assertNotIn(SENTINEL_KEY, str(ctx.exception))
        self.assertNotIn(SENTINEL_URL, str(ctx.exception))

    def test_r2_rubric_prompt_and_hash(self):
        from llm.experiment1_v2 import rubrics

        r2 = rubrics.get_rubric("e1v2-r2")
        self.assertIsNone(rubrics.get_rubric(rubrics.R1))
        self.assertEqual(rubrics.rubric_sha256(rubrics.R1), judges.PROMPT_SHA256)  # r1 unchanged
        self.assertEqual(judges.PROMPT_SHA256,
                         "b6f9a66873db8d3062f166c4a41789d556c648c6c6c1c2c360f0f8935f3aca11")
        self.assertNotEqual(r2.sha256, judges.PROMPT_SHA256)
        # default.txt is r1's instruction text, verbatim
        self.assertEqual(r2.texts["default"].rstrip("\n"), judges.V2_INSTRUCTIONS)
        text = r2.user_prompt(dataset="d2_support", question="And the seven-day one?",
                              reference="5 GB", answer="Please check the app.",
                              key_facts=["5 GB of data"],
                              history=[{"role": "user", "content": "Roaming packs?"},
                                       {"role": "assistant", "content": "Two packs."}])
        self.assertIn("states EVERY key fact", text)
        self.assertNotIn("Omitting some of the reference's details is CORRECT", text)
        self.assertIn("Conversation before the question:\nCaller: Roaming packs?\nAgent: Two packs.", text)
        self.assertIn("Key facts:\n- 5 GB of data", text)
        # d1_open under r2 keeps r1's rubric text
        d1 = r2.user_prompt(dataset="d1_open", question="q", reference="r", answer="a")
        self.assertTrue(d1.startswith(judges.V2_INSTRUCTIONS))
        # identity and cache key carry the rubric
        cfg = dict(self.CFG, prompt_version="e1v2-r2")
        spec = judges.resolve_spec("primary", cfg, self.ENV)
        self.assertEqual(spec.identity()["prompt_sha256"], r2.sha256)
        j2 = judges.AzureChatJudge(spec, transport=lambda m, p: None)
        j1 = judges.AzureChatJudge(judges.resolve_spec(
            "primary", dict(self.CFG, prompt_version=rubrics.R1), self.ENV), transport=lambda m, p: None)
        kw = dict(item_id="i", question="q", reference="r", answer="a", key_facts=["f"], aliases=[],
                  context="customer-support", dataset="d2_support")
        self.assertNotEqual(judges.cache_key(j1, **kw), judges.cache_key(j2, **kw))
        self.assertNotEqual(judges.cache_key(j2, **kw), judges.cache_key(j2, **dict(kw, history=["x"])))

    def test_r3_core_facts_prompt_key_and_fallback(self):
        from llm.experiment1_v2 import rubrics

        r3 = rubrics.get_rubric("e1v2-r3")
        self.assertNotEqual(r3.sha256, rubrics.get_rubric("e1v2-r2").sha256)
        self.assertEqual(r3.texts["default"], rubrics.get_rubric("e1v2-r2").texts["default"])
        text = r3.user_prompt(dataset="d2_support", question="Fuse?", reference="13 A; if it blows...",
                              answer="13 A", key_facts=["a 13 amp fuse", "blowing again means a fault"],
                              core_facts=["a 13 amp fuse"])
        self.assertIn("states EVERY core fact", text)
        self.assertIn("Core facts:\n- a 13 amp fuse", text)
        self.assertIn("Supplementary facts (may be omitted, must not be contradicted):\n"
                      "- blowing again means a fault", text)
        self.assertNotIn("Key facts:", text)
        fallback = r3.user_prompt(dataset="d2_support", question="q", reference="r", answer="a",
                                  key_facts=["f1", "f2"], core_facts=None)
        self.assertIn("Core facts:\n- f1\n\nSupplementary facts", fallback)
        # d1_open under r3 keeps r1's text, no core-fact block
        self.assertNotIn("Core facts", r3.user_prompt(dataset="d1_open", question="q", reference="r",
                                                       answer="a"))
        cfg3 = dict(self.CFG, prompt_version="e1v2-r3")
        j3 = judges.AzureChatJudge(judges.resolve_spec("primary", cfg3, self.ENV), transport=lambda m, p: None)
        j2 = judges.AzureChatJudge(judges.resolve_spec("primary", dict(self.CFG, prompt_version="e1v2-r2"),
                                                       self.ENV), transport=lambda m, p: None)
        kw = dict(item_id="i", question="q", reference="r", answer="a", key_facts=["f1", "f2"],
                  aliases=[], context="customer-support", dataset="d2_support")
        self.assertNotEqual(judges.cache_key(j3, **kw, core_facts=["f1"]),
                            judges.cache_key(j3, **kw, core_facts=["f2"]))
        self.assertEqual(judges.cache_key(j3, **kw, core_facts=None),  # fallback == explicit first fact
                         judges.cache_key(j3, **kw, core_facts=["f1"]))
        self.assertEqual(judges.cache_key(j2, **kw, core_facts=["f2"]),  # r2 keys ignore core facts
                         judges.cache_key(j2, **kw))
        self.assertEqual(judges.load_judges_config()["prompt_version"], "e1v2-r3")

    def test_label_warns_on_missing_core_facts(self):
        items = [{"id": "a", "dataset": "d2_support", "meta": {"key_facts": ["x"], "core_facts": ["x"]}},
                 {"id": "b", "dataset": "d2_support", "meta": {"key_facts": ["y"]}},
                 {"id": "c", "dataset": "d1_open", "meta": {}}]
        self.assertEqual(label.core_fact_warnings(items, "e1v2-r3"), ["b"])
        self.assertEqual(label.core_fact_warnings(items, "e1v2-r2"), [])
        self.assertEqual(label.item_core_facts(items[0]), ["x"])
        self.assertIsNone(label.item_core_facts(items[1]))

    def test_rubric_hash_tracks_file_text(self):
        from llm.experiment1_v2 import rubrics

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "vX"
            folder.mkdir()
            (folder / "default.txt").write_bytes(b"A rubric.\n")
            with mock.patch.object(rubrics, "RUBRIC_DIR", Path(tmp)):
                first = rubrics.load_file_rubric("vX").sha256
                (folder / "default.txt").write_bytes(b"A rubric.\r\n")
                self.assertEqual(rubrics.load_file_rubric("vX").sha256, first)  # CRLF-insensitive
                (folder / "default.txt").write_bytes(b"A changed rubric.\n")
                self.assertNotEqual(rubrics.load_file_rubric("vX").sha256, first)

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

    def test_latency_model_prefers_unqueued_calls(self):
        def rec(key, ttft, gen, n, in_flight):
            return {"call_key": key, "ttft_ms": ttft, "gen_ms": gen, "n_tokens": n,
                    "in_flight_at_send": in_flight}

        queued = [rec(f"q{i}", 20000.0, 22000.0, 41, 16) for i in range(10)]
        solo = [rec(f"s{i}", 300.0, 1500.0, 41, 1) for i in range(5)]
        model = analyze.latency_model([], queued + solo)
        self.assertTrue(model["valid_latency_estimate"])
        self.assertEqual(model["ttft_ms_median"], 300.0)
        self.assertAlmostEqual(model["per_token_ms_median"], 30.0)
        probe = analyze.latency_model([rec("p", 250.0, 1450.0, 41, 1)], queued)
        self.assertIn("probe", probe["source"])
        self.assertFalse(analyze.latency_model([], queued)["valid_latency_estimate"])
        r = {"n_tokens": 20, "t_first_k_ms": {"8": 999.0}, "gen_ms": 5000.0}
        self.assertAlmostEqual(analyze.signal_latency_ms(r, "first_k_mean_entropy_nats@8", model),
                               300.0 + 7 * 30.0)
        self.assertAlmostEqual(analyze.signal_latency_ms(r, "mean_entropy_nats", model), 300.0 + 19 * 30.0)
        self.assertEqual(analyze.signal_latency_ms(r, "first_k_mean_entropy_nats@8"), 999.0)

    def test_prefix_aware_logistic_recovers_coefficients(self):
        rng = np.random.default_rng(5)
        n = 20000
        x, f = rng.normal(0.5, 0.3, n), rng.uniform(0, 1, n)
        true = {"intercept": -2.0, "slope": 1.5, "frac_slope": 0.5, "interaction": 2.5}
        p = stats.predict_prefix_logistic(true, x, f)
        y = (rng.uniform(size=n) < p).astype(float)
        fit = stats.fit_prefix_logistic(x, f, y)
        self.assertEqual(fit["method"], "mle")
        for key, value in true.items():
            self.assertAlmostEqual(fit[key], value, delta=0.35)

    def _combined_records(self, n_items=120, seed=3):
        rng = np.random.default_rng(seed)
        labels, records = {}, []
        for i in range(n_items):
            need = bool(rng.random() < 0.5)
            labels[f"i{i}"] = need
            ent_item, sa_item = rng.normal(0.4 * need, 0.4), rng.normal(1.2 * need - 0.6, 1.0)
            for s in (11, 23):
                for frac in (0.3, 1.0):
                    records.append({
                        "item_id": f"i{i}", "draw_seed": s, "n_tokens": 40, "_sa_gen_ms": 500.0,
                        "t_first_k_ms": {"16": 900.0}, "gen_ms": 2500.0,
                        "prefix": {"source": "text", "kind": "full" if frac == 1.0 else "partial",
                                   "word_frac": frac, "text": f"q{i}-{frac}"},
                        "_flat": {"mean_entropy_nats": float(ent_item + rng.normal(0, 0.1)),
                                  "first_k_mean_entropy_nats@16": float(ent_item + rng.normal(0, 0.3)),
                                  "self_assessed_need": float(1 / (1 + np.exp(-(sa_item + rng.normal(0, 0.2)))))},
                    })
        return records, labels

    def test_combined_signal_is_out_of_fold_and_helps(self):
        records, labels = self._combined_records()
        combiners = analyze.add_combined_signals(records, labels, {}, 5)
        self.assertEqual(set(combiners), {"combined_v1", "combined_fast"})
        self.assertTrue(all("combined_v1" in r["_flat"] for r in records))  # every prefix gets one
        self.assertGreater(combiners["combined_v1"]["weights"][0], 0)
        self.assertGreater(combiners["combined_v1"]["weights"][1], 0)
        full = [r for r in records if r["prefix"]["kind"] == "full"]
        auc = {s: analyze.item_auc(analyze.item_scores(full, s), labels, {}, 0)["auc"]
               for s in ("mean_entropy_nats", "self_assessed_need", "combined_v1")}
        self.assertGreater(auc["combined_v1"], max(auc["mean_entropy_nats"], auc["self_assessed_need"]))
        # No leakage: flipping one item's label leaves that item's own OOF value unchanged.
        before = next(r["_flat"]["combined_v1"] for r in records if r["item_id"] == "i0")
        flipped = dict(labels, i0=not labels["i0"])
        records2, _ = self._combined_records()
        analyze.add_combined_signals(records2, flipped, {}, 5)
        after = next(r["_flat"]["combined_v1"] for r in records2 if r["item_id"] == "i0")
        self.assertAlmostEqual(before, after)
        # Latency: max of the parallel inputs
        r = full[0]
        self.assertEqual(analyze.signal_latency_ms(r, "combined_fast"), 900.0)
        self.assertEqual(analyze.signal_latency_ms(r, "combined_v1"), 2500.0)
        r2 = dict(r, _sa_gen_ms=1200.0)
        self.assertEqual(analyze.signal_latency_ms(r2, "combined_fast"), 1200.0)

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
class LabelingKitTests(unittest.TestCase):
    def _fixture(self, tmp: Path, with_core=True):
        items, draws = [], []
        types = ["general", "org_specific", "counter_prior", "ambiguous"]
        for n in range(60):
            item = {"id": f"d2_support-{n:05d}", "dataset": "d2_support", "domain": "telecom",
                    "org": "Kestrel Mobile", "question": f"Question {n}?",
                    "expected_answer": f"Answer {n}.", "answer_aliases": [],
                    "retrieval_needed_authored": n % 2 == 0, "item_type": types[n % 4],
                    "history": [{"role": "user", "content": "hi"}] if n % 5 == 0 else [],
                    "meta": {"key_facts": [f"fact {n}a", f"fact {n}b"],
                             **({"core_facts": [f"fact {n}a"]} if with_core else {})}}
            items.append(item)
            for seed in (11, 23, 37):
                p = (n + seed) % 3 != 0
                s = p if (n * seed) % 4 else not p
                draws.append({"item_id": item["id"], "draw_seed": seed, "condition": "closed_book",
                              "answer": f"answer {n} v{seed % 2}",
                              "judges": {"primary": {"answer_correct": p},
                                         "secondary": {"answer_correct": s}}})
        items_path = tmp / "items.jsonl"
        items_path.write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")
        labels = {"meta": {"split": "dev", "dataset": "d2_support",
                           "judges": {"primary": {"prompt_version": "e1v2-r2"}}}, "draws": draws}
        labels_path = tmp / "labels.json"
        labels_path.write_text(json.dumps(labels), encoding="utf-8")
        return items_path, labels_path

    def test_build_refused_without_core_facts(self):
        from simulator.judge_validation.v2_labeling import build_kit

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            items_path, labels_path = self._fixture(tmp, with_core=False)
            out = tmp / "kit"
            self.assertEqual(build_kit.main(["--labels", str(labels_path), "--items", str(items_path),
                                             "--n", "30", "--out-dir", str(out)]), 2)
            self.assertFalse((out / "sheet_annotator_A.csv").exists())
            # same sample with or without core facts (sample depends on labels + seed only)
            self.assertEqual(build_kit.main(["--labels", str(labels_path), "--items", str(items_path),
                                             "--n", "30", "--out-dir", str(out), "--allow-missing-core"]), 0)
            (tmp / "c").mkdir()
            items_core, _ = self._fixture(tmp / "c", with_core=True)
            out2 = tmp / "kit2"
            build_kit.main(["--labels", str(labels_path), "--items", str(items_core), "--n", "30",
                            "--out-dir", str(out2)])
            import csv
            ids = lambda p: sorted(r["case_id"] for r in csv.DictReader(open(p, encoding="utf-8-sig")))  # noqa: E731
            self.assertEqual(ids(out / "kit_master_DO_NOT_SHARE.csv"), ids(out2 / "kit_master_DO_NOT_SHARE.csv"))

    def test_build_merge_and_gate(self):
        import csv

        from simulator.judge_validation.v2_labeling import build_kit, merge_sheets

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            items_path, labels_path = self._fixture(tmp)
            out = tmp / "kit"
            self.assertEqual(build_kit.main(["--labels", str(labels_path), "--items", str(items_path),
                                             "--n", "45", "--out-dir", str(out)]), 0)
            read = lambda p: list(csv.DictReader(open(p, encoding="utf-8-sig", newline="")))  # noqa: E731
            a, b = read(out / "sheet_annotator_A.csv"), read(out / "sheet_annotator_B.csv")
            master = read(out / "kit_master_DO_NOT_SHARE.csv")
            self.assertEqual(len(a), 45)
            self.assertEqual({r["case_id"] for r in a}, {r["case_id"] for r in b})
            self.assertNotEqual([r["case_id"] for r in a], [r["case_id"] for r in b])  # shuffled apart
            for hidden in ("primary_r2", "secondary_r2", "item_type", "stratum", "path", "judges_agree"):
                self.assertNotIn(hidden, a[0])
            self.assertEqual(set(build_kit.SHEET_COLUMNS), set(a[0]))
            self.assertNotIn("key_facts", a[0])
            row = a[0]
            n = row["case_id"].split("-")[1].split("#")[0].lstrip("0") or "0"
            self.assertEqual(row["core_facts"], f"- fact {n}a")
            self.assertEqual(row["supplementary_facts"], f"- fact {n}b")
            self.assertTrue(all(r["verdict"] == "" for r in a))
            self.assertEqual(sum(r["gate_subset"] == "prompt_dev" for r in master), 15)
            per_item = {}
            for r in master:
                per_item[r["item_id"]] = per_item.get(r["item_id"], 0) + 1
            self.assertLessEqual(max(per_item.values()), 2)
            self.assertEqual({r["stratum"].split("|")[1] for r in master}, {"agree", "disagree"})
            # Deterministic for a fixed seed
            again = tmp / "kit2"
            build_kit.main(["--labels", str(labels_path), "--items", str(items_path), "--n", "45",
                            "--out-dir", str(again)])
            self.assertEqual(read(again / "sheet_annotator_A.csv"), a)

            def fill(rows, path, flip):
                with open(path, "w", encoding="utf-8-sig", newline="") as handle:
                    w = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    w.writeheader()
                    for r in rows:
                        n = int(r["case_id"].split("-")[1].split("#")[0])
                        ok = n % 2 == 0
                        w.writerow(dict(r, verdict=("correct" if ok != (flip and n % 7 == 0)
                                                    else "incorrect")))

            fill(a, out / "A_done.csv", False)
            fill(b, out / "B_done.csv", True)
            merged = out / "human.csv"
            self.assertEqual(merge_sheets.main(["--a", str(out / "A_done.csv"), "--b",
                                                str(out / "B_done.csv"), "--master",
                                                str(out / "kit_master_DO_NOT_SHARE.csv"),
                                                "--out", str(merged)]), 0)
            disputes = read(out / "adjudication_needed.csv")
            self.assertTrue(disputes)
            cases = score_judges.load_v2_labels(merged)
            self.assertEqual(len(cases), 45)
            self.assertEqual(sum(c["gate_subset"] == "prompt_dev" for c in cases), 15)
            self.assertEqual(cases[0]["key_facts"][0][:5], "fact ")
            report = score_judges.validate_v2(merged, ("primary", "secondary"), "mock", tmp / "jc", 2)
            self.assertEqual(report["gate"]["n_prompt_dev"], 15)
            # a sheet with an unknown case_id is refused
            bad = out / "bad.csv"
            bad.write_text("case_id,verdict\nnope,correct\n", encoding="utf-8")
            self.assertEqual(merge_sheets.main(["--a", str(bad), "--b", str(out / "B_done.csv"),
                                                "--master", str(out / "kit_master_DO_NOT_SHARE.csv"),
                                                "--out", str(tmp / "x.csv")]), 1)


class NullAuthoredLabelTests(unittest.TestCase):
    """d1_open has retrieval_needed_authored = null by design (CONTRACTS.md)."""

    def _null_dataset(self, tmp: Path) -> Path:
        items_path = _make_dataset(tmp, traces=False)
        rows = common.read_jsonl(items_path)
        for n, row in enumerate(rows):
            row.update(dataset="d1_open", retrieval_needed_authored=None, org=None, domain="open_domain",
                       item_type="popqa" if n % 4 else "multihop",
                       meta={**row["meta"], "log10_pop": float(n % 7)})
        items_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return items_path

    def test_null_authored_is_not_applicable_everywhere(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            items = self._null_dataset(tmp)
            self.assertTrue(all(i["retrieval_needed_authored"] is None for i in common.load_items(items)))
            self.assertNotIn("authored", analyze.label_maps(common.load_items(items), None))
            root = tmp / "results"
            out = root / "mistral7b_v03"
            self.assertEqual(run.main(["--items", str(items), "--split", "dev", "--backend", "mock",
                                       "--model", "mistral7b_v03", "--out-dir", str(out),
                                       "--seeds", "11,23,37,53,71", "--limit", "16"]), 0)
            # alias match only, LLM judges on a stratified subsample of 12 answers
            self.assertEqual(label.main(["--items", str(items), "--split", "dev", "--model",
                                         "mistral7b_v03", "--results-dir", str(out),
                                         "--judge-backend", "mock", "--judge-subsample", "12",
                                         "--cache-dir", str(tmp / "jc")]), 0)
            labels = common.load_json(out / "labels_dev.json")
            self.assertEqual(labels["meta"]["authored_labels"],
                             "not applicable (dataset has no authored labels)")
            judged = [d for d in labels["draws"] if d["judges"]]
            self.assertGreaterEqual(len(judged), 12)
            self.assertLess(len(judged), len(labels["draws"]))
            self.assertEqual(labels["agreement"]["closed_book"]["alias vs judge:primary"]["n"], len(judged))
            self.assertTrue(all(v["empirical_need"] is not None for v in labels["items"].values()))
            # alias-only labelling also works
            self.assertEqual(label.main(["--items", str(items), "--split", "dev", "--model",
                                         "mistral7b_v03", "--results-dir", str(out), "--judges",
                                         "none"]), 0)
            self.assertEqual(analyze.main(["--items", str(items), "--split", "dev", "--results-root",
                                           str(root), "--bootstrap", "20", "--no-figures"]), 0)
            analysis = common.load_json(root / "analysis_dev" / "analysis_dev.json")
            model = analysis["models"]["mistral7b_v03"]
            self.assertNotIn("authored", model["auc_full"]["mean_entropy_nats"])
            self.assertIn("empirical", model["auc_full"]["mean_entropy_nats"])
            self.assertEqual(model["authored_vs_empirical"],
                             "not applicable (dataset has no authored labels)")
            self.assertEqual(model["calibration_label"], "empirical")
            self.assertEqual(model["labels_not_applicable"], ["authored"])
            # even when authored is requested, it falls back to empirical
            self.assertEqual(analyze.main(["--items", str(items), "--split", "dev", "--results-root",
                                           str(root), "--bootstrap", "20", "--no-figures",
                                           "--calibration-label", "authored"]), 0)
            analysis = common.load_json(root / "analysis_dev" / "analysis_dev.json")
            self.assertEqual(analysis["models"]["mistral7b_v03"]["calibration_label"], "empirical")

    def test_judge_subsample_refused_for_judge_graded_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            items = _make_dataset(tmp, traces=False)  # v1_e1: graded by judge:primary
            out = tmp / "o"
            run.main(["--items", str(items), "--split", "dev", "--backend", "mock", "--out-dir",
                      str(out), "--seeds", "11", "--limit", "4", "--no-text-prefixes"])
            self.assertEqual(label.main(["--items", str(items), "--split", "dev", "--model",
                                         "mistral7b_v03", "--results-dir", str(out), "--judge-backend",
                                         "mock", "--judge-subsample", "3"]), 1)


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

            from llm.experiment1_v2 import self_assess
            self.assertEqual(self_assess.main([
                "--items", str(items), "--split", "dev", "--backend", "mock", "--model",
                "mistral7b_v03", "--out-dir", str(out), "--limit", "12"]), 0)
            sa_rows = common.read_jsonl(out / "self_assessment_dev.jsonl")
            sa_texts = {(r["item_id"], r["prefix_text"]) for r in sa_rows}
            sig_texts = {(r["item_id"], r["prefix"]["text"]) for r in records}
            self.assertEqual(sa_texts, sig_texts)  # one call per distinct prefix text, all covered
            self.assertEqual(self_assess.main([  # resumable: nothing left to do
                "--items", str(items), "--split", "dev", "--backend", "mock", "--model",
                "mistral7b_v03", "--out-dir", str(out), "--limit", "12"]), 0)
            self.assertEqual(len(common.read_jsonl(out / "self_assessment_dev.jsonl")), len(sa_rows))
            self.assertEqual(analyze.main([
                "--items", str(items), "--split", "dev", "--results-root", str(root),
                "--bootstrap", "50", "--controller-artifact"]), 0)
            analysis = common.load_json(root / "analysis_dev" / "analysis_dev.json")
            model = analysis["models"]["mistral7b_v03"]
            self.assertIn("empirical", model["auc_full"]["mean_entropy_nats"])
            self.assertEqual([p["word_frac"] for p in model["prefix_curve_text"]["mean_entropy_nats"]],
                             [0.3, 0.5, 0.7, 1.0])
            self.assertTrue(model["asr_curves"]["mean_entropy_nats"])
            self.assertEqual([r["k"] for r in model["first_k"]][:5], [4, 8, 16, "all", "self"])
            self.assertIn("self_assessed_need", model["auc_full"])
            # (the first 12 v1 items are all authored-needed, so only the empirical AUC exists)
            self.assertIsNotNone(model["auc_full"]["self_assessed_need"]["empirical"]["auc"])
            self.assertEqual(set(model["calibrations"]),
                             {"mean_entropy_nats", "self_assessed_need", "combined_v1", "combined_fast"})
            wf = next(iter(model["asr_word_frac_curve"]["mean_entropy_nats"].values()))
            self.assertEqual([p["word_frac"] for p in wf["curve"]], [0.3, 0.5, 0.7, 1.0])
            self.assertIn("auc_diff", wf["paired_diff"]["authored"])
            self.assertIn("authored", wf["c2_available"])
            self.assertIn("boundary_mass", model)
            self.assertNotIn("_beliefs_oof", json.dumps(analysis))
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
