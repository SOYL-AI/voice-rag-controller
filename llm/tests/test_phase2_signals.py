import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm.client import AlternativeLogprob, MockClient, TokenLogprob, load_config
from eventlog.log_writer import LogWriter
from llm.experiment1.run import load_dataset, roc_auc, DEFAULT_DATASET
from llm.signals import build_signal_snapshot, retrieval_summary
from llm.tracker import uncertainty


class AucTests(unittest.TestCase):
    def test_auc_orients_higher_score_as_retrieval_needed(self):
        self.assertEqual(roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(roc_auc([0, 0, 1, 1], [0.8, 0.9, 0.1, 0.2]), 0.0)

    def test_auc_gives_half_credit_for_ties(self):
        self.assertEqual(roc_auc([0, 1], [0.5, 0.5]), 0.5)


class SignalTests(unittest.TestCase):
    def test_token_entropy_includes_residual_mass(self):
        token = TokenLogprob(
            "yes",
            -0.223143551,
            [
                AlternativeLogprob("yes", -0.223143551),
                AlternativeLogprob("no", -2.302585093),
            ],
        )
        self.assertAlmostEqual(token.confidence, 0.8, places=6)
        self.assertIsNotNone(token.entropy_nats)
        self.assertGreater(token.entropy_nats, 0.0)

    def test_retrieval_distance_margin_has_correct_direction(self):
        summary = retrieval_summary([0.8, 0.2, 0.5], "distance")
        self.assertEqual(summary["best_score"], 0.2)
        self.assertAlmostEqual(summary["top_two_margin"], 0.3)

    def test_tracker_uncertainty_excludes_boolean_slot(self):
        signal = uncertainty(
            {
                "intent": "book_spa",
                "room": "unknown",
                "time": "4pm",
                "people": 2,
                "confirmed": False,
            }
        )
        self.assertEqual(signal["measured_slot_count"], 4)
        self.assertEqual(signal["unknown_count"], 1)
        self.assertEqual(signal["unknown_fraction"], 0.25)

    @patch("llm.client.time.sleep", return_value=None)
    def test_mock_exposes_deterministic_entropy_signals(self, _sleep):
        cfg = load_config()
        first = MockClient(cfg).generate(
            [{"role": "user", "content": "test"}], seed=11
        )
        second = MockClient(cfg).generate(
            [{"role": "user", "content": "test"}], seed=11
        )
        self.assertEqual(first.logprob_dicts(), second.logprob_dicts())
        snapshot = build_signal_snapshot(generation=first, tracker_cfg=cfg)
        self.assertIn("mean_entropy_nats", snapshot["model"]["uncertainty"])

    @patch("llm.client.time.sleep", return_value=None)
    def test_llm_final_event_logs_phase2_signal_payload(self, _sleep):
        cfg = load_config()
        with tempfile.TemporaryDirectory() as directory:
            writer = LogWriter("phase2_test", log_dir=directory)
            try:
                MockClient(cfg).generate(
                    [{"role": "user", "content": "test"}],
                    log_writer=writer,
                    turn_id=1,
                    seed=11,
                )
            finally:
                writer.close()

            events = [
                json.loads(line)
                for line in Path(writer._path).read_text(encoding="utf-8").splitlines()
            ]
        final = next(event for event in events if event["event_type"] == "llm_final")
        self.assertIn("mean_entropy_nats", final["payload"])
        self.assertIn("mean_confidence", final["payload"])
        self.assertIn("entropy_nats", final["payload"]["logprobs"][0])


class DatasetTests(unittest.TestCase):
    def test_experiment_dataset_is_balanced_and_answer_bearing(self):
        rows = load_dataset(DEFAULT_DATASET)
        positives = sum(row["retrieval_needed"] for row in rows)
        self.assertEqual(positives, len(rows) - positives)
        self.assertTrue(all(row["expected_answer"] for row in rows))


if __name__ == "__main__":
    unittest.main()
