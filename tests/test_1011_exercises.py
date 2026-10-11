"""Regression checks for the current synthetic-data exercise contracts."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "10"), str(ROOT / "11")]
from teacher_t02_1010 import assign_nearest
from teacher_t03_1011 import update_centers, squared_objective
from teacher_t04_1011 import run_kmeans
from basic_test import find_duplicate_groups
from basic_text_01 import parse_labeled_line
from basic_test_02 import build_records, to_text_rows, find_label_conflicts


class TextChecks(unittest.TestCase):
    def test_parse_preserves_body_tabs_and_spaces(self):
        self.assertEqual(parse_labeled_line("spam\t  Gift\tReply  \r\n"),
                         ("spam", "  Gift\tReply  "))

    def test_records_keep_order_ids_and_labels(self):
        rows = build_records(["ham\tMeeting\n", "spam\tPrize\tNow\r\n"])
        self.assertEqual(rows, [(1, "Meeting", 0), (2, "Prize\tNow", 1)])
        self.assertEqual(to_text_rows(rows), [(1, "Meeting"), (2, "Prize\tNow")])

    def test_duplicates_keep_all_original_ids(self):
        self.assertEqual(find_duplicate_groups([(7, "A"), (2, "B"), (9, "A")]),
                         {"A": [7, 9]})

    def test_conflicts_exclude_agreeing_duplicates(self):
        rows = [(7, "A", 0), (2, "B", 1), (9, "A", 0),
                (12, "B", 0), (16, "B", 1), (20, "C", 1)]
        self.assertEqual(find_label_conflicts(rows), {"B": [2, 12, 16]})


class ClusteringChecks(unittest.TestCase):
    def setUp(self):
        self.points = np.array([(1, 1), (1, 2), (2, 1),
                                (7, 7), (7, 8), (8, 7)], dtype=np.float64)
        self.initial = np.array([[1., 1.], [7., 7.]])
        self.expected = np.array([[4 / 3, 4 / 3], [22 / 3, 22 / 3]])

    def test_single_update_matches_reference(self):
        _, labels = assign_nearest(self.points, self.initial)
        updated = update_centers(self.points, labels, 2)
        np.testing.assert_allclose(updated, self.expected, rtol=0, atol=1e-12)
        self.assertAlmostEqual(squared_objective(self.points, updated, labels), 8 / 3)

    def test_convergence_records_stopping_round(self):
        centers, labels, history, reason = run_kmeans(self.points, self.initial, 5)
        np.testing.assert_allclose(centers, self.expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1, 1])
        self.assertEqual([round_no for round_no, _ in history], [1, 2])
        np.testing.assert_allclose([j for _, j in history], [8 / 3, 8 / 3],
                                   rtol=0, atol=1e-12)
        self.assertLessEqual(history[1][1], history[0][1] + 1e-12)
        self.assertEqual(reason, "中心不变")
        np.testing.assert_array_equal(self.initial, [[1., 1.], [7., 7.]])

    def test_round_limit_preserves_completed_record(self):
        centers, labels, history, reason = run_kmeans(self.points, self.initial, 1)
        self.assertEqual(len(history), 1)
        self.assertEqual(reason, "达到最大轮数")
        self.assertAlmostEqual(history[-1][1], squared_objective(self.points, centers, labels))

    def test_convergence_on_final_allowed_round_has_priority(self):
        _, _, history, reason = run_kmeans(self.points, self.initial, 2)
        self.assertEqual(len(history), 2)
        self.assertEqual(reason, "中心不变")


if __name__ == "__main__":
    unittest.main()
