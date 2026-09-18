import unittest

from person_tracking.benchmark import compare_results, timing_summary


class BenchmarkTests(unittest.TestCase):
    def test_comparison_reports_missing_boxes_and_class_disagreement(self):
        box = {"x1": 0, "y1": 0, "x2": 10, "y2": 10}
        baseline = [{"bbox": box, "class_name": "stand"}, {"bbox": box, "class_name": "stand"}]
        candidate = [{"bbox": box, "class_name": "squat"}, {"bbox": None, "class_name": None}]
        comparison = compare_results(baseline, candidate)
        self.assertEqual(comparison["mean_box_iou"], 1.0)
        self.assertEqual(comparison["class_agreement"], 0.0)
        self.assertEqual(comparison["presence_mismatch_frames"], 1)
        self.assertEqual(timing_summary([]), {"count": 0, "mean_ms": None, "p95_ms": None})
        with self.assertRaises(ValueError):
            compare_results(baseline, candidate[:1])


if __name__ == "__main__":
    unittest.main()
