import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np

from person_tracking.detector import YoloPersonDetector
from person_tracking.types import BoundingBox


class DetectorResultTests(unittest.TestCase):
    def test_nonempty_results_in_single_and_batch_inference(self):
        boxes = MagicMock()
        boxes.__len__.return_value = 1
        boxes.xyxy.detach.return_value.cpu.return_value.numpy.return_value = np.array([[10, 20, 30, 40]])
        boxes.conf.detach.return_value.cpu.return_value.numpy.return_value = np.array([0.9])
        boxes.cls.detach.return_value.cpu.return_value.numpy.return_value = np.array([1.0])
        detector = object.__new__(YoloPersonDetector)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        for names, expected_name in (({1: "standing"}, "standing"), (["sitting", "standing"], "standing"), ({}, "1")):
            for batched in (False, True):
                with self.subTest(names=names, batched=batched):
                    result = SimpleNamespace(boxes=boxes, names=names)
                    detector._predict = Mock(return_value=[result, result] if batched else [result])
                    groups = detector.detect_batch([frame, frame]) if batched else [detector.detect(frame)]
                    self.assertEqual(len(groups), 2 if batched else 1)
                    for detections in groups:
                        self.assertEqual(len(detections), 1)
                        self.assertEqual(detections[0].bbox, BoundingBox(10, 20, 30, 40))
                        self.assertAlmostEqual(detections[0].confidence, 0.9)
                        self.assertEqual(detections[0].class_id, 1)
                        self.assertEqual(detections[0].class_name, expected_name)


if __name__ == "__main__":
    unittest.main()
