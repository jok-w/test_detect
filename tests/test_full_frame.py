import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from person_tracking.detector import YoloPersonDetector
from person_tracking.engine import PersonTrackingEngine
from person_tracking.tracking_config import PersonTrackingConfig
from person_tracking.types import BoundingBox, PersonDetection


class FullFrameTests(unittest.TestCase):
    def make_engine(self, detections, roi_y_min=0):
        engine = object.__new__(PersonTrackingEngine)
        engine.config = SimpleNamespace(roi_y_min=roi_y_min, image_size=640)
        engine.detector = SimpleNamespace(detect=Mock(return_value=detections),
                                         select_single_person=YoloPersonDetector.select_single_person)
        return engine

    def test_4k_frame_is_passed_once_without_cropping_or_coordinate_offset(self):
        target = PersonDetection(BoundingBox(3100, 900, 3300, 1600), 0.9, 1, "stand")
        engine = self.make_engine([target], roi_y_min=100)
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        result = engine._detect_global_person(frame, None, roi_y_max=2000)
        engine.detector.detect.assert_called_once()
        call = engine.detector.detect.call_args
        self.assertIs(call.args[0], frame)
        self.assertEqual(call.kwargs, {"image_size": 640, "scope": "global"})
        self.assertIs(result, target)

    def test_roi_filters_centers_after_inference_without_moving_boxes(self):
        outside_top = PersonDetection(BoundingBox(10, 0, 60, 100), 0.99, 0, "squat")
        inside = PersonDetection(BoundingBox(20, 120, 80, 600), 0.8, 1, "stand")
        outside_bottom = PersonDetection(BoundingBox(20, 700, 80, 900), 0.95, 2, "prone")
        engine = self.make_engine([outside_top, inside, outside_bottom], roi_y_min=100)
        frame = np.zeros((900, 1300, 3), dtype=np.uint8)
        result = engine._detect_global_person(frame, None, roi_y_max=800)
        self.assertIs(result, inside)
        self.assertIs(engine.detector.detect.call_args.args[0], frame)

    def test_empty_results_do_not_trigger_repeated_inference(self):
        engine = self.make_engine([])
        self.assertIsNone(engine._detect_global_person(np.zeros((100, 200, 3), dtype=np.uint8), None, 100))
        engine.detector.detect.assert_called_once()

    def test_initial_warmup_and_recovery_use_full_frame_while_local_keeps_crop(self):
        engine = object.__new__(PersonTrackingEngine)
        engine.config = PersonTrackingConfig(model_path=Path("model.pt"), warmup_detections=3)
        target = PersonDetection(BoundingBox(500, 200, 600, 500), 0.9, 1, "stand")
        calls = []

        def detect(frame, image_size=None, scope="local"):
            calls.append((frame, scope, image_size))
            if scope == "global":
                return [target]
            return []

        engine.detector = SimpleNamespace(detect=detect, select_single_person=YoloPersonDetector.select_single_person)
        engine.reset()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        for timestamp in (0, 10, 20, 30, 400):
            engine.process_frame(frame, timestamp)
        self.assertEqual([scope for _, scope, _ in calls], ["global", "global", "global", "local", "global"])
        for actual_frame, scope, size in calls:
            if scope == "global":
                self.assertIs(actual_frame, frame)
                self.assertEqual(size, 640)
            else:
                self.assertLess(actual_frame.shape[0], frame.shape[0])
                self.assertEqual(size, 384)


if __name__ == "__main__":
    unittest.main()
