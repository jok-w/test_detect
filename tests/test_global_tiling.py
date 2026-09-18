import unittest
from types import SimpleNamespace

import numpy as np

from person_tracking.engine import PersonTrackingEngine
from person_tracking.types import BoundingBox, PersonDetection


class FakeDetector:
    def __init__(self, target_tile_index: int) -> None:
        self.target_tile_index = target_tile_index
        self.tile_count = 0
        self.batch_lengths: list[int] = []

    def detect_batch(self, frames: list[np.ndarray], image_size: int) -> list[list[PersonDetection]]:
        self.batch_lengths.append(len(frames))
        assert image_size == 640
        detections = []
        for frame in frames:
            assert frame.shape == (640, 640, 3)
            if self.tile_count == self.target_tile_index:
                detections.append([PersonDetection(BoundingBox(10, 20, 50, 100), 0.9, 0, "person")])
            else:
                detections.append([])
            self.tile_count += 1
        return detections

    @staticmethod
    def select_single_person(detections, predicted_bbox):
        return max(detections, key=lambda detection: detection.confidence, default=None)


class GlobalTilingTests(unittest.TestCase):
    def test_4k_tiles_cover_frame_with_overlap(self) -> None:
        starts_x = PersonTrackingEngine._tile_starts(3840, 640, 128)
        starts_y = PersonTrackingEngine._tile_starts(2160, 640, 128)
        self.assertEqual((len(starts_x), len(starts_y)), (8, 4))
        for starts, length in ((starts_x, 3840), (starts_y, 2160)):
            self.assertEqual(starts[0], 0)
            self.assertEqual(starts[-1] + 640, length)
            self.assertTrue(all(0 < right - left <= 512 for left, right in zip(starts, starts[1:])))

    def test_global_batch_and_coordinate_mapping_with_vertical_roi(self) -> None:
        engine = object.__new__(PersonTrackingEngine)
        engine.config = SimpleNamespace(
            roi_y_min=100,
            global_tile_size=640,
            global_tile_overlap=128,
            global_tile_batch_size=4,
            image_size=640,
            iou_threshold=0.45,
        )
        engine.detector = FakeDetector(target_tile_index=5)
        frame = np.zeros((900, 1300, 3), dtype=np.uint8)

        detection = engine._detect_global_person(frame, None, roi_y_max=900)

        self.assertEqual(engine.detector.batch_lengths, [4, 2])
        self.assertEqual(detection.bbox, BoundingBox(670, 280, 710, 360))

    def test_cross_tile_nms_ignores_pose_class(self) -> None:
        detections = [
            PersonDetection(BoundingBox(100, 100, 200, 300), 0.9, 0, "standing"),
            PersonDetection(BoundingBox(105, 105, 205, 305), 0.8, 1, "bending"),
            PersonDetection(BoundingBox(400, 100, 500, 300), 0.7, 0, "standing"),
        ]
        kept = PersonTrackingEngine._deduplicate_detections(detections, 0.45)
        self.assertEqual([d.confidence for d in kept], [0.9, 0.7])


if __name__ == "__main__":
    unittest.main()
