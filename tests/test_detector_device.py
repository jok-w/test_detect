import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from person_tracking.detector import YoloPersonDetector


class DetectorDeviceTests(unittest.TestCase):
    def make_detector(self, cuda_available, device=None):
        cuda = Mock()
        cuda.is_available.return_value = cuda_available
        model = Mock()
        model.predict.return_value = []
        modules = {
            "torch": SimpleNamespace(cuda=cuda),
            "ultralytics": SimpleNamespace(YOLO=Mock(return_value=model)),
        }
        with patch.dict("sys.modules", modules), patch.object(Path, "is_file", return_value=True):
            detector = YoloPersonDetector(Path("model.pt"), device=device)
        return detector, cuda

    def test_default_gpu_reaches_single_and_batch_inference(self):
        with self.assertLogs("person_tracking.detector", level="INFO") as logs:
            detector, _ = self.make_detector(True)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        detector.detect(frame)
        detector.model.predict.return_value = [SimpleNamespace(boxes=None)] * 2
        detector.detect_batch([frame, frame])
        self.assertEqual([call.kwargs["device"] for call in detector.model.predict.call_args_list], ["0", "0"])
        self.assertIn("GPU", logs.output[0])

    def test_unavailable_cuda_falls_back_even_when_gpu_requested(self):
        for device in (None, "auto", "0", "cuda", "cuda:0", "0,1"):
            with self.subTest(device=device), self.assertLogs("person_tracking.detector", level="WARNING") as logs:
                detector, _ = self.make_detector(False, device)
                detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))
                self.assertEqual(detector.model.predict.call_args.kwargs["device"], "cpu")
                self.assertIn("自动回退到 CPU", logs.output[0])

    def test_explicit_cpu_overrides_available_gpu(self):
        with self.assertLogs("person_tracking.detector", level="INFO") as logs:
            detector, cuda = self.make_detector(True, "cpu")
        self.assertEqual(detector.device, "cpu")
        cuda.is_available.assert_not_called()
        self.assertIn("CPU（手动指定）", logs.output[0])

    def test_requested_gpu_index_is_preserved(self):
        for requested, expected in (("cuda", "0"), ("cuda:1", "1"), ("1", "1"), ("0,1", "0,1")):
            with self.subTest(device=requested):
                detector, _ = self.make_detector(True, requested)
                self.assertEqual(detector.device, expected)


if __name__ == "__main__":
    unittest.main()
