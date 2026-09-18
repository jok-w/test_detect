import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from person_tracking.detector import YoloPersonDetector
from person_tracking.model_artifacts import artifact_paths, file_sha256, validate_engine


ENVIRONMENT = {"tensorrt": "10.3.0", "cuda": "12.6", "gpu_name": "Orin",
               "gpu_capability": [8, 7], "ultralytics": "8.4.154"}


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model_path = Path(self.directory.name) / "best.pt"
        self.model_path.write_bytes(b"test model")
        self.models = {}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.factory = Mock(side_effect=self.load_model)
        self.stack.enter_context(patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=self.factory)}))
        self.stack.enter_context(patch.object(YoloPersonDetector, "_select_device", return_value="0"))
        self.stack.enter_context(patch("person_tracking.detector.runtime_info", return_value=ENVIRONMENT))

    def load_model(self, path, **kwargs):
        model = Mock()
        model.predict.side_effect = lambda **kw: [SimpleNamespace(boxes=None)] * (len(kw["source"]) if isinstance(kw["source"], list) else 1)
        self.models[Path(path).name] = model
        return model

    def write_engine(self, scope, size, batch):
        _, global_path, local_path = artifact_paths(self.model_path)
        path = global_path if scope == "global" else local_path
        metadata = {"task": "detect", "names": {0: "standing"}, "person_tracking": {
            "schema": 1, "source_sha256": file_sha256(self.model_path), "image_size": size,
            "max_batch": batch, "precision": "fp16", "runtime": ENVIRONMENT}}
        data = json.dumps(metadata).encode()
        path.write_bytes(len(data).to_bytes(4, "little") + data + b"engine")
        return metadata

    def test_two_engines_route_local_and_all_tail_batches(self):
        self.write_engine("global", 640, 4)
        self.write_engine("local", 384, 1)
        detector = YoloPersonDetector(self.model_path, backend="tensorrt")
        frame = np.zeros((180, 300, 3), dtype=np.uint8)
        for count in range(1, 5):
            self.assertEqual(len(detector.detect_batch([frame] * count)), count)
        detector.detect(frame)
        self.assertNotIn("best.pt", self.models)
        global_model, local_model = self.models["best.global.engine"], self.models["best.local.engine"]
        self.assertEqual(global_model.predict.call_args.kwargs["imgsz"], 640)
        self.assertEqual(local_model.predict.call_args.kwargs["imgsz"], 384)
        self.assertIs(local_model.predict.call_args.kwargs["rect"], False)
        with self.assertRaises(ValueError):
            detector.detect(frame, image_size=640)
        with self.assertRaises(ValueError):
            detector.detect_batch([frame] * 5)

    def test_auto_falls_back_per_scope_for_shape_mismatch(self):
        self.write_engine("global", 640, 4)
        self.write_engine("local", 640, 1)
        with self.assertLogs("person_tracking.detector", level="WARNING"):
            detector = YoloPersonDetector(self.model_path)
        self.assertEqual(detector.backend_by_scope, {"global": "tensorrt", "local": "pt"})

    def test_cpu_never_loads_engine(self):
        self.write_engine("global", 640, 4)
        with patch.object(YoloPersonDetector, "_select_device", return_value="cpu"):
            detector = YoloPersonDetector(self.model_path)
            self.assertEqual(detector.backend_by_scope, {"global": "pt", "local": "pt"})
            with self.assertRaises(ValueError):
                YoloPersonDetector(self.model_path, backend="tensorrt")
        self.assertEqual(set(self.models), {"best.pt"})

    def test_old_weights_fall_back_but_explicit_backend_fails(self):
        self.write_engine("global", 640, 4)
        self.write_engine("local", 384, 1)
        self.model_path.write_bytes(b"updated weights")
        with self.assertLogs("person_tracking.detector", level="WARNING"):
            detector = YoloPersonDetector(self.model_path)
        self.assertEqual(detector.backend_by_scope, {"global": "pt", "local": "pt"})
        with self.assertRaisesRegex(RuntimeError, "权重不一致"):
            YoloPersonDetector(self.model_path, backend="tensorrt")

    def test_warmup_failure_falls_back_but_inference_errors_propagate(self):
        self.write_engine("global", 640, 4)
        self.write_engine("local", 384, 1)
        with patch.object(YoloPersonDetector, "_warmup", side_effect=RuntimeError("bad engine")):
            with self.assertLogs("person_tracking.detector", level="WARNING"):
                detector = YoloPersonDetector(self.model_path)
        self.assertEqual(detector.backend_by_scope, {"global": "pt", "local": "pt"})
        detector.model.predict.side_effect = NameError("bug in result parser")
        with self.assertRaises(NameError):
            detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))

    def test_runtime_and_profile_mismatches_are_rejected(self):
        metadata = self.write_engine("global", 640, 4)
        source_hash = file_sha256(self.model_path)
        for key in ENVIRONMENT:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_engine(metadata, source_hash, 640, 4, {**ENVIRONMENT, key: "different"})
        for size, batch in ((384, 4), (640, 5)):
            with self.subTest(size=size, batch=batch), self.assertRaises(ValueError):
                validate_engine(metadata, source_hash, size, batch, ENVIRONMENT)


if __name__ == "__main__":
    unittest.main()
