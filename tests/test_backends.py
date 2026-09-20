import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from person_tracking.detector import YoloPersonDetector
from person_tracking.model_artifacts import ENGINE_STRATEGY, artifact_paths, file_sha256, validate_engine


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

    def write_engine(self, scope, size, batch=1):
        _, global_path, local_path = artifact_paths(self.model_path)
        path = global_path if scope == "global" else local_path
        metadata = {"task": "detect", "names": {0: "standing"}, "person_tracking": {
            "schema": 1, "source_sha256": file_sha256(self.model_path), "image_size": size,
            "max_batch": batch, "dynamic": batch > 1, "strategy": ENGINE_STRATEGY,
            "precision": "fp16", "runtime": ENVIRONMENT}}
        data = json.dumps(metadata).encode()
        path.write_bytes(len(data).to_bytes(4, "little") + data + b"engine")
        return metadata

    def test_two_engines_route_one_image_per_scope(self):
        self.write_engine("global", 640)
        self.write_engine("local", 384, 1)
        detector = YoloPersonDetector(self.model_path, backend="tensorrt")
        frame = np.zeros((180, 300, 3), dtype=np.uint8)
        detector.detect(frame, scope="global")
        detector.detect(frame)
        self.assertNotIn("best.pt", self.models)
        global_model, local_model = self.models["best.global.engine"], self.models["best.local.engine"]
        self.assertEqual(global_model.predict.call_args.kwargs["imgsz"], 640)
        self.assertEqual(local_model.predict.call_args.kwargs["imgsz"], 384)
        self.assertIs(local_model.predict.call_args.kwargs["rect"], False)
        with self.assertRaises(ValueError):
            detector.detect(frame, image_size=640)
        with self.assertRaises(ValueError):
            detector.detect([frame, frame], scope="global")
        with self.assertRaises(ValueError):
            detector.detect(frame, scope="unknown")

    def test_auto_falls_back_per_scope_for_shape_mismatch(self):
        self.write_engine("global", 640)
        self.write_engine("local", 640, 1)
        with self.assertLogs("person_tracking.detector", level="WARNING"):
            detector = YoloPersonDetector(self.model_path)
        self.assertEqual(detector.backend_by_scope, {"global": "tensorrt", "local": "pt"})

    def test_cpu_never_loads_engine(self):
        self.write_engine("global", 640)
        with patch.object(YoloPersonDetector, "_select_device", return_value="cpu"):
            detector = YoloPersonDetector(self.model_path)
            self.assertEqual(detector.backend_by_scope, {"global": "pt", "local": "pt"})
            with self.assertRaises(ValueError):
                YoloPersonDetector(self.model_path, backend="tensorrt")
        self.assertEqual(set(self.models), {"best.pt"})

    def test_old_weights_fall_back_but_explicit_backend_fails(self):
        self.write_engine("global", 640)
        self.write_engine("local", 384, 1)
        self.model_path.write_bytes(b"updated weights")
        with self.assertLogs("person_tracking.detector", level="WARNING"):
            detector = YoloPersonDetector(self.model_path)
        self.assertEqual(detector.backend_by_scope, {"global": "pt", "local": "pt"})
        with self.assertRaisesRegex(RuntimeError, "权重不一致"):
            YoloPersonDetector(self.model_path, backend="tensorrt")

    def test_warmup_failure_falls_back_but_inference_errors_propagate(self):
        self.write_engine("global", 640)
        self.write_engine("local", 384, 1)
        with patch.object(YoloPersonDetector, "_warmup", side_effect=RuntimeError("bad engine")):
            with self.assertLogs("person_tracking.detector", level="WARNING"):
                detector = YoloPersonDetector(self.model_path)
        self.assertEqual(detector.backend_by_scope, {"global": "pt", "local": "pt"})
        detector.model.predict.side_effect = NameError("bug in result parser")
        with self.assertRaises(NameError):
            detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))

    def test_runtime_and_profile_mismatches_are_rejected(self):
        metadata = self.write_engine("global", 640)
        source_hash = file_sha256(self.model_path)
        for key in ENVIRONMENT:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_engine(metadata, source_hash, 640, {**ENVIRONMENT, key: "different"})
        with self.assertRaises(ValueError):
            validate_engine(metadata, source_hash, 384, ENVIRONMENT)

    def test_old_dynamic_engine_falls_back_or_requests_rebuild(self):
        self.write_engine("global", 640, 4)
        self.write_engine("local", 384)
        with self.assertLogs("person_tracking.detector", level="WARNING") as logs:
            detector = YoloPersonDetector(self.model_path)
        self.assertEqual(detector.backend_by_scope, {"global": "pt", "local": "tensorrt"})
        self.assertIn("重新构建", " ".join(logs.output))
        with self.assertRaisesRegex(RuntimeError, "固定 batch=1"):
            YoloPersonDetector(self.model_path, backend="tensorrt")

    def test_scope_selection_does_not_depend_on_image_size(self):
        self.write_engine("global", 640)
        self.write_engine("local", 640)
        detector = YoloPersonDetector(self.model_path, backend="tensorrt", local_image_size=640)
        for model in self.models.values():
            model.predict.reset_mock()
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        detector.detect(frame, scope="global")
        self.models["best.global.engine"].predict.assert_called_once()
        self.models["best.local.engine"].predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
