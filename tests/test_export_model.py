import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from person_tracking.export_model import build_engine
from person_tracking.model_artifacts import read_engine_metadata


class EngineBuildTests(unittest.TestCase):
    def build(self, batch=4, size=640, fail_warmup=False):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        onnx = root / "best.onnx"
        onnx.write_bytes(b"onnx")
        output = root / "best.engine"
        output.write_bytes(b"previous valid engine")
        tensor = SimpleNamespace(name="images", shape=(-1, 3, -1, -1), dtype="float32")
        network = Mock(num_inputs=1)
        network.get_input.return_value = tensor
        builder = Mock(platform_has_fast_fp16=True)
        builder.create_network.return_value = network
        builder.build_serialized_network.return_value = b"built engine"
        parser = Mock()
        parser.parse_from_file.return_value = True
        trt = SimpleNamespace(__version__="10.3.0", Builder=Mock(return_value=builder),
                              Logger=Mock(INFO=1), OnnxParser=Mock(return_value=parser),
                              MemoryPoolType=SimpleNamespace(WORKSPACE="workspace"),
                              BuilderFlag=SimpleNamespace(FP16="fp16"), float32="float32")
        yolo = Mock()
        yolo.predict.side_effect = (RuntimeError("warmup failed") if fail_warmup else
                                    lambda **kw: [SimpleNamespace(boxes=None)] * len(kw["source"]))
        metadata = {"task": "detect", "names": {0: "stand"}, "person_tracking": {
            "schema": 1, "source_sha256": "hash", "dynamic": True, "precision": "fp32", "ultralytics": "test"}}
        with patch.dict("sys.modules", {"torch": Mock(), "tensorrt": trt,
                                        "ultralytics": SimpleNamespace(YOLO=Mock(return_value=yolo))}), \
                patch("person_tracking.export_model.runtime_info", return_value={"ultralytics": "test"}), \
                patch("person_tracking.export_model.read_onnx_metadata", return_value=metadata):
            if fail_warmup:
                with self.assertRaisesRegex(RuntimeError, "warmup failed"):
                    build_engine(onnx, output, size, batch)
            else:
                build_engine(onnx, output, size, batch)
        return tensor, builder, yolo, output

    def test_global_profile_accepts_partial_batches_and_metadata_is_preserved(self):
        tensor, builder, model, output = self.build()
        self.assertEqual(tensor.shape, (-1, 3, 640, 640))
        builder.create_optimization_profile.return_value.set_shape.assert_called_once_with(
            "images", (1, 3, 640, 640), (4, 3, 640, 640), (4, 3, 640, 640))
        self.assertEqual([len(c.kwargs["source"]) for c in model.predict.call_args_list], [1, 4])
        metadata = read_engine_metadata(output)
        self.assertEqual(metadata["names"], {"0": "stand"})
        self.assertEqual(metadata["person_tracking"]["image_size"], 640)
        self.assertEqual(metadata["person_tracking"]["max_batch"], 4)

    def test_local_engine_is_fixed_shape(self):
        tensor, builder, model, output = self.build(batch=1, size=384)
        self.assertEqual(tensor.shape, (1, 3, 384, 384))
        builder.create_optimization_profile.assert_not_called()
        self.assertFalse(read_engine_metadata(output)["dynamic"])

    def test_failed_warmup_preserves_existing_engine(self):
        _, _, _, output = self.build(fail_warmup=True)
        self.assertEqual(output.read_bytes(), b"previous valid engine")


if __name__ == "__main__":
    unittest.main()
