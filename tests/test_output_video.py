import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from person_tracking.__main__ import build_config, build_parser
from person_tracking.engine import FrameTrackingResult
from person_tracking.processor import PersonVideoProcessor, ProcessorConfig
from person_tracking.types import BoundingBox, CropWindow
from person_tracking.video_writer import GStreamerVideoWriter


class OutputVideoTests(unittest.TestCase):
    def make_processor(self, **kwargs):
        processor = object.__new__(PersonVideoProcessor)
        processor.config = ProcessorConfig(model_path=Path("models/best.pt"),
                                           input_path=Path("input.mp4"),
                                           output_path=Path("output.mp4"), **kwargs)
        processor.local_miss_count = 0
        return processor

    def test_output_dimensions_preserve_aspect_without_upscaling(self):
        processor = self.make_processor()
        for source, expected in [((3840, 2160), (1920, 1080)),
                                 ((1280, 720), (1280, 720)),
                                 ((1080, 1920), (1080, 1920)),
                                 ((1921, 1081), (1920, 1080))]:
            self.assertEqual(processor._resolve_output_size(*source), expected)
        processor.config = replace(processor.config, output_max_width=0)
        self.assertEqual(processor._resolve_output_size(3840, 2160), (3840, 2160))
        processor.config = replace(processor.config, output_max_width=1279)
        self.assertEqual(processor._resolve_output_size(3840, 2160), (1278, 718))

    def test_cli_default_and_original_size_option(self):
        parser = build_parser()
        self.assertEqual(build_config(parser.parse_args(["--input", "x.mp4"])).output_max_width, 1920)
        args = parser.parse_args(["--input", "x.mp4", "--output-max-width", "0"])
        self.assertEqual(build_config(args).output_max_width, 0)
        for width in (-1, 1):
            processor = self.make_processor(output_max_width=width)
            with patch.object(Path, "is_file", return_value=True):
                with self.assertRaisesRegex(ValueError, "输出最大宽度"):
                    processor.config.validate()

    def test_annotations_scale_without_mutating_tracking_or_input(self):
        processor = self.make_processor(roi_y_min=200)
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        result = FrameTrackingResult(BoundingBox(1000, 800, 1800, 1600), "model",
                                     "stand", 0.9, "local", "local",
                                     CropWindow(800, 600, 2000, 1800))
        with patch("person_tracking.engine.cv2.rectangle", wraps=cv2.rectangle) as rectangles:
            out = processor._draw_result(frame, 0, 0, result, 2000, output_size=(1920, 1080))
        self.assertEqual(out.shape, (1080, 1920, 3))
        self.assertEqual(rectangles.call_args_list[0].args[1:3], ((400, 300), (1000, 900)))
        self.assertEqual(rectangles.call_args_list[1].args[1:3], ((500, 400), (900, 800)))
        np.testing.assert_array_equal(out[100, 1500], [255, 160, 0])
        np.testing.assert_array_equal(out[1000, 1500], [255, 160, 0])
        self.assertFalse(frame.any())
        self.assertEqual(result.bbox, BoundingBox(1000, 800, 1800, 1600))

    def test_real_encoding_preserves_frames_fps_and_original_inference_size(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.avi"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 25, (320, 180))
            self.assertTrue(writer.isOpened())
            for value in (20, 40, 60, 80):
                writer.write(np.full((180, 320, 3), value, dtype=np.uint8))
            writer.release()
            for width, size in [(160, (160, 90)), (0, (320, 180))]:
                processor = self.make_processor(output_max_width=width, display=False)
                output = Path(folder) / f"output-{width}.mp4"
                processor.config = replace(processor.config, input_path=source, output_path=output)
                result = FrameTrackingResult(None, "none", None, None, "waiting", "global", None)
                processor.process_frame = Mock(return_value=(result, True))
                stats = processor.process()
                self.assertEqual((stats.total_frames, stats.written_frames), (4, 4))
                for call in processor.process_frame.call_args_list:
                    self.assertEqual(call.args[0].shape, (180, 320, 3))
                components = [stats.average_read_time_ms, stats.average_tracking_time_ms,
                              stats.average_drawing_time_ms, stats.average_display_time_ms,
                              stats.average_encoding_time_ms]
                self.assertAlmostEqual(sum(components), stats.average_frame_time_ms)
                self.assertGreaterEqual(stats.encoder_finalize_time_ms, 0)
                self.assertGreater(stats.processing_fps_with_finalize, 0)
                self.assertLessEqual(stats.processing_fps_with_finalize, stats.average_processing_fps)
                capture = cv2.VideoCapture(str(output))
                try:
                    self.assertEqual((int(capture.get(3)), int(capture.get(4))), size)
                    self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 25)
                    decoded = 0
                    while capture.read()[0]:
                        decoded += 1
                    self.assertEqual(decoded, 4)
                finally:
                    capture.release()

    def test_person_label_at_right_edge_fits_scaled_output(self):
        processor = self.make_processor()
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        result = FrameTrackingResult(BoundingBox(3600, 800, 3800, 1600), "model",
                                     "stand", 0.9, "local", "local", None)
        with patch.object(processor, "_draw_text") as draw_text:
            processor._draw_result(frame, 0, 0, result, 2160, output_size=(1920, 1080))
        _, text, (x, _), _ = draw_text.call_args_list[0].args
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 3)[0][0]
        self.assertLessEqual(x + text_width, 1920 - 4)

    def test_early_stop_drains_encoder_and_finalization_failure_never_returns_success(self):
        for fail_finalize in (False, True):
            with self.subTest(fail_finalize=fail_finalize):
                processor = self.make_processor(display=False)
                frame = np.zeros((180, 320, 3), np.uint8)
                capture = Mock()
                capture.read.return_value = (True, frame)
                capture.get.side_effect = lambda prop: {
                    cv2.CAP_PROP_FPS: 25, cv2.CAP_PROP_FRAME_WIDTH: 320,
                    cv2.CAP_PROP_FRAME_HEIGHT: 180, cv2.CAP_PROP_POS_MSEC: 0,
                }[prop]
                writer = Mock(spec=GStreamerVideoWriter)
                if fail_finalize:
                    writer.release.side_effect = RuntimeError("编码收尾失败")
                processor._create_writer = Mock(return_value=writer)
                processor._display_frame = Mock(side_effect=[True, False])
                result = FrameTrackingResult(None, "none", None, None, "waiting", "global", None)
                processor.process_frame = Mock(return_value=(result, True))
                with patch("person_tracking.processor.cv2.VideoCapture", return_value=capture):
                    if fail_finalize:
                        with self.assertRaisesRegex(RuntimeError, "编码收尾失败"):
                            processor.process()
                        writer.abort.assert_called_once()
                    else:
                        stats = processor.process()
                        self.assertTrue(stats.stopped_by_user)
                        self.assertEqual((stats.total_frames, stats.written_frames), (2, 2))
                        writer.abort.assert_not_called()
                writer.release.assert_called_once()
                self.assertEqual(writer.write.call_count, 2)
                capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
