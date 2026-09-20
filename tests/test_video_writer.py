import os
import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from person_tracking.__main__ import build_config, build_parser
from person_tracking.video_writer import (
    GStreamerVideoWriter, check_gstreamer, create_video_writer, pipeline_command,
)


class VideoWriterTests(unittest.TestCase):
    def make_writer(self, folder, timeout=30):
        process = Mock()
        process.poll.return_value = None
        process.stdin.fileno.return_value = 123
        process.wait.return_value = 0
        with patch("person_tracking.video_writer.subprocess.Popen", return_value=process), \
                patch("person_tracking.video_writer.os.set_blocking", create=True):
            writer = GStreamerVideoWriter("gst-launch-1.0", Path(folder) / "输出 space.mp4",
                                          30000 / 1001, 6, 2, 8000000, timeout=timeout)
        return writer, process

    def test_partial_pipe_writes_preserve_every_pixel_and_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            writer, _ = self.make_writer(folder)
            # Non-contiguous BGR, width is even but not divisible by four.
            frame = np.arange(72, dtype=np.uint8).reshape(2, 12, 3)[:, ::2]
            received = bytearray()

            def write_chunk(fd, data):
                self.assertEqual(fd, 123)
                size = min(7, len(data))
                received.extend(data[:size])
                return size

            try:
                with patch("person_tracking.video_writer.select.select", return_value=([], [123], [])), \
                        patch("person_tracking.video_writer.os.write", side_effect=write_chunk):
                    writer.write(frame)
                    writer.write(frame)
                self.assertEqual(writer.frames, 2)
                expected = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA).tobytes()
                self.assertEqual(received, expected * 2)
            finally:
                writer.abort()

    def test_broken_pipe_aborts_without_counting_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            writer, process = self.make_writer(folder)
            with patch("person_tracking.video_writer.select.select", return_value=([], [123], [])), \
                    patch("person_tracking.video_writer.os.write", side_effect=BrokenPipeError):
                with self.assertRaisesRegex(RuntimeError, "硬件编码写入失败"):
                    writer.write(np.zeros((2, 6, 3), np.uint8))
            self.assertEqual(writer.frames, 0)
            process.kill.assert_called_once()
            self.assertFalse(writer.isOpened())
            # A subsequent submission must not mask the encoder error with a closed-log error.
            with self.assertRaisesRegex(RuntimeError, "编码进程已退出"):
                writer.write(np.zeros((2, 6, 3), np.uint8))

    def test_stalled_pipe_times_out(self):
        with tempfile.TemporaryDirectory() as folder:
            writer, process = self.make_writer(folder)
            with patch("person_tracking.video_writer.time.monotonic", side_effect=[0, 31]):
                with self.assertRaisesRegex(RuntimeError, "超时"):
                    writer.write(np.zeros((2, 6, 3), np.uint8))
            process.kill.assert_called_once()

    def test_release_closes_stdin_before_wait_and_verifies_after_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            writer, process = self.make_writer(folder)
            writer.frames = 4

            def wait(timeout):
                process.stdin.close.assert_called()
                process.poll.return_value = 0
                return 0

            process.wait.side_effect = wait
            with patch.object(writer, "_verify_output") as verify:
                writer.release()
                writer.release()
                verify.assert_called_once()
            process.kill.assert_not_called()

    def test_nonzero_exit_and_drain_timeout_are_errors(self):
        for failure in (7, subprocess.TimeoutExpired("gst", 30)):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                writer, process = self.make_writer(folder)
                process.wait.side_effect = [failure, 0]
                with self.assertRaisesRegex(RuntimeError, "编码失败|收尾超时"):
                    writer.release()
                self.assertFalse(writer.isOpened())

    def test_metadata_mismatch_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as folder:
            writer, process = self.make_writer(folder)
            writer.frames = 2
            process.poll.return_value = 0
            capture = Mock()
            capture.isOpened.return_value = True
            capture.get.side_effect = [1, 30000 / 1001, 6, 2]
            with patch("person_tracking.video_writer.cv2.VideoCapture", return_value=capture):
                with self.assertRaisesRegex(RuntimeError, "元数据校验失败"):
                    writer.release()
            capture.release.assert_called_once()

    def test_auto_uses_opencv_off_jetson_and_falls_back_only_during_probe(self):
        args = (Path("video.mp4"), 25, 320, 180)
        with patch("person_tracking.video_writer.is_jetson", return_value=False), \
                patch("person_tracking.video_writer.check_gstreamer") as probe, \
                patch("person_tracking.video_writer.cv2.VideoWriter") as cv_writer:
            self.assertIs(create_video_writer(*args, "auto", "mp4v", 8000000), cv_writer.return_value)
            probe.assert_not_called()
        with patch("person_tracking.video_writer.is_jetson", return_value=True), \
                patch("person_tracking.video_writer.check_gstreamer", side_effect=RuntimeError("missing")), \
                patch("person_tracking.video_writer.cv2.VideoWriter") as cv_writer:
            self.assertIs(create_video_writer(*args, "auto", "mp4v", 8000000), cv_writer.return_value)
            with self.assertRaisesRegex(RuntimeError, "missing"):
                create_video_writer(*args, "gstreamer", "mp4v", 8000000)
        with patch("person_tracking.video_writer.is_jetson", return_value=True), \
                patch("person_tracking.video_writer.check_gstreamer", return_value="gst"), \
                patch("person_tracking.video_writer.GStreamerVideoWriter", side_effect=OSError("launch failed")), \
                patch("person_tracking.video_writer.cv2.VideoWriter") as cv_writer:
            with self.assertRaisesRegex(OSError, "launch failed"):
                create_video_writer(*args, "auto", "mp4v", 8000000)
            cv_writer.assert_not_called()

    def test_cli_and_pipeline_keep_fractional_fps(self):
        args = build_parser().parse_args(["--input", "in.mp4", "--encoder", "gstreamer",
                                         "--video-bitrate", "12000000"])
        config = build_config(args)
        self.assertEqual((config.encoder, config.video_bitrate), ("gstreamer", 12000000))
        command = pipeline_command("gst", Path("a space.mp4"), 30000 / 1001, 1278, 718, 12000000)
        self.assertIn("framerate=30000/1001", command)
        self.assertIn("format=bgrx", command)


@unittest.skipUnless(os.environ.get("RUN_JETSON_GSTREAMER_TEST") == "1",
                     "Requires Jetson NVENC; set RUN_JETSON_GSTREAMER_TEST=1")
class JetsonEncodingIntegrationTests(unittest.TestCase):
    def test_real_pipe_eos_frame_order_fps_and_dimensions(self):
        # Includes a path with spaces/non-ASCII and an even, non-/4 width.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "硬件 output.mp4"
            fps = float(Fraction(30000, 1001))
            writer = GStreamerVideoWriter(check_gstreamer(), path, fps, 318, 180, 8000000)
            try:
                for value in (20, 60, 100, 140, 180, 220):
                    writer.write(np.full((180, 318, 3), value, np.uint8))
                writer.release()
            finally:
                writer.abort()
            capture = cv2.VideoCapture(str(path))
            try:
                self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), fps, places=3)
                decoded = []
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    self.assertEqual(frame.shape, (180, 318, 3))
                    decoded.append(frame.mean())
                np.testing.assert_allclose(decoded, [20, 60, 100, 140, 180, 220], atol=10)
            finally:
                capture.release()


if __name__ == "__main__":
    unittest.main()
