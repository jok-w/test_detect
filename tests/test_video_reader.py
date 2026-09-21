import json
import mmap
import os
import socket
import struct
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from person_tracking.__main__ import build_config, build_parser
from person_tracking.gst_decoder_worker import copy_bgrx, send_message
from person_tracking.processor import PersonVideoProcessor
from person_tracking.video_reader import (
    GStreamerVideoCapture, PreparedVideoCapture, create_video_capture, decoder_parser, receive_message,
)


class ReaderTests(unittest.TestCase):
    def test_row_padding_and_offset_are_removed_without_touching_pixels(self):
        pixels = bytes(range(48))
        padded = b'prefix!!' + pixels[:24] + b'PAD!' + pixels[24:] + b'PAD!'
        target = bytearray(48)
        copy_bgrx(padded, target, 6, 2, 28, 8)
        self.assertEqual(target, pixels)
        copy_bgrx(pixels, target, 6, 2, 24)
        self.assertEqual(target, pixels)
        with self.assertRaisesRegex(RuntimeError, "布局"):
            copy_bgrx(pixels[:20], target, 6, 2, 24)

    def test_fragmented_control_messages_and_abnormal_eof(self):
        data = json.dumps({"event": "frame", "pts_ns": 123456789}).encode()
        wire = struct.pack('!I', len(data)) + data
        connection = Mock()
        connection.recv.side_effect = [bytes([b]) for b in wire]
        self.assertEqual(receive_message(connection)["pts_ns"], 123456789)
        connection.recv.side_effect = [b'\x00', b'']
        with self.assertRaisesRegex(RuntimeError, "正常 EOS"):
            receive_message(connection)
        connection.recv.side_effect = [struct.pack('!I', 20000)]
        with self.assertRaisesRegex(RuntimeError, "长度"):
            receive_message(connection)

    def make_reader(self, expected=3):
        # Real mmap + socket transport, independent of Jetson/GI and subprocess inheritance.
        reader = object.__new__(GStreamerVideoCapture)
        reader.fps, reader.width, reader.height = 25, 6, 2
        reader.expected_frames, reader.frames = expected, 0
        reader.pts_ms, reader._first_pts_ns = float('nan'), None
        reader.missing_pts = 0
        reader.pull_ms = reader.copy_ms = reader.convert_ms = 0.0
        reader._closed = reader._eos = False
        reader._pending_frame = None
        reader._process = Mock()
        reader._process.poll.return_value = 0
        reader._process.wait.return_value = 0
        reader._shared = mmap.mmap(-1, 48)
        reader._memory = None
        reader._log = tempfile.TemporaryFile()
        reader._control, remote = socket.socketpair()
        reader._control.settimeout(1)
        remote.settimeout(1)
        return reader, remote

    def test_shared_slot_preserves_owned_frames_vfr_pts_and_eos(self):
        for read_ahead in (0, 1, 2):
            with self.subTest(read_ahead=read_ahead):
                self.check_primed_frames(read_ahead)

    def check_primed_frames(self, read_ahead):
        reader, remote = self.make_reader()
        capture = reader
        failures = []

        def producer():
            try:
                for index, pts in enumerate([2_000_000_000, 2_040_000_000, 2_110_000_000]):
                    self.assertEqual(remote.recv(1), b'N')
                    reader._shared[:] = bytes([index * 60, 50, 100, 255]) * 12
                    send_message(remote, {"event": "frame", "index": index,
                        "width": 6, "height": 2, "pts_ns": pts, "pull_ms": 1, "copy_ms": 2})
                self.assertEqual(remote.recv(1), b'N')
                send_message(remote, {"event": "eos", "frames": 3})
            except BaseException as error:
                failures.append(error)
            finally:
                remote.close()

        worker = threading.Thread(target=producer)
        worker.start()
        try:
            reader.prime()
            reader.prime()  # Rechecking startup must not advance the stream.
            self.assertEqual(reader.get(cv2.CAP_PROP_POS_FRAMES), 0)
            self.assertTrue(np.isnan(reader.get(cv2.CAP_PROP_POS_MSEC)))
            if read_ahead:
                capture = PreparedVideoCapture(reader, read_ahead)
            frames, timestamps = [], []
            for _ in range(3):
                ok, frame = capture.read()
                self.assertTrue(ok)
                frames.append(frame)
                timestamps.append(capture.get(cv2.CAP_PROP_POS_MSEC))
                self.assertEqual(capture.get(cv2.CAP_PROP_POS_FRAMES), len(frames))
                self.assertEqual(capture.pull_ms, len(frames))
                self.assertEqual(capture.copy_ms, len(frames) * 2)
            self.assertEqual(timestamps, [0, 40, 110])
            for index, frame in enumerate(frames):
                np.testing.assert_array_equal(frame[0, 0], [index * 60, 50, 100])
                self.assertTrue(frame.flags.c_contiguous)
            self.assertEqual(capture.read(), (False, None))
            self.assertEqual(capture.read(), (False, None))
        finally:
            worker.join(timeout=2)
            capture.release()
        self.assertFalse(worker.is_alive())
        self.assertFalse(failures, failures)

    def test_truncated_video_and_worker_error_are_not_normal_eof(self):
        for message in ({"event": "eos", "frames": 0},
                        {"event": "error", "message": "decoder failed"},
                        {"event": "frame", "index": 2, "width": 6, "height": 2}):
            reader, remote = self.make_reader()
            try:
                send_message(remote, message)
                with self.assertRaisesRegex(RuntimeError, "硬件视频读取失败"):
                    reader.read()
                self.assertFalse(reader.isOpened())
            finally:
                remote.close()
                reader.release()

    def test_missing_pts_uses_existing_monotonic_timestamp_fallback(self):
        reader, remote = self.make_reader(expected=1)
        try:
            reader._shared[:] = bytes(48)
            send_message(remote, {"event": "frame", "index": 0, "width": 6, "height": 2,
                                  "pts_ns": None, "pull_ms": 0, "copy_ms": 0})
            reader.prime()
            self.assertTrue(reader.read()[0])
            self.assertEqual(PersonVideoProcessor._frame_timestamp_ms(reader, 0, 25, -1), 0)
            self.assertEqual(reader.missing_pts, 1)
        finally:
            remote.close()
            reader.release()

    def test_decoder_timeout_closes_resources(self):
        reader, remote = self.make_reader()
        reader._control.settimeout(0.01)
        try:
            with self.assertRaisesRegex(RuntimeError, "硬件视频读取失败"):
                reader.read()
            self.assertFalse(reader.isOpened())
        finally:
            remote.close()
            reader.release()

    def test_auto_fallback_and_forced_decoder(self):
        capture = Mock()
        with patch('person_tracking.video_reader.cv2.VideoCapture', return_value=capture), \
                patch('person_tracking.video_reader.is_jetson', return_value=False), \
                patch('person_tracking.video_reader.check_decoder') as probe:
            self.assertIs(create_video_capture(Path('in.mp4')), capture)
            probe.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, 'Jetson'):
                create_video_capture(Path('in.mp4'), 'gstreamer')
        self.assertEqual(decoder_parser(Path('in.mp4'), 'hevc'), 'h265parse')
        self.assertEqual(decoder_parser(Path('in.mov'), 'avc1'), 'h264parse')
        with self.assertRaises(RuntimeError):
            decoder_parser(Path('in.avi'), 'MJPG')
        capture.get.side_effect = lambda p: {
            cv2.CAP_PROP_FPS: 25, 3: 320, 4: 180, cv2.CAP_PROP_FRAME_COUNT: 3,
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*'hevc'),
        }[p]
        with patch('person_tracking.video_reader.cv2.VideoCapture', return_value=capture), \
                patch('person_tracking.video_reader.is_jetson', return_value=True), \
                patch('person_tracking.video_reader.check_decoder', side_effect=RuntimeError('GI missing')):
            self.assertIs(create_video_capture(Path('in.mp4')), capture)
            with self.assertRaisesRegex(RuntimeError, 'GI missing'):
                create_video_capture(Path('in.mp4'), 'gstreamer')

    def test_cli_exposes_independent_decoder_selection(self):
        args = build_parser().parse_args(['--input', 'in.mp4', '--decoder', 'gstreamer',
                                         '--decode-prefetch', '1', '--encoder', 'opencv'])
        config = build_config(args)
        self.assertEqual((config.decoder, config.decode_prefetch, config.encoder), ('gstreamer', 1, 'opencv'))

    def test_mpeg4_aliases_and_container_limit(self):
        for codec in ('FMP4', 'mp4v', 'DIVX', 'DX50', 'XVID', 'fmp4'):
            for suffix in ('.mp4', '.MOV'):
                with self.subTest(codec=codec, suffix=suffix):
                    self.assertEqual(decoder_parser(Path('in' + suffix), codec), 'mpeg4videoparse')
        for path, codec in (('in.avi', 'FMP4'), ('in.mp4', 'MJPG'), ('in.mp4', 'DIV3')):
            with self.assertRaises(RuntimeError):
                decoder_parser(Path(path), codec)

    def test_worker_accepts_mpeg4_parser(self):
        from person_tracking import gst_decoder_worker as worker
        gst = Mock()
        with patch('sys.argv', ['worker', '--check', '--parser', 'mpeg4videoparse']), \
                patch.object(worker, 'load_gst', return_value=(gst, Mock())) as load, \
                patch('builtins.print'):
            self.assertEqual(worker.main(), 0)
            load.assert_called_once_with('mpeg4videoparse')

    def metadata_capture(self):
        capture = Mock()
        capture.get.side_effect = lambda p: {
            cv2.CAP_PROP_FPS: 25, 3: 320, 4: 180, cv2.CAP_PROP_FRAME_COUNT: 3,
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*'FMP4'),
        }[p]
        return capture

    def test_startup_failures_fallback_only_in_auto_and_release_resources(self):
        for stage in ('construct', 'prime'):
            for decoder in ('auto', 'gstreamer'):
                for error in (RuntimeError('startup failed'), OSError('startup failed')):
                    with self.subTest(stage=stage, decoder=decoder, error=type(error)):
                        capture, reader = self.metadata_capture(), Mock()
                        if stage == 'prime':
                            reader.prime.side_effect = error
                        with patch('person_tracking.video_reader.cv2.VideoCapture', return_value=capture), \
                                patch('person_tracking.video_reader.is_jetson', return_value=True), \
                                patch('person_tracking.video_reader.check_decoder'), \
                                patch('person_tracking.video_reader.GStreamerVideoCapture',
                                      return_value=reader,
                                      side_effect=error if stage == 'construct' else None):
                            if decoder == 'auto':
                                self.assertIs(create_video_capture(Path('in.mp4'), decoder), capture)
                                capture.release.assert_not_called()
                            else:
                                with self.assertRaisesRegex(type(error), 'startup failed'):
                                    create_video_capture(Path('in.mp4'), decoder)
                                capture.release.assert_called_once()
                            capture.read.assert_not_called()
                            if stage == 'prime':
                                reader.release.assert_called_once()

    def test_successful_startup_commits_to_nvdec_and_later_errors_propagate(self):
        capture, reader = self.metadata_capture(), Mock()
        with patch('person_tracking.video_reader.cv2.VideoCapture', return_value=capture), \
                patch('person_tracking.video_reader.is_jetson', return_value=True), \
                patch('person_tracking.video_reader.check_decoder') as check, \
                patch('person_tracking.video_reader.GStreamerVideoCapture', return_value=reader) as factory:
            result = create_video_capture(Path('in.mp4'), 'auto', read_ahead=0)
            self.assertIs(result, reader)
            check.assert_called_once_with('/usr/bin/python3', 'mpeg4videoparse')
            self.assertEqual(factory.call_args.args[5], 'mpeg4videoparse')
            reader.prime.assert_called_once()
            capture.release.assert_called_once()
            reader.read.side_effect = RuntimeError('midstream error')
            with self.assertRaisesRegex(RuntimeError, 'midstream error'):
                result.read()
            capture.read.assert_not_called()

    def test_prime_rejects_empty_video(self):
        reader, remote = self.make_reader(expected=0)
        try:
            send_message(remote, {'event': 'eos', 'frames': 0})
            with self.assertRaisesRegex(RuntimeError, '未产生首帧'):
                reader.prime()
        finally:
            remote.close()
            reader.release()


@unittest.skipUnless(os.environ.get('RUN_JETSON_GSTREAMER_TEST') == '1', 'Requires Jetson NVDEC/NVENC')
class JetsonReaderIntegrationTests(unittest.TestCase):
    def test_real_mpeg4_decode_preserves_frames_pts_and_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'MPEG4 原始 video.mp4'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 25, (320, 180))
            try:
                self.assertTrue(writer.isOpened())
                for value in (30, 60, 90, 120, 150, 180):
                    writer.write(np.full((180, 320, 3), value, np.uint8))
            finally:
                writer.release()
            for read_ahead in (0, 1, 2):
                with self.subTest(read_ahead=read_ahead):
                    reader = create_video_capture(path, 'gstreamer', read_ahead=read_ahead)
                    try:
                        pts, means = [], []
                        while True:
                            ok, frame = reader.read()
                            if not ok:
                                break
                            self.assertEqual(frame.shape, (180, 320, 3))
                            pts.append(reader.get(cv2.CAP_PROP_POS_MSEC))
                            means.append(frame.mean())
                        np.testing.assert_allclose(pts, [0, 40, 80, 120, 160, 200], atol=1)
                        self.assertTrue(all(b > a + 15 for a, b in zip(means, means[1:])))
                    finally:
                        reader.release()

    def test_real_h264_decode_keeps_nonuniform_pts_frame_order_and_early_close(self):
        # The system helper's input has intentionally nonuniform PTS to test VFR handling.
        create_video = '''
import sys, gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)
p = Gst.parse_launch('appsrc name=source format=time ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! nvv4l2h264enc bitrate=8000000 ! h264parse ! qtmux ! filesink name=out')
p.get_by_name('out').set_property('location', sys.argv[1])
s = p.get_by_name('source')
s.set_property('caps', Gst.Caps.from_string('video/x-raw,format=I420,width=320,height=180,framerate=25/1'))
p.set_state(Gst.State.PLAYING)
try:
    times = [0, 40, 100, 140, 260, 300, 340]
    for i in range(6):
        payload = bytes([30+i*30]) * (320*180) + bytes([128]) * (320*180//2)
        b = Gst.Buffer.new_wrapped(payload)
        b.pts = times[i] * Gst.MSECOND
        b.duration = (times[i+1]-times[i]) * Gst.MSECOND
        assert s.emit('push-buffer', b) == Gst.FlowReturn.OK
    s.emit('end-of-stream')
    msg = p.get_bus().timed_pop_filtered(30*Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
    assert msg is not None, 'Encoder timeout'
    if msg.type == Gst.MessageType.ERROR:
        raise RuntimeError(str(msg.parse_error()))
finally:
    p.set_state(Gst.State.NULL)
'''
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / '原始 video.mp4'
            subprocess.run(['/usr/bin/python3', '-I', '-c', create_video, str(path)], check=True, timeout=45)
            for read_ahead in (0, 1, 2):
                reader = create_video_capture(path, 'gstreamer', read_ahead=read_ahead)
                try:
                    pts, means = [], []
                    while True:
                        ok, frame = reader.read()
                        if not ok:
                            break
                        self.assertEqual(frame.shape, (180, 320, 3))
                        pts.append(reader.get(cv2.CAP_PROP_POS_MSEC))
                        means.append(frame.mean())
                    np.testing.assert_allclose(pts, [0, 40, 100, 140, 260, 300], atol=1)
                    self.assertTrue(all(b > a + 15 for a, b in zip(means, means[1:])))
                finally:
                    reader.release()
            reader = create_video_capture(path, 'gstreamer')
            self.assertTrue(reader.read()[0])
            reader.release()
            self.assertIsNotNone(reader.source._process.poll())
            self.assertFalse(reader._thread.is_alive())


if __name__ == '__main__':
    unittest.main()
