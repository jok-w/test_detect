import queue
import threading
import unittest

import cv2
import numpy as np

from person_tracking.video_reader import PreparedVideoCapture


class FakeSource:
    def __init__(self, count=6, fail_at=None, block_at=None):
        self.count, self.fail_at, self.block_at = count, fail_at, block_at
        self.frames = self.missing_pts = 0
        self.pull_ms = self.copy_ms = self.convert_ms = 0.0
        self.calls = queue.Queue()
        self.cancelled = threading.Event()
        self.releases = 0
        self.inside_read = False

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: 25, cv2.CAP_PROP_FRAME_WIDTH: 6,
                cv2.CAP_PROP_FRAME_HEIGHT: 2, cv2.CAP_PROP_FRAME_COUNT: self.count,
                cv2.CAP_PROP_POS_MSEC: [0, 40, 110, 150, 260, 300][max(0, self.frames - 1)]}.get(prop, 0)

    def read(self):
        self.inside_read = True
        try:
            self.calls.put(self.frames)
            if self.frames == self.block_at:
                if not self.cancelled.wait(2):
                    raise RuntimeError('test cancellation timed out')
                raise RuntimeError('cancelled')
            if self.frames == self.fail_at:
                raise RuntimeError('broken source')
            if self.frames == self.count:
                return False, None
            frame = np.full((2, 6, 3), self.frames, np.uint8)
            self.frames += 1
            self.pull_ms += 1
            self.copy_ms += 2
            self.convert_ms += 3
            return True, frame
        finally:
            self.inside_read = False

    def cancel_pending_read(self):
        self.cancelled.set()

    def release(self):
        if self.inside_read:
            raise AssertionError('released while a read still owns memory')
        self.releases += 1


class PreparedReaderTests(unittest.TestCase):
    def test_preparation_overlaps_consumer_but_cannot_read_beyond_credit_limit(self):
        for capacity in (1, 2):
            with self.subTest(capacity=capacity):
                source = FakeSource()
                reader = PreparedVideoCapture(source, capacity)
                try:
                    # Lazy start keeps all preparation inside the measured processing interval.
                    self.assertEqual(source.frames, 0)
                    ok, first = reader.read()
                    self.assertTrue(ok)
                    self.assertEqual(source.calls.get(timeout=1), 0)
                    # Consumer has not asked for another frame, but next frames are being prepared.
                    for expected in range(1, capacity + 1):
                        self.assertEqual(source.calls.get(timeout=1), expected)
                    with self.assertRaises(queue.Empty):
                        source.calls.get(timeout=0.05)
                    self.assertEqual(reader.get(cv2.CAP_PROP_POS_MSEC), 0)
                    self.assertEqual(reader.get(cv2.CAP_PROP_POS_FRAMES), 1)
                    self.assertEqual((reader.pull_ms, reader.copy_ms, reader.convert_ms), (1, 2, 3))
                    self.assertFalse(first.any())
                finally:
                    reader.release()
                self.assertFalse(reader._thread.is_alive())
                self.assertEqual(source.releases, 1)

    def test_frames_pts_and_statistics_are_delivered_together_in_order(self):
        source = FakeSource()
        reader = PreparedVideoCapture(source)
        try:
            images, pts = [], []
            while True:
                ok, frame = reader.read()
                if not ok:
                    break
                images.append(frame)
                pts.append(reader.get(cv2.CAP_PROP_POS_MSEC))
            self.assertEqual(pts, [0, 40, 110, 150, 260, 300])
            for i, image in enumerate(images):
                self.assertTrue(np.all(image == i))
            self.assertEqual(reader.frames, 6)
            self.assertEqual(reader.copy_ms, 12)
            self.assertEqual(reader.read(), (False, None))
        finally:
            reader.release()
        self.assertEqual(source.releases, 1)

    def test_source_failure_is_delivered_after_preceding_frames_not_as_eos(self):
        reader = PreparedVideoCapture(FakeSource(fail_at=2))
        try:
            self.assertTrue(reader.read()[0])
            self.assertTrue(reader.read()[0])
            with self.assertRaisesRegex(RuntimeError, 'broken source'):
                reader.read()
            self.assertEqual(reader.frames, 2)
            self.assertFalse(reader.isOpened())
            self.assertFalse(reader._thread.is_alive())
        finally:
            reader.release()

    def test_early_stop_wakes_pending_read_before_releasing_memory(self):
        source = FakeSource(block_at=1)
        reader = PreparedVideoCapture(source)
        self.assertTrue(reader.read()[0])
        self.assertEqual(source.calls.get(timeout=1), 0)
        self.assertEqual(source.calls.get(timeout=1), 1)
        reader.release()
        self.assertTrue(source.cancelled.is_set())
        self.assertFalse(reader._thread.is_alive())
        self.assertEqual(source.releases, 1)
        self.assertTrue(reader._queue.empty())

    def test_close_before_first_read_and_repeated_close(self):
        source = FakeSource()
        reader = PreparedVideoCapture(source)
        reader.release()
        reader.release()
        self.assertEqual(source.frames, 0)
        self.assertEqual(source.releases, 1)
        with self.assertRaisesRegex(RuntimeError, '已关闭'):
            reader.read()


if __name__ == '__main__':
    unittest.main()
