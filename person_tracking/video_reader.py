"""OpenCV-compatible NVDEC reader with PTS and bounded appsink prefetch."""
from __future__ import annotations

import json
import logging
import math
import mmap
import queue
import socket
import struct
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .video_writer import is_jetson

logger = logging.getLogger(__name__)
WORKER = Path(__file__).with_name("gst_decoder_worker.py")


def decoder_parser(path: Path, codec: str) -> str:
    if path.suffix.lower() not in {".mp4", ".mov"}:
        raise RuntimeError("NVDEC 读取当前支持 MP4/MOV 中的 H.264/H.265")
    if codec.lower() in {"hevc", "hvc1", "hev1"}:
        return "h265parse"
    if codec.lower() in {"avc1", "h264", "x264"}:
        return "h264parse"
    raise RuntimeError(f"NVDEC 读取暂不支持输入编码：{codec!r}")


def check_decoder(python: str, parser: str) -> None:
    try:
        result = subprocess.run([python, "-I", str(WORKER), "--check", "--parser", parser],
                                capture_output=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"无法检查系统 Python GStreamer：{error}") from error
    if result.returncode:
        details = (result.stdout + result.stderr)[-4096:].decode("utf-8", errors="replace")
        raise RuntimeError(f"系统 Python 缺少可用的 GStreamer/GstVideo GI 或 NVDEC 插件：{details}")


def receive_message(control) -> dict:
    def read_exact(size):
        data = bytearray()
        while len(data) < size:
            chunk = control.recv(size - len(data))
            if not chunk:
                raise RuntimeError("解码进程异常关闭控制通道，未收到正常 EOS")
            data.extend(chunk)
        return data
    size, = struct.unpack("!I", read_exact(4))
    if not 0 < size <= 16384:
        raise RuntimeError("解码元数据长度无效")
    message = json.loads(read_exact(size))
    if not isinstance(message, dict):
        raise RuntimeError("解码元数据格式无效")
    if message.get("event") == "error":
        raise RuntimeError(message.get("message", "解码进程报错"))
    return message


class GStreamerVideoCapture:
    def __init__(self, path, fps, width, height, expected_frames, parser,
                 python="/usr/bin/python3", prefetch=2, timeout=30):
        if (not math.isfinite(fps) or fps <= 0 or min(width, height) < 2
                or prefetch not in (1, 2) or not math.isfinite(timeout) or timeout <= 5):
            raise ValueError("无效的硬件读取帧率、尺寸、预读或超时配置")
        self.fps, self.width, self.height = fps, width, height
        self.expected_frames, self.frames = expected_frames, 0
        self.pts_ms, self._first_pts_ns = float("nan"), None
        self.missing_pts = 0
        self.pull_ms = self.copy_ms = self.convert_ms = 0.0
        self._closed, self._eos = False, False
        self._process = self._shared = self._memory = self._control = self._log = None
        child = None
        try:
            # tmpfs, not the SSD: exactly one reusable BGRx slot, with request/response ownership.
            self._memory = tempfile.TemporaryFile(dir="/dev/shm")
            size = width * height * 4
            self._memory.truncate(size)
            self._shared = mmap.mmap(self._memory.fileno(), size)
            self._control, child = socket.socketpair()
            self._control.settimeout(timeout)
            self._log = tempfile.TemporaryFile()
            self._process = subprocess.Popen(
                [python, "-I", str(WORKER), "--input", str(path.resolve()),
                 "--parser", parser, "--width", str(width), "--height", str(height),
                 "--prefetch", str(prefetch), "--timeout", str(timeout - 5),
                 "--control-fd", str(child.fileno()), "--memory-fd", str(self._memory.fileno())],
                pass_fds=(child.fileno(), self._memory.fileno()), stdin=subprocess.DEVNULL,
                stdout=self._log, stderr=subprocess.STDOUT,
            )
            child.close()
            child = None
            if receive_message(self._control).get("event") != "ready":
                raise RuntimeError("硬件读取初始化协议错误")
        except BaseException:
            if child is not None:
                child.close()
            self.release()
            raise

    def isOpened(self):
        return not self._closed

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_WIDTH: self.width,
                cv2.CAP_PROP_FRAME_HEIGHT: self.height, cv2.CAP_PROP_FRAME_COUNT: self.expected_frames,
                cv2.CAP_PROP_POS_MSEC: self.pts_ms, cv2.CAP_PROP_POS_FRAMES: self.frames}.get(prop, 0)

    def read(self):
        if self._closed:
            raise RuntimeError("硬件读取器已关闭")
        if self._eos:
            return False, None
        try:
            self._control.sendall(b"N")
            message = receive_message(self._control)
            if message.get("event") == "eos":
                if message.get("frames") != self.frames:
                    raise RuntimeError("解码进程与主进程帧数不一致")
                if self.expected_frames > 0 and self.frames != self.expected_frames:
                    raise RuntimeError(f"解码帧数与输入元数据不一致：读取={self.frames}，预期={self.expected_frames}")
                code = self._process.wait(timeout=5)
                if code != 0:
                    raise RuntimeError(f"解码进程异常退出：{code}")
                self._eos = True
                return False, None
            if (message.get("event") != "frame" or message.get("index") != self.frames
                    or (message.get("width"), message.get("height")) != (self.width, self.height)):
                raise RuntimeError("解码帧序或尺寸不一致")
            # Copy into an independently owned contiguous BGR frame before NEXT can reuse the slot.
            started = time.perf_counter()
            view = np.ndarray((self.height, self.width, 4), np.uint8, buffer=self._shared)
            try:
                frame = cv2.cvtColor(view, cv2.COLOR_BGRA2BGR)
            finally:
                del view
            self.convert_ms += (time.perf_counter() - started) * 1000
            self.pull_ms += float(message["pull_ms"])
            self.copy_ms += float(message["copy_ms"])
            pts = message.get("pts_ns")
            if pts is None:
                self.pts_ms = float("nan")
                self.missing_pts += 1
            else:
                if not isinstance(pts, int) or pts < 0:
                    raise RuntimeError("解码时间戳格式无效")
                if self._first_pts_ns is None:
                    # Preserve spacing while aligning the first valid frame with the video origin.
                    self._first_pts_ns = pts - round(self.frames * 1e9 / self.fps)
                self.pts_ms = (pts - self._first_pts_ns) / 1e6
            self.frames += 1
            return True, frame
        except Exception as error:
            details = ""
            if self._log is not None:
                self._log.seek(0, 2)
                self._log.seek(max(0, self._log.tell() - 4096))
                details = self._log.read().decode("utf-8", errors="replace")
            self.release()
            raise RuntimeError(f"硬件视频读取失败：{error}\n{details}") from error

    def release(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._process is not None:
                if self._process.poll() is None:
                    try:
                        self._control.sendall(b"Q")
                        self._process.wait(timeout=2)
                    except (OSError, subprocess.TimeoutExpired):
                        self._process.kill()
                        self._process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            logger.exception("清理解码辅助进程失败")
        finally:
            for resource in (self._control, self._shared, self._memory, self._log):
                if resource is not None:
                    resource.close()

    def cancel_pending_read(self):
        """Wake the sole reading thread without closing its in-use mmap."""
        if self._control is not None:
            try:
                self._control.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def log_statistics(self):
        if self.frames:
            logger.info("NVDEC 读取统计：帧数=%s，等待解码样本=%.2fms，共享内存复制=%.2fms，"
                        "BGR 转换=%.2fms，缺失 PTS 帧数=%s（均为每帧平均，等待样本不是纯解码耗时）",
                        self.frames, self.pull_ms / self.frames, self.copy_ms / self.frames,
                        self.convert_ms / self.frames, self.missing_pts)


class PreparedVideoCapture:
    """Prepare independently owned BGR frames ahead of processing, with bounded credits.

One producer owns source.read/release and the shared slot. Queue + in-flight read
never exceeds capacity; each packet carries the PTS/statistics for that exact frame.
"""

    def __init__(self, source: GStreamerVideoCapture, capacity: int = 2):
        if capacity not in (1, 2):
            raise ValueError("完整帧预读必须为 1 或 2 帧")
        self.source = source
        self.capacity = capacity
        self._metadata = {prop: source.get(prop) for prop in (
            cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_COUNT)}
        self._queue = queue.Queue(maxsize=capacity)
        self._credits = threading.BoundedSemaphore(capacity)
        self._stop = threading.Event()
        self._thread = None
        self._closed = self._eos = False
        self.frames = self.missing_pts = 0
        self.pts_ms = float("nan")
        self.pull_ms = self.copy_ms = self.convert_ms = self.prepare_ms = self.wait_ms = 0.0

    def _produce(self):
        prepare_ms = 0.0
        try:
            while not self._stop.is_set():
                if not self._credits.acquire(timeout=0.1):
                    continue
                if self._stop.is_set():
                    break
                started = time.perf_counter()
                try:
                    ok, frame = self.source.read()
                    prepare_ms += (time.perf_counter() - started) * 1000
                    if ok:
                        packet = ("frame", frame, self.source.get(cv2.CAP_PROP_POS_MSEC),
                                  (self.source.pull_ms, self.source.copy_ms,
                                   self.source.convert_ms, self.source.missing_pts, prepare_ms))
                    else:
                        packet = ("eos",)
                except Exception as error:
                    packet = ("error", error)
                if self._stop.is_set():
                    break
                # A credit was reserved before allocating/preparing the frame.
                self._queue.put_nowait(packet)
                if packet[0] != "frame":
                    break
                # Do not retain the previous image while preparing another one.
                del packet, frame
        finally:
            self.source.release()

    def isOpened(self):
        return not self._closed

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.pts_ms
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self.frames
        return self._metadata.get(prop, 0)

    def read(self):
        if self._closed:
            raise RuntimeError("完整帧预读器已关闭")
        if self._eos:
            return False, None
        started = time.perf_counter()
        if self._thread is None:
            self._thread = threading.Thread(target=self._produce, name="nvdec-frame-preparation", daemon=True)
            self._thread.start()
        try:
            deadline = time.monotonic() + 40
            while True:
                try:
                    packet = self._queue.get(timeout=0.1)
                    break
                except queue.Empty:
                    if not self._thread.is_alive():
                        raise RuntimeError("完整帧预读线程异常结束，未收到 EOS")
                    if time.monotonic() >= deadline:
                        raise RuntimeError("等待完整 BGR 帧超时")
            self._credits.release()
            if packet[0] == "error":
                raise RuntimeError(f"完整帧预读失败：{packet[1]}") from packet[1]
            if packet[0] == "eos":
                self._eos = True
                return False, None
            self.pts_ms = packet[2]
            (self.pull_ms, self.copy_ms, self.convert_ms,
             self.missing_pts, self.prepare_ms) = packet[3]
            self.frames += 1
            self.wait_ms += (time.perf_counter() - started) * 1000
            return True, packet[1]
        except BaseException:
            self.release()
            raise

    def release(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._thread is None:
            self.source.release()
        else:
            if self._thread.is_alive():
                self.source.cancel_pending_read()
                self._thread.join(timeout=10)
            if self._thread.is_alive():
                # Do not unmap memory that a native conversion could still be reading.
                raise RuntimeError("完整帧预读线程未能及时退出")
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def log_statistics(self):
        if self.frames:
            logger.info("NVDEC 完整帧预读：消费帧数=%s，上限=%s，主线程取帧等待=%.2fms，"
                        "后台准备=%.2fms，等待解码样本=%.2fms，共享内存复制=%.2fms，"
                        "BGR 转换=%.2fms，缺失 PTS 帧数=%s（每消费帧平均，后台阶段与检测重叠，不可相加）",
                        self.frames, self.capacity, self.wait_ms / self.frames,
                        self.prepare_ms / self.frames, self.pull_ms / self.frames,
                        self.copy_ms / self.frames, self.convert_ms / self.frames, self.missing_pts)


def create_video_capture(path: Path, decoder="auto", prefetch=2, python="/usr/bin/python3", read_ahead=2):
    probe = cv2.VideoCapture(str(path))
    if not probe.isOpened():
        probe.release()
        raise RuntimeError(f"无法打开输入视频：{path}")
    if decoder == "opencv" or (decoder == "auto" and not is_jetson()):
        logger.info("视频读取：OpenCV")
        return probe
    try:
        if not is_jetson():
            raise RuntimeError("NVDEC 读取需要 Jetson Linux")
        fps = probe.get(cv2.CAP_PROP_FPS)
        width, height = int(probe.get(3)), int(probe.get(4))
        expected_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(probe.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * i)) & 255) for i in range(4))
        parser = decoder_parser(path, codec)
        check_decoder(python, parser)
    except (RuntimeError, ValueError, OverflowError) as error:
        if decoder != "auto":
            probe.release()
            raise
        logger.warning("硬件读取不可用，回退 OpenCV：%s", error)
        return probe
    probe.release()
    reader = GStreamerVideoCapture(path, fps, width, height, expected_frames, parser, python, prefetch)
    logger.info("视频读取：GStreamer / nvv4l2decoder（NVDEC），原图=%sx%s，预读上限=%s 帧，保留 PTS",
                width, height, prefetch)
    if read_ahead:
        try:
            prepared = PreparedVideoCapture(reader, read_ahead)
        except BaseException:
            reader.release()
            raise
        logger.info("NVDEC 完整帧预读已启用：最多 %s 帧（含准备中），复制与 BGR 转换在后台执行", read_ahead)
        return prepared
    logger.info("NVDEC 完整帧预读关闭：同步准备 BGR，使用上一版路径")
    return reader
