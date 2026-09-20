"""Jetson NVENC output through a bounded stdin pipe; no OpenCV GStreamer required."""
from __future__ import annotations

import logging
import math
import os
import platform
import select
import shutil
import subprocess
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)
GST_ELEMENTS = ("fdsrc", "rawvideoparse", "nvvidconv", "nvv4l2h264enc",
                "h264parse", "qtmux", "filesink")


def is_jetson() -> bool:
    return platform.system() == "Linux" and Path("/etc/nv_tegra_release").is_file()


def check_gstreamer() -> str:
    if platform.system() != "Linux":
        raise RuntimeError("GStreamer NVENC 输出仅支持 Jetson Linux")
    launch = shutil.which("gst-launch-1.0")
    inspect = shutil.which("gst-inspect-1.0")
    if not launch or not inspect:
        raise RuntimeError("缺少 gst-launch-1.0 或 gst-inspect-1.0")
    for element in GST_ELEMENTS:
        try:
            result = subprocess.run([inspect, element], capture_output=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"检查 GStreamer 插件 {element} 失败：{error}") from error
        if result.returncode:
            raise RuntimeError(f"缺少可用 GStreamer 插件：{element}")
    return launch


def pipeline_command(launch: str, path: Path, fps: float, width: int,
                     height: int, bitrate: int) -> list[str]:
    rate = Fraction(fps).limit_denominator(100000)
    # Gst parses property strings itself, even though no shell is involved.
    location = str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"')
    return [launch, "-q", "-e", "fdsrc", "fd=0", "blocksize=65536", "!",
            "rawvideoparse", "format=bgrx", f"width={width}", f"height={height}",
            f"framerate={rate.numerator}/{rate.denominator}", "!",
            "nvvidconv", "!", "video/x-raw(memory:NVMM),format=NV12", "!",
            "nvv4l2h264enc", f"bitrate={bitrate}", "!", "h264parse", "!",
            "qtmux", "!", "filesink", f'location="{location}"', "sync=false"]


class GStreamerVideoWriter:
    """Submit every frame in order, drain EOS, then check the completed MP4 metadata."""

    def __init__(self, launch: str, path: Path, fps: float, width: int,
                 height: int, bitrate: int, timeout: float = 30.0) -> None:
        if not math.isfinite(fps) or fps <= 0 or min(width, height) < 2:
            raise ValueError("无效的输出帧率或尺寸")
        if width % 2 or height % 2 or bitrate <= 0:
            raise ValueError("硬件编码需要偶数宽高和正码率")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("编码器超时必须为有限正数")
        self.path, self.fps = path, fps
        self.width, self.height, self.timeout = width, height, timeout
        self.frames = 0
        self._closed = False
        self._last_details = ""
        # A regular file cannot deadlock on verbose encoder stderr like a PIPE can.
        self._log = tempfile.TemporaryFile()
        self._process = None
        try:
            self._process = subprocess.Popen(
                pipeline_command(launch, path, fps, width, height, bitrate),
                stdin=subprocess.PIPE, stdout=self._log, stderr=subprocess.STDOUT,
                bufsize=0,
            )
            os.set_blocking(self._process.stdin.fileno(), False)
        except BaseException:
            self.abort()
            raise

    def _details(self) -> str:
        if self._log.closed:
            return self._last_details
        self._log.seek(0, os.SEEK_END)
        self._log.seek(max(0, self._log.tell() - 4096))
        self._last_details = self._log.read().decode("utf-8", errors="replace").strip()
        return self._last_details

    def isOpened(self) -> bool:
        return not self._closed and self._process.poll() is None

    def write(self, frame: np.ndarray) -> None:
        if not self.isOpened():
            raise RuntimeError(f"GStreamer 编码进程已退出：{self._details()}")
        if frame.dtype != np.uint8 or frame.shape != (self.height, self.width, 3):
            raise ValueError("硬件编码输入必须为指定尺寸的 uint8 BGR 图像")
        # BGRx has an unambiguous width*4 row stride (including widths not /4).
        # nvvidconv accepts BGRx directly and performs RGB -> NV12 on Jetson.
        data = memoryview(cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)).cast("B")
        fd = self._process.stdin.fileno()
        deadline = time.monotonic() + self.timeout
        try:
            while data:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("向 GStreamer 提交视频帧超时")
                if self._process.poll() is not None:
                    raise BrokenPipeError("GStreamer 编码进程提前退出")
                if not select.select([], [fd], [], min(remaining, 0.5))[1]:
                    continue
                try:
                    written = os.write(fd, data[:65536])
                except BlockingIOError:
                    continue
                if written <= 0:
                    raise BrokenPipeError("GStreamer 管道未接受数据")
                data = data[written:]
            self.frames += 1
        except (OSError, TimeoutError) as error:
            details = self._details()
            self.abort()
            raise RuntimeError(f"硬件编码写入失败：{error}\n{details}") from error

    def release(self) -> None:
        if self._closed:
            return
        try:
            # EOF causes fdsrc EOS: the encoder drains and qtmux writes the MP4 index.
            self._process.stdin.close()
            try:
                code = self._process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"GStreamer 编码器收尾超时：{self._details()}") from error
            if code != 0:
                raise RuntimeError(f"GStreamer 编码失败，退出码={code}：{self._details()}")
            if self.frames:
                self._verify_output()
        finally:
            self.abort()

    def _verify_output(self) -> None:
        capture = cv2.VideoCapture(str(self.path))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"硬件编码输出无法打开：{self.path}")
            count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = capture.get(cv2.CAP_PROP_FPS)
            width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            values = (count, fps, width, height)
            if (not all(math.isfinite(v) for v in values)
                    or abs(count - self.frames) > 0.1
                    or (width, height) != (self.width, self.height)
                    or not math.isclose(fps, self.fps, rel_tol=1e-4, abs_tol=1e-3)):
                raise RuntimeError(
                    f"硬件输出元数据校验失败：提交={self.frames} 帧，文件={count} 帧，"
                    f"尺寸={width}x{height}，fps={fps}，预期={self.width}x{self.height}@{self.fps}")
            logger.info("硬件输出元数据校验通过：帧数=%s，尺寸=%sx%s，fps=%.6f",
                        self.frames, self.width, self.height, fps)
        finally:
            capture.release()

    def abort(self) -> None:
        """Bounded cleanup on failure; never hide the original processing exception."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._process is not None:
                if self._process.poll() is None:
                    self._process.kill()
                self._process.wait(timeout=5)
                if self._process.stdin is not None:
                    self._process.stdin.close()
        except (OSError, subprocess.TimeoutExpired):
            logger.exception("清理 GStreamer 编码进程失败")
        finally:
            self._log.close()


def create_video_writer(path: Path, fps: float, width: int, height: int,
                        encoder: str, codec: str, bitrate: int):
    use_gst = encoder == "gstreamer" or (encoder == "auto" and is_jetson()
                                         and path.suffix.lower() == ".mp4")
    if use_gst:
        try:
            launch = check_gstreamer()
        except RuntimeError as error:
            if encoder == "gstreamer":
                raise
            logger.warning("硬件输出不可用，回退 OpenCV：%s", error)
        else:
            writer = GStreamerVideoWriter(launch, path, fps, width, height, bitrate)
            logger.info("视频输出：GStreamer / nvv4l2h264enc（NVENC），H.264，码率=%s bps", bitrate)
            return writer
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"无法创建输出视频，请检查编码 {codec} 和路径：{path}")
    logger.info("视频输出：OpenCV，编码=%s", codec)
    return writer
