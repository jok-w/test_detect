from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .engine import FrameTrackingResult, PersonTrackingEngine
from .tracking_config import PersonTrackingConfig
from .video_writer import GStreamerVideoWriter, create_video_writer
from .video_reader import GStreamerVideoCapture, create_video_capture

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ProcessorConfig(PersonTrackingConfig):
    """在共用识别配置上补充离线视频输入、输出和显示选项。"""

    input_path: Path
    output_path: Path
    codec: str = "mp4v"
    encoder: str = "auto"
    video_bitrate: int = 8000000
    decoder: str = "auto"
    decode_prefetch: int = 2
    gst_python: str = "/usr/bin/python3"
    output_max_width: int = 1920
    display: bool = True
    display_window_name: str = "人物检测与卡尔曼跟踪"

    def validate(self) -> None:
        """
        作用：校验识别配置及离线视频输入、输出和编码参数。
        返回：无。
        异常：文件缺失或配置无效时抛出 ValueError。
        """
        super().validate()
        if not self.input_path.is_file():
            raise ValueError(f"输入视频不存在：{self.input_path}")
        if self.input_path.resolve() == self.output_path.resolve():
            raise ValueError("输出视频不能覆盖输入视频")
        if len(self.codec) != 4:
            raise ValueError("输出视频编码必须是四字符编码")
        if self.encoder not in {"auto", "opencv", "gstreamer"}:
            raise ValueError("输出编码后端必须为 auto、opencv 或 gstreamer")
        if self.decoder not in {"auto", "opencv", "gstreamer"}:
            raise ValueError("读取后端必须为 auto、opencv 或 gstreamer")
        if self.decode_prefetch not in (1, 2):
            raise ValueError("解码预读上限必须为 1 或 2 帧")
        if not self.gst_python.strip():
            raise ValueError("系统 Python 路径不能为空")
        if self.video_bitrate <= 0 or self.video_bitrate > 4294967295:
            raise ValueError("视频码率必须在 1 到 4294967295 bps 之间")
        if self.encoder == "gstreamer" and self.output_path.suffix.lower() != ".mp4":
            raise ValueError("GStreamer 硬件输出需要 .mp4 文件")
        if self.output_max_width < 0 or self.output_max_width == 1:
            raise ValueError("输出最大宽度必须为 0 或至少 2 像素")
        if self.display and not self.display_window_name.strip():
            raise ValueError("启用即时显示时窗口名称不能为空")


@dataclass(frozen=True)
class ProcessingStats:
    """记录本次视频处理的帧数和推理调度统计。"""

    total_frames: int
    written_frames: int
    model_frames: int
    global_model_frames: int
    local_model_frames: int
    kalman_only_frames: int
    frames_without_box: int
    average_frame_time_ms: float
    average_processing_fps: float
    average_read_time_ms: float
    average_tracking_time_ms: float
    average_drawing_time_ms: float
    average_display_time_ms: float
    average_encoding_time_ms: float
    stopped_by_user: bool
    output_path: Path
    encoder_finalize_time_ms: float
    processing_fps_with_finalize: float
    reader_finalize_time_ms: float



class PersonVideoProcessor(PersonTrackingEngine):
    """复用逐帧识别引擎，保留离线视频读取、显示和同步输出。"""

    def __init__(self, config: ProcessorConfig) -> None:
        """
        作用：校验离线视频配置并初始化共用识别引擎。
        返回：无。
        异常：配置无效或模型加载失败时抛出异常。
        副作用：读取模型权重并初始化推理资源。
        """
        super().__init__(config)

    def process(self) -> ProcessingStats:
        """
        作用：逐帧同步处理、即时显示人物跟踪框并生成输出视频。
        返回：包含帧数、平均单帧耗时、处理速度和输出路径的统计。
        异常：
            视频无法打开、ROI 无效或输出编码器无法创建时抛出异常。
        副作用：读取输入视频、执行模型推理并写入输出视频文件。
        """
        capture = create_video_capture(self.config.input_path, self.config.decoder,
                                       self.config.decode_prefetch, self.config.gst_python)
        writer: cv2.VideoWriter | GStreamerVideoWriter | None = None
        display_window_opened = False
        try:
            fps, frame_width, frame_height = self._read_video_metadata(capture)
            roi_y_max = self._resolve_roi_y_max(frame_height)
            output_size = self._resolve_output_size(frame_width, frame_height)
            logger.info("视频输入=%sx%s，输出=%sx%s，fps=%.3f",
                        frame_width, frame_height, *output_size, fps)
            self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = self._create_writer(fps, *output_size)
            if self.config.display:
                self._open_display_window(*output_size)
                display_window_opened = True
            total_frames = 0
            written_frames = 0
            model_frames = 0
            global_model_frames = 0
            local_model_frames = 0
            kalman_only_frames = 0
            frames_without_box = 0
            total_processing_seconds = 0.0
            read_seconds = tracking_seconds = drawing_seconds = 0.0
            display_seconds = encoding_seconds = 0.0
            stopped_by_user = False
            last_timestamp_ms = -1.0
            processing_started_at = time.perf_counter()
            while True:
                frame_started_at = time.perf_counter()
                read_succeeded, frame = capture.read()
                if not read_succeeded:
                    break
                timestamp_ms = self._frame_timestamp_ms(
                    capture=capture,
                    frame_index=total_frames,
                    fps=fps,
                    last_timestamp_ms=last_timestamp_ms,
                )
                last_timestamp_ms = timestamp_ms
                read_finished_at = time.perf_counter()
                tracking_result, used_model = self.process_frame(frame, timestamp_ms)
                if used_model:
                    model_frames += 1
                    if tracking_result.model_scope == "global":
                        global_model_frames += 1
                    elif tracking_result.model_scope == "local":
                        local_model_frames += 1
                elif tracking_result.bbox is not None:
                    kalman_only_frames += 1
                if tracking_result.bbox is None:
                    frames_without_box += 1
                tracking_finished_at = time.perf_counter()
                annotated_frame = self._draw_result(
                    frame=frame,
                    frame_index=total_frames,
                    timestamp_ms=timestamp_ms,
                    result=tracking_result,
                    roi_y_max=roi_y_max,
                    output_size=output_size,
                )
                drawing_finished_at = time.perf_counter()
                should_continue = self._display_frame(annotated_frame)
                display_finished_at = time.perf_counter()
                self._write_frame_synchronously(
                    writer=writer,
                    frame=annotated_frame,
                    frame_index=total_frames,
                )
                encoding_finished_at = time.perf_counter()
                written_frames += 1
                read_seconds += read_finished_at - frame_started_at
                tracking_seconds += tracking_finished_at - read_finished_at
                drawing_seconds += drawing_finished_at - tracking_finished_at
                display_seconds += display_finished_at - drawing_finished_at
                encoding_seconds += encoding_finished_at - display_finished_at
                total_processing_seconds += encoding_finished_at - frame_started_at
                total_frames += 1
                if total_frames % max(int(round(fps)), 1) == 0:
                    logger.info("已处理人物视频 %s 帧", total_frames)
                if not should_continue:
                    stopped_by_user = True
                    logger.info("用户已请求停止人物视频处理")
                    break
            if total_frames == 0:
                raise RuntimeError("输入视频不包含可读取的视频帧")
            if written_frames != total_frames:
                raise RuntimeError(
                    "同步输出帧数与处理帧数不一致："
                    f"处理 {total_frames} 帧，写入 {written_frames} 帧"
                )
            reader_finalize_started_at = time.perf_counter()
            if isinstance(capture, GStreamerVideoCapture):
                capture.log_statistics()
            capture.release()
            capture = None
            finalize_started_at = time.perf_counter()
            writer.release()
            writer = None
            output_finished_at = time.perf_counter()
            average_frame_time_ms = (
                total_processing_seconds * 1000.0 / total_frames
            )
            average_processing_fps = (
                1000.0 / average_frame_time_ms
                if average_frame_time_ms > 0.0
                else 0.0
            )
            return ProcessingStats(
                total_frames=total_frames,
                written_frames=written_frames,
                model_frames=model_frames,
                global_model_frames=global_model_frames,
                local_model_frames=local_model_frames,
                kalman_only_frames=kalman_only_frames,
                frames_without_box=frames_without_box,
                average_frame_time_ms=average_frame_time_ms,
                average_processing_fps=average_processing_fps,
                average_read_time_ms=read_seconds * 1000.0 / total_frames,
                average_tracking_time_ms=tracking_seconds * 1000.0 / total_frames,
                average_drawing_time_ms=drawing_seconds * 1000.0 / total_frames,
                average_display_time_ms=display_seconds * 1000.0 / total_frames,
                average_encoding_time_ms=encoding_seconds * 1000.0 / total_frames,
                stopped_by_user=stopped_by_user,
                output_path=self.config.output_path,
                encoder_finalize_time_ms=(output_finished_at - finalize_started_at) * 1000,
                processing_fps_with_finalize=total_frames / (output_finished_at - processing_started_at),
                reader_finalize_time_ms=(finalize_started_at - reader_finalize_started_at) * 1000,
            )
        finally:
            if capture is not None:
                capture.release()
            if writer is not None:
                if isinstance(writer, GStreamerVideoWriter):
                    writer.abort()
                else:
                    writer.release()
            if display_window_opened:
                self._close_display_window()


    def _resolve_output_size(self, frame_width: int, frame_height: int) -> tuple[int, int]:
        """计算不放大的输出尺寸；按宽度等比例缩小，两边向下对齐偶数以供编码。"""
        if frame_width < 2 or frame_height < 2:
            raise ValueError("输出视频宽高必须至少为 2 像素")
        limit = self.config.output_max_width
        width = min(frame_width, limit) if limit else frame_width
        width -= width % 2
        height = max(2, int(frame_height * width / frame_width) // 2 * 2)
        return width, height

    def _open_display_window(self, frame_width: int, frame_height: int) -> None:
        """
        作用：创建用于逐帧即时显示处理结果的可缩放窗口。
        参数：
            frame_width：输入视频原始宽度，单位为像素。
            frame_height：输入视频原始高度，单位为像素。
        返回：无。
        异常：
            当前 OpenCV 环境不支持图形窗口时抛出 RuntimeError。
        副作用：创建 OpenCV 桌面显示窗口。
        """
        try:
            cv2.namedWindow(self.config.display_window_name, cv2.WINDOW_NORMAL)
            display_width = min(frame_width, 1280)
            display_height = max(
                int(round(frame_height * display_width / frame_width)),
                1,
            )
            cv2.resizeWindow(
                self.config.display_window_name,
                display_width,
                display_height,
            )
        except cv2.error as exc:
            raise RuntimeError(
                "无法创建即时显示窗口；无桌面环境请使用 --no-display"
            ) from exc


    def _display_frame(self, frame: np.ndarray) -> bool:
        """
        作用：在当前帧处理完成后立即显示这一帧并读取停止按键。
        参数：
            frame：已经绘制人物框和状态信息的当前视频帧。
        返回：继续处理时返回 True；按下 Q、Esc 或关闭窗口时返回 False。
        异常：
            OpenCV 即时显示失败时抛出 RuntimeError。
        副作用：刷新桌面窗口并处理一次窗口键盘事件。
        """
        if not self.config.display:
            return True
        try:
            cv2.imshow(self.config.display_window_name, frame)
            pressed_key = cv2.waitKey(1) & 0xFF
            if pressed_key in {27, ord("q"), ord("Q")}:
                return False
            window_visible = cv2.getWindowProperty(
                self.config.display_window_name,
                cv2.WND_PROP_VISIBLE,
            )
            return window_visible >= 1.0
        except cv2.error as exc:
            raise RuntimeError(
                "即时显示视频帧失败；无桌面环境请使用 --no-display"
            ) from exc


    def _close_display_window(self) -> None:
        """
        作用：安全关闭人物跟踪即时显示窗口。
        返回：无。
        副作用：关闭 OpenCV 桌面窗口并处理最后一次窗口事件。
        """
        try:
            cv2.destroyWindow(self.config.display_window_name)
            cv2.waitKey(1)
        except cv2.error:
            logger.debug("人物跟踪显示窗口已经关闭")


    @staticmethod
    def _write_frame_synchronously(
        writer: cv2.VideoWriter | GStreamerVideoWriter,
        frame: np.ndarray,
        frame_index: int,
    ) -> None:
        """
        作用：处理一帧后立即向输出视频编码器提交这一帧。
        参数：
            writer：已经打开的 OpenCV 视频写入对象。
            frame：完成模型或卡尔曼标注的当前视频帧。
            frame_index：当前从零开始的视频帧编号。
        返回：无。
        异常：
            输出视频编码器已经关闭时抛出 RuntimeError。
        副作用：向输出视频写入一个画面帧。
        """
        if not isinstance(writer, GStreamerVideoWriter) and not writer.isOpened():
            raise RuntimeError(f"写入第 {frame_index} 帧前视频编码器已经关闭")
        writer.write(frame)


    @staticmethod
    def _read_video_metadata(
        capture: cv2.VideoCapture | GStreamerVideoCapture,
    ) -> tuple[float, int, int]:
        """
        作用：读取输入视频帧率和画面尺寸。
        参数：
            capture：已经打开的 OpenCV 视频读取对象。
        返回：视频帧率、宽度和高度。
        异常：
            元数据缺失或数值无效时抛出 RuntimeError。
        """
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        frame_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if not np.isfinite(fps) or fps <= 0.0:
            raise RuntimeError("输入视频缺少有效帧率")
        if frame_width <= 0 or frame_height <= 0:
            raise RuntimeError("输入视频缺少有效画面尺寸")
        return fps, frame_width, frame_height


    def _create_writer(
        self,
        fps: float,
        frame_width: int,
        frame_height: int,
    ) -> cv2.VideoWriter | GStreamerVideoWriter:
        """
        作用：按输入帧率和指定输出画面尺寸创建编码器。
        参数：
            fps：输入视频帧率。
            frame_width：输出视频宽度，单位为像素。
            frame_height：输出视频高度，单位为像素。
        返回：已经打开的 OpenCV 视频写入对象。
        异常：
            系统不支持指定编码或输出路径无法写入时抛出 RuntimeError。
        副作用：创建或覆盖输出视频文件。
        """
        return create_video_writer(
            self.config.output_path, fps, frame_width, frame_height,
            self.config.encoder, self.config.codec, self.config.video_bitrate,
        )


    @staticmethod
    def _frame_timestamp_ms(
        capture: cv2.VideoCapture | GStreamerVideoCapture,
        frame_index: int,
        fps: float,
        last_timestamp_ms: float,
    ) -> float:
        """
        作用：读取当前视频帧时间戳，并在容器时间无效时按帧率回退计算。
        参数：
            capture：当前输入视频读取对象。
            frame_index：当前从零开始的视频帧编号。
            fps：输入视频帧率。
            last_timestamp_ms：上一帧最终采用的视频毫秒时间戳。
        返回：严格递增的当前帧毫秒时间戳。
        """
        container_timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
        calculated_timestamp_ms = frame_index * 1000.0 / fps
        if np.isfinite(container_timestamp_ms) and container_timestamp_ms > last_timestamp_ms:
            return container_timestamp_ms
        if frame_index == 0:
            return 0.0
        return max(calculated_timestamp_ms, last_timestamp_ms + 1000.0 / fps)
