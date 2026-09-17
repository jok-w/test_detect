from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detector import YoloPersonDetector
from .kalman import BoundingBoxKalmanFilter
from .tracking_config import PersonTrackingConfig
from .types import BoundingBox, CropWindow, PersonDetection


@dataclass(frozen=True)
class FrameTrackingResult:
    """表示单帧最终绘制的人物框和结果来源。"""

    bbox: BoundingBox | None
    source: str
    class_name: str | None
    confidence: float | None
    state: str
    model_scope: str
    crop_window: CropWindow | None



class PersonTrackingEngine:
    """共用的单人物检测、卡尔曼跟踪与动态裁剪引擎。"""

    def __init__(self, config: PersonTrackingConfig) -> None:
        """
        作用：初始化共用人物识别引擎，加载模型并建立空跟踪状态。
        参数：
            config：模型、ROI 和检测节奏配置。
        返回：无。
        异常：
            配置无效或模型无法加载时抛出异常。
        副作用：读取 YOLO11n 权重并初始化推理资源。
        """
        config.validate()
        self.config = config
        self.detector = YoloPersonDetector(
            model_path=config.model_path,
            confidence=config.confidence,
            iou_threshold=config.iou_threshold,
            image_size=config.image_size,
            device=config.device,
        )
        self.reset()

    def reset(self) -> None:
        """
        作用：清空人物轨迹、历史姿态和动态裁剪，保留已加载的模型。
        返回：无。
        """
        self.kalman = BoundingBoxKalmanFilter()
        self.state = "uninitialized"
        self.warmup_success_count = 0
        self.prediction_only_count = 0
        self.local_miss_count = 0
        self.recovery_success_count = 0
        self.last_success_timestamp_ms: float | None = None
        self.last_detection: PersonDetection | None = None
        self.crop_center: tuple[float, float] | None = None
        self.crop_base_size: float | None = None
        self.crop_edge_expansion = 1.0
        self._last_frame_timestamp_ms: float | None = None
        self._frame_shape: tuple[int, ...] | None = None
        self._session_id = ""

    def process_frame(self, frame: np.ndarray, timestamp_ms: float, session_id: str = "") -> tuple[FrameTrackingResult, bool]:
        """
        作用：处理原始 BGR 帧；新会话、尺寸变化或长时间断帧时重建跟踪状态。
        参数：timestamp_ms：单调递增的帧时间，单位毫秒；session_id：采集会话编号。
        返回：当前帧识别结果和本帧是否执行模型检测。
        异常：帧或时间戳无效、引擎已关闭时抛出 ValueError 或 RuntimeError。
        副作用：执行模型推理并更新跟踪状态。
        """
        if self.detector is None:
            raise RuntimeError("人物识别引擎已关闭")
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            raise ValueError("人物识别需要有效的 uint8 BGR 图像")
        if not np.isfinite(timestamp_ms) or timestamp_ms < 0:
            raise ValueError("帧时间戳必须是有效的非负毫秒数")
        if self._frame_shape != frame.shape or self._session_id != session_id:
            self.reset()
        previous = self._last_frame_timestamp_ms
        if previous is not None:
            if timestamp_ms <= previous:
                raise ValueError("同一会话的帧时间戳必须严格递增")
            if timestamp_ms - previous > 2000:
                self.reset()
        self._frame_shape = frame.shape
        self._session_id = session_id
        self._last_frame_timestamp_ms = timestamp_ms
        return self._process_frame(frame, timestamp_ms, self._resolve_roi_y_max(frame.shape[0]))

    def draw_frame(self, frame: np.ndarray, result: FrameTrackingResult, sequence: int, timestamp_ms: float, show_regions: bool = True) -> np.ndarray:
        """
        作用：在产生识别结果的同一帧上绘制人物框和状态，可隐藏区域辅助线。
        返回：独立的 BGR 标注画面。
        """
        return self._draw_result(frame, sequence, timestamp_ms, result, self._resolve_roi_y_max(frame.shape[0]), show_regions)

    def close(self) -> None:
        """
        作用：清空轨迹并释放引擎持有的模型引用。
        返回：无。
        副作用：释放模型引用；独立识别进程退出后回收其推理资源。
        """
        self.reset()
        self.detector = None

    def _process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: float,
        roi_y_max: int,
    ) -> tuple[FrameTrackingResult, bool]:
        """
        作用：按照全局预热、动态局部跟踪和全局恢复状态处理一个视频帧。
        参数：
            frame：当前原始视频帧。
            timestamp_ms：当前帧的视频毫秒时间戳。
            roi_y_max：固定纵向人物区域的结束像素坐标。
        返回：单帧跟踪结果和本帧是否运行模型。
        副作用：可能执行模型推理并更新卡尔曼状态和调度状态。
        """
        if not self.kalman.initialized:
            detection = self._detect_global_person(frame, None, roi_y_max)
            if detection is None:
                return self._empty_result("waiting", "global"), True
            bbox = self.kalman.initialize(detection.bbox, timestamp_ms)
            self.state = "warmup"
            self.warmup_success_count = 1
            self._remember_detection(detection, timestamp_ms)
            return self._result_from_detection(
                bbox=bbox,
                detection=detection,
                model_scope="global",
            ), True

        predicted_bbox = self.kalman.predict(timestamp_ms)
        if self.state == "warmup":
            return self._process_warmup_frame(
                frame=frame,
                predicted_bbox=predicted_bbox,
                timestamp_ms=timestamp_ms,
                roi_y_max=roi_y_max,
            )
        if self.state == "local":
            return self._process_local_frame(
                frame=frame,
                predicted_bbox=predicted_bbox,
                timestamp_ms=timestamp_ms,
                roi_y_max=roi_y_max,
            )
        return self._process_recovery_frame(
            frame=frame,
            predicted_bbox=predicted_bbox,
            timestamp_ms=timestamp_ms,
            roi_y_max=roi_y_max,
        )


    def _process_warmup_frame(
        self,
        frame: np.ndarray,
        predicted_bbox: BoundingBox,
        timestamp_ms: float,
        roi_y_max: int,
    ) -> tuple[FrameTrackingResult, bool]:
        """
        作用：在固定纵向区域连续检测人物并建立可靠的卡尔曼运动状态。
        参数：
            frame：当前原始视频帧。
            predicted_bbox：当前帧卡尔曼预测框。
            timestamp_ms：当前帧的视频毫秒时间戳。
            roi_y_max：固定纵向人物区域的结束像素坐标。
        返回：当前帧结果和固定为 True 的模型使用标志。
        副作用：执行全局模型推理并可能更新卡尔曼和跟踪状态。
        """
        detection = self._detect_global_person(
            frame=frame,
            predicted_bbox=self._selection_reference(
                predicted_bbox,
                timestamp_ms,
            ),
            roi_y_max=roi_y_max,
        )
        if detection is None:
            self.warmup_success_count = 0
            return self._predicted_result(
                bbox=predicted_bbox,
                timestamp_ms=timestamp_ms,
                model_scope="global",
            ), True

        corrected_bbox = self.kalman.correct(detection.bbox)
        self._remember_detection(detection, timestamp_ms)
        self.warmup_success_count += 1
        if self.warmup_success_count >= self.config.warmup_detections:
            self.state = "local"
            self.prediction_only_count = 0
            self._reset_dynamic_crop()
        return self._result_from_detection(
            bbox=corrected_bbox,
            detection=detection,
            model_scope="global",
        ), True


    def _process_local_frame(
        self,
        frame: np.ndarray,
        predicted_bbox: BoundingBox,
        timestamp_ms: float,
        roi_y_max: int,
    ) -> tuple[FrameTrackingResult, bool]:
        """
        作用：围绕卡尔曼预测中心生成动态裁剪并执行局部人物检测。
        参数：
            frame：当前原始视频帧。
            predicted_bbox：当前帧卡尔曼预测人物框。
            timestamp_ms：当前帧的视频毫秒时间戳。
            roi_y_max：固定纵向人物区域的结束像素坐标。
        返回：当前帧结果和本帧是否运行模型。
        副作用：可能执行局部或全局模型推理并更新卡尔曼和裁剪状态。
        """
        if self._prediction_age_ms(timestamp_ms) > self.config.global_recovery_ms:
            self.state = "recovery"
            self.recovery_success_count = 0
            return self._process_recovery_frame(
                frame=frame,
                predicted_bbox=predicted_bbox,
                timestamp_ms=timestamp_ms,
                roi_y_max=roi_y_max,
            )

        crop_window = self._create_dynamic_crop(
            predicted_bbox=predicted_bbox,
            frame_width=frame.shape[1],
            roi_y_max=roi_y_max,
            expansion=self._local_crop_expansion(),
        )
        if self.prediction_only_count < self.config.prediction_frames:
            self.prediction_only_count += 1
            return self._predicted_result(
                bbox=predicted_bbox,
                timestamp_ms=timestamp_ms,
                crop_window=crop_window,
            ), False

        self.prediction_only_count = 0
        detection = self._detect_local_person(
            frame=frame,
            crop_window=crop_window,
            predicted_bbox=predicted_bbox,
        )
        if detection is None:
            self.local_miss_count += 1
            if self.local_miss_count >= self.config.recovery_misses:
                self.state = "recovery"
                self.recovery_success_count = 0
            return self._predicted_result(
                bbox=predicted_bbox,
                timestamp_ms=timestamp_ms,
                model_scope="local",
                crop_window=crop_window,
            ), True

        corrected_bbox = self.kalman.correct(detection.bbox)
        self._remember_detection(detection, timestamp_ms)
        self.local_miss_count = 0
        self.crop_edge_expansion = (
            1.25
            if self._is_detection_near_crop_edge(detection.bbox, crop_window)
            else 1.0
        )
        return self._result_from_detection(
            bbox=corrected_bbox,
            detection=detection,
            model_scope="local",
            crop_window=crop_window,
        ), True


    def _process_recovery_frame(
        self,
        frame: np.ndarray,
        predicted_bbox: BoundingBox,
        timestamp_ms: float,
        roi_y_max: int,
    ) -> tuple[FrameTrackingResult, bool]:
        """
        作用：在局部跟踪失效后使用固定纵向区域重新捕获唯一人员。
        参数：
            frame：当前原始视频帧。
            predicted_bbox：当前帧卡尔曼预测人物框。
            timestamp_ms：当前帧的视频毫秒时间戳。
            roi_y_max：固定纵向人物区域的结束像素坐标。
        返回：当前帧结果和固定为 True 的模型使用标志。
        副作用：执行全局模型推理并可能更新卡尔曼和跟踪状态。
        """
        detection = self._detect_global_person(
            frame=frame,
            predicted_bbox=self._selection_reference(
                predicted_bbox,
                timestamp_ms,
            ),
            roi_y_max=roi_y_max,
        )
        if detection is None:
            self.recovery_success_count = 0
            return self._predicted_result(
                bbox=predicted_bbox,
                timestamp_ms=timestamp_ms,
                model_scope="global",
            ), True

        corrected_bbox = self.kalman.correct(detection.bbox)
        self._remember_detection(detection, timestamp_ms)
        self.recovery_success_count += 1
        if self.recovery_success_count >= self.config.recovery_detections:
            self.state = "local"
            self.local_miss_count = 0
            self.prediction_only_count = 0
            self._reset_dynamic_crop()
        return self._result_from_detection(
            bbox=corrected_bbox,
            detection=detection,
            model_scope="global",
        ), True


    def _detect_global_person(
        self,
        frame: np.ndarray,
        predicted_bbox: BoundingBox | None,
        roi_y_max: int,
    ) -> PersonDetection | None:
        """
        作用：在固定纵向 ROI 内运行较大输入尺寸模型并选择唯一受训人员。
        参数：
            frame：当前原始视频帧。
            predicted_bbox：可靠时用于筛选候选框的卡尔曼预测框。
            roi_y_max：固定纵向人物区域的结束像素坐标。
        返回：恢复到原始画面坐标的唯一人物检测；未检测到时返回 None。
        副作用：执行一次 YOLO11n 全局模型推理。
        """
        roi_frame = frame[self.config.roi_y_min : roi_y_max, :]
        roi_detections = self.detector.detect(
            roi_frame,
            image_size=self.config.image_size,
        )
        detections = [
            PersonDetection(
                bbox=detection.bbox.shifted(y_offset=self.config.roi_y_min),
                confidence=detection.confidence,
                class_id=detection.class_id,
                class_name=detection.class_name,
            )
            for detection in roi_detections
        ]
        return self.detector.select_single_person(
            detections=detections,
            predicted_bbox=predicted_bbox,
        )


    def _detect_local_person(
        self,
        frame: np.ndarray,
        crop_window: CropWindow,
        predicted_bbox: BoundingBox,
    ) -> PersonDetection | None:
        """
        作用：对卡尔曼引导的局部裁剪运行模型并将结果映射回原始画面。
        参数：
            frame：当前原始视频帧。
            crop_window：原始画面坐标系中的动态裁剪窗口。
            predicted_bbox：用于从候选框中选择唯一人员的卡尔曼预测框。
        返回：原始画面坐标系中的人物检测；未检测到时返回 None。
        异常：
            裁剪窗口不包含有效像素时抛出 RuntimeError。
        副作用：执行一次 YOLO11n 局部模型推理。
        """
        if not crop_window.contains_valid_area():
            raise RuntimeError("动态人物裁剪窗口不包含有效像素")
        crop_frame = frame[
            crop_window.y1 : crop_window.y2,
            crop_window.x1 : crop_window.x2,
        ]
        crop_detections = self.detector.detect(
            crop_frame,
            image_size=self.config.local_image_size,
        )
        detections = [
            PersonDetection(
                bbox=detection.bbox.shifted(
                    x_offset=crop_window.x1,
                    y_offset=crop_window.y1,
                ),
                confidence=detection.confidence,
                class_id=detection.class_id,
                class_name=detection.class_name,
            )
            for detection in crop_detections
        ]
        return self.detector.select_single_person(
            detections=detections,
            predicted_bbox=predicted_bbox,
        )


    def _create_dynamic_crop(
        self,
        predicted_bbox: BoundingBox,
        frame_width: int,
        roi_y_max: int,
        expansion: float,
    ) -> CropWindow:
        """
        作用：根据全局卡尔曼预测框生成经过平滑和边界平移的近似方形裁剪窗口。
        参数：
            predicted_bbox：原始画面坐标系中的卡尔曼预测框。
            frame_width：原始视频宽度，单位为像素。
            roi_y_max：固定纵向人物区域的结束像素坐标。
            expansion：漏检或靠近边缘时使用的裁剪扩大倍数。
        返回：完全位于固定人物区域内的整数裁剪窗口。
        副作用：更新下一帧使用的平滑裁剪中心和基础尺寸。
        """
        roi_height = roi_y_max - self.config.roi_y_min
        maximum_size = float(
            min(self.config.crop_max_size, frame_width, roi_height)
        )
        minimum_size = float(min(self.config.crop_min_size, maximum_size))
        target_center_x, target_center_y = predicted_bbox.center
        target_size = max(
            predicted_bbox.height / self.config.crop_person_height_ratio,
            predicted_bbox.width / self.config.crop_person_width_ratio,
            minimum_size,
        )
        target_size = min(target_size, maximum_size)

        if self.crop_center is None:
            smoothed_center_x = target_center_x
            smoothed_center_y = target_center_y
        else:
            center_alpha = self.config.crop_center_smoothing
            smoothed_center_x = (
                (1.0 - center_alpha) * self.crop_center[0]
                + center_alpha * target_center_x
            )
            smoothed_center_y = (
                (1.0 - center_alpha) * self.crop_center[1]
                + center_alpha * target_center_y
            )
        self.crop_center = (smoothed_center_x, smoothed_center_y)

        if self.crop_base_size is None:
            smoothed_size = target_size
        else:
            size_alpha = self.config.crop_size_smoothing
            blended_size = (
                (1.0 - size_alpha) * self.crop_base_size
                + size_alpha * target_size
            )
            change_ratio = self.config.crop_max_size_change_ratio
            smoothed_size = min(
                max(
                    blended_size,
                    self.crop_base_size * (1.0 - change_ratio),
                ),
                self.crop_base_size * (1.0 + change_ratio),
            )
        self.crop_base_size = min(
            max(smoothed_size, minimum_size),
            maximum_size,
        )
        crop_size = int(
            round(
                min(
                    max(self.crop_base_size * max(expansion, 1.0), minimum_size),
                    maximum_size,
                )
            )
        )
        crop_size = max(crop_size, 1)
        x1 = int(round(smoothed_center_x - crop_size / 2.0))
        y1 = int(round(smoothed_center_y - crop_size / 2.0))
        x1 = min(max(x1, 0), frame_width - crop_size)
        y1 = min(
            max(y1, self.config.roi_y_min),
            roi_y_max - crop_size,
        )
        return CropWindow(
            x1=x1,
            y1=y1,
            x2=x1 + crop_size,
            y2=y1 + crop_size,
        )


    def _local_crop_expansion(self) -> float:
        """
        作用：根据连续局部漏检次数和边缘预警确定下一次裁剪扩大倍数。
        返回：正常为 1.0，首次漏检后为 1.25，再次漏检后为 1.75。
        """
        if self.local_miss_count >= 2:
            miss_expansion = 1.75
        elif self.local_miss_count == 1:
            miss_expansion = 1.25
        else:
            miss_expansion = 1.0
        return max(miss_expansion, self.crop_edge_expansion)


    def _is_detection_near_crop_edge(
        self,
        bbox: BoundingBox,
        crop_window: CropWindow,
    ) -> bool:
        """
        作用：判断局部检测框是否接近裁剪边缘并需要下一帧提前扩大区域。
        参数：
            bbox：已经映射到原始画面的模型检测框。
            crop_window：本次模型推理使用的动态裁剪窗口。
        返回：检测框进入任一边缘预警带时返回 True，否则返回 False。
        """
        margin = min(crop_window.width, crop_window.height)
        margin *= self.config.crop_edge_margin_ratio
        return (
            bbox.x1 <= crop_window.x1 + margin
            or bbox.y1 <= crop_window.y1 + margin
            or bbox.x2 >= crop_window.x2 - margin
            or bbox.y2 >= crop_window.y2 - margin
        )


    def _prediction_age_ms(self, timestamp_ms: float) -> float:
        """
        作用：计算当前帧距离最近一次成功模型检测的时间。
        参数：
            timestamp_ms：当前帧视频毫秒时间戳。
        返回：经过的毫秒数；没有成功检测记录时返回正无穷。
        """
        if self.last_success_timestamp_ms is None:
            return float("inf")
        return max(timestamp_ms - self.last_success_timestamp_ms, 0.0)


    def _reset_dynamic_crop(self) -> None:
        """
        作用：清除旧轨迹留下的裁剪平滑和边缘扩大状态。
        返回：无。
        副作用：下一次局部裁剪将直接使用最新卡尔曼预测框初始化。
        """
        self.crop_center = None
        self.crop_base_size = None
        self.crop_edge_expansion = 1.0


    def _selection_reference(
        self,
        predicted_bbox: BoundingBox,
        timestamp_ms: float,
    ) -> BoundingBox | None:
        """
        作用：判断当前预测框是否仍适合作为模型候选框筛选依据。
        参数：
            predicted_bbox：当前帧卡尔曼预测人物框。
            timestamp_ms：当前帧视频毫秒时间戳。
        返回：预测仍可靠时返回预测框，超时后返回 None 以允许全 ROI 重捕获。
        """
        if self.last_success_timestamp_ms is None:
            return None
        if (
            timestamp_ms - self.last_success_timestamp_ms
            > self.config.max_prediction_ms
        ):
            return None
        return predicted_bbox


    def _predicted_result(
        self,
        bbox: BoundingBox,
        timestamp_ms: float,
        model_scope: str = "none",
        crop_window: CropWindow | None = None,
    ) -> FrameTrackingResult:
        """
        作用：构造卡尔曼纯预测帧的输出，并在预测超时后停止绘制框。
        参数：
            bbox：当前帧卡尔曼预测人物框。
            timestamp_ms：当前帧视频毫秒时间戳。
            model_scope：本帧模型搜索范围；未运行模型时为 none。
            crop_window：本帧使用或展示的动态裁剪窗口。
        返回：可靠预测或已经失效的单帧跟踪结果。
        """
        if self.last_success_timestamp_ms is None:
            return self._empty_result("waiting", model_scope, crop_window)
        prediction_age_ms = timestamp_ms - self.last_success_timestamp_ms
        if prediction_age_ms > self.config.max_prediction_ms:
            return self._empty_result("lost", model_scope, crop_window)
        class_name = self.last_detection.class_name if self.last_detection else None
        confidence = self.last_detection.confidence if self.last_detection else None
        return FrameTrackingResult(
            bbox=bbox,
            source="kalman",
            class_name=class_name,
            confidence=confidence,
            state=self.state,
            model_scope=model_scope,
            crop_window=crop_window,
        )


    def _result_from_detection(
        self,
        bbox: BoundingBox,
        detection: PersonDetection,
        model_scope: str,
        crop_window: CropWindow | None = None,
    ) -> FrameTrackingResult:
        """
        作用：构造模型测量修正后的单帧跟踪结果。
        参数：
            bbox：卡尔曼修正后的平滑人物框。
            detection：本帧模型姿态检测结果。
            model_scope：本帧使用 global 或 local 模型搜索范围。
            crop_window：局部模型推理使用的动态裁剪窗口。
        返回：来源标记为模型检测的单帧结果。
        """
        return FrameTrackingResult(
            bbox=bbox,
            source="model",
            class_name=detection.class_name,
            confidence=detection.confidence,
            state=self.state,
            model_scope=model_scope,
            crop_window=crop_window,
        )


    def _remember_detection(
        self,
        detection: PersonDetection,
        timestamp_ms: float,
    ) -> None:
        """
        作用：保存最近一次成功模型检测及其视频时间。
        参数：
            detection：最近一次有效人物姿态检测。
            timestamp_ms：成功检测帧的视频毫秒时间戳。
        返回：无。
        副作用：更新预测帧使用的姿态标签和可靠时间基准。
        """
        self.last_detection = detection
        self.last_success_timestamp_ms = timestamp_ms


    @staticmethod
    def _empty_result(
        state: str,
        model_scope: str = "none",
        crop_window: CropWindow | None = None,
    ) -> FrameTrackingResult:
        """
        作用：构造当前帧没有可靠人物框的结果。
        参数：
            state：等待首次检测或跟踪丢失状态。
            model_scope：本帧模型搜索范围；未运行模型时为 none。
            crop_window：本帧使用或展示的动态裁剪窗口。
        返回：不包含人物框的单帧结果。
        """
        return FrameTrackingResult(
            bbox=None,
            source="none",
            class_name=None,
            confidence=None,
            state=state,
            model_scope=model_scope,
            crop_window=crop_window,
        )


    def _draw_result(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_ms: float,
        result: FrameTrackingResult,
        roi_y_max: int,
        show_regions: bool = True,
    ) -> np.ndarray:
        """
        作用：在视频帧上绘制 ROI、人物框、姿态类别和调度状态。
        参数：
            frame：当前原始视频帧。
            frame_index：当前从零开始的视频帧编号。
            timestamp_ms：当前帧视频毫秒时间戳。
            result：当前帧最终人物跟踪结果。
            roi_y_max：固定纵向人物区域的结束像素坐标。
        返回：完成可视化标注的视频帧。
        """
        annotated = frame.copy()
        frame_height, frame_width = annotated.shape[:2]
        if show_regions:
            cv2.line(
                annotated,
                (0, self.config.roi_y_min),
                (frame_width - 1, self.config.roi_y_min),
                (255, 160, 0),
                1,
            )
            cv2.line(
                annotated,
                (0, roi_y_max - 1),
                (frame_width - 1, roi_y_max - 1),
                (255, 160, 0),
                1,
            )
            if result.crop_window is not None:
                crop = result.crop_window
                cv2.rectangle(
                    annotated,
                    (crop.x1, crop.y1),
                    (crop.x2 - 1, crop.y2 - 1),
                    (255, 0, 255),
                    1,
                )
        if result.bbox is not None:
            bbox = result.bbox.clamped(frame_width, frame_height)
            x1, y1, x2, y2 = bbox.as_int_xyxy()
            color = (0, 200, 0) if result.source == "model" else (0, 180, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            confidence_text = (
                f" {result.confidence:.2f}"
                if result.confidence is not None
                else ""
            )
            label = (
                f"{result.source}: {result.class_name or 'person'}"
                f"{confidence_text}"
            )
            self._draw_text(annotated, label, (x1, max(y1 - 8, 20)), color)
        status_text = (
            f"frame={frame_index} time={timestamp_ms:.1f}ms "
            f"state={result.state} source={result.source} "
            f"scope={result.model_scope} misses={self.local_miss_count}"
        )
        self._draw_text(annotated, status_text, (12, 28), (255, 255, 255))
        return annotated


    @staticmethod
    def _draw_text(
        frame: np.ndarray,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        """
        作用：在视频帧上绘制带黑色描边的英文状态文本。
        参数：
            frame：需要修改的 OpenCV BGR 视频帧。
            text：需要绘制的状态字符串。
            origin：文本左下角像素坐标。
            color：文本的 BGR 颜色。
        返回：无。
        副作用：直接修改传入的视频帧像素。
        """
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
            cv2.LINE_AA,
        )


    def _resolve_roi_y_max(self, frame_height: int) -> int:
        """
        作用：根据视频高度确定固定纵向人物区域的有效终点。
        参数：
            frame_height：输入视频原始高度，单位为像素。
        返回：限制在视频范围内的纵向裁剪终点。
        异常：
            ROI 起点超出视频或最终区域为空时抛出 RuntimeError。
        """
        if self.config.roi_y_min >= frame_height:
            raise RuntimeError("纵向 ROI 起点超出视频画面高度")
        roi_y_max = (
            frame_height
            if self.config.roi_y_max is None
            else min(self.config.roi_y_max, frame_height)
        )
        if roi_y_max <= self.config.roi_y_min:
            raise RuntimeError("纵向 ROI 在当前视频中没有有效高度")
        return roi_y_max

