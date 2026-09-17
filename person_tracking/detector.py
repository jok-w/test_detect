from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .types import (
    BoundingBox,
    PersonDetection,
    bbox_iou,
)


class YoloPersonDetector:
    """封装自定义 YOLO11n 三姿态人物检测模型。"""

    def __init__(
        self,
        model_path: Path,
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        device: str | None = None,
    ) -> None:
        """
        作用：加载 YOLO11n 模型并保存单帧推理参数。
        参数：
            model_path：用户提供的三姿态 YOLO11n 权重文件路径。
            confidence：模型输出人物框的最低置信度。
            iou_threshold：YOLO 非极大值抑制使用的交并比阈值。
            image_size：模型推理输入尺寸。
            device：Ultralytics 使用的计算设备；为空时自动选择。
        返回：无。
        异常：
            模型文件不存在或 Ultralytics 无法加载模型时抛出异常。
        副作用：读取并加载模型权重。
        """
        if not model_path.is_file():
            raise FileNotFoundError(f"模型文件不存在：{model_path}")
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = device

    def detect(
        self,
        frame: np.ndarray,
        image_size: int | None = None,
    ) -> list[PersonDetection]:
        """
        作用：对单帧、固定纵向区域或动态裁剪区域执行三姿态人物检测。
        参数：
            frame：OpenCV BGR 格式的单帧图像。
            image_size：本次推理使用的输入尺寸；为空时使用初始化尺寸。
        返回：当前帧全部人物姿态检测结果。
        副作用：执行一次模型推理，可能使用 GPU 计算资源。
        """
        results = self._predict(frame, image_size)
        return self._parse_result(results[0]) if results else []

    def detect_batch(
        self,
        frames: list[np.ndarray],
        image_size: int | None = None,
    ) -> list[list[PersonDetection]]:
        """批量检测全局分片；每项结果对应输入的同序分片。"""
        if not frames:
            return []
        results = self._predict(frames, image_size)
        if len(results) != len(frames):
            raise RuntimeError("全局分片模型结果数量与输入数量不一致")
        return [self._parse_result(result) for result in results]

    def _predict(
        self,
        source: np.ndarray | list[np.ndarray],
        image_size: int | None,
    ) -> Any:
        predict_arguments: dict[str, Any] = {
            "source": source,
            "conf": self.confidence,
            "iou": self.iou_threshold,
            "imgsz": image_size or self.image_size,
            "verbose": False,
            "agnostic_nms": True,
            "stream": False,
        }
        if self.device:
            predict_arguments["device"] = self.device
        return self.model.predict(**predict_arguments)

    @staticmethod
    def _parse_result(result: Any) -> list[PersonDetection]:
        if result.boxes is None:
            return []
        boxes = result.boxes
        if len(boxes) == 0:
            return []
        xyxy_values = boxes.xyxy.detach().cpu().numpy()
        confidence_values = boxes.conf.detach().cpu().numpy()
        class_values = boxes.cls.detach().cpu().numpy().astype(int)
        detections: list[PersonDetection] = []
        for xyxy, confidence, class_id in zip(
            xyxy_values,
            confidence_values,
            class_values,
        ):
            detections.append(
                PersonDetection(
                    bbox=BoundingBox(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                    ),
                    confidence=float(confidence),
                    class_id=int(class_id),
                    class_name=self._class_name(result.names, int(class_id)),
                )
            )
        return detections

    @staticmethod
    def select_single_person(
        detections: list[PersonDetection],
        predicted_bbox: BoundingBox | None,
        max_center_distance_ratio: float = 1.5,
    ) -> PersonDetection | None:
        """
        作用：从模型结果中选出场地内唯一受训人员的检测框。
        参数：
            detections：当前帧模型输出的候选人物框。
            predicted_bbox：卡尔曼给出的当前帧预测框；首次检测时为空。
            max_center_distance_ratio：允许检测中心偏离预测框的最大比例。
        返回：最符合唯一人员轨迹的检测结果；没有合理结果时返回 None。
        """
        if not detections:
            return None
        if predicted_bbox is None:
            return max(detections, key=lambda detection: detection.confidence)
        predicted_center_x, predicted_center_y = predicted_bbox.center
        normalization = max(predicted_bbox.diagonal, 1.0)
        scored_detections: list[tuple[float, PersonDetection]] = []
        for detection in detections:
            center_x, center_y = detection.bbox.center
            center_distance = float(
                np.hypot(
                    center_x - predicted_center_x,
                    center_y - predicted_center_y,
                )
            )
            normalized_distance = center_distance / normalization
            if normalized_distance > max_center_distance_ratio:
                continue
            score = (
                detection.confidence
                + 0.5 * bbox_iou(detection.bbox, predicted_bbox)
                - 0.25 * normalized_distance
            )
            scored_detections.append((score, detection))
        if not scored_detections:
            return None
        return max(scored_detections, key=lambda item: item[0])[1]

    @staticmethod
    def _class_name(names: Any, class_id: int) -> str:
        """
        作用：从 Ultralytics 模型类别表中读取姿态名称。
        参数：
            names：模型结果携带的类别名称映射或列表。
            class_id：当前人物框的类别编号。
        返回：模型类别名称；无法读取时返回类别编号字符串。
        """
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)
