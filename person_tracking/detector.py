from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .model_artifacts import (
    artifact_paths, file_sha256, read_engine_metadata, read_onnx_metadata,
    runtime_info, validate_engine, validate_source,
)
from .types import (
    BoundingBox,
    PersonDetection,
    bbox_iou,
)


logger = logging.getLogger(__name__)


class YoloPersonDetector:
    """封装自定义 YOLO11n 三姿态人物检测模型。"""

    def __init__(
        self,
        model_path: Path,
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        device: str | None = None,
        backend: str = "auto",
        local_image_size: int = 384,
        global_batch_size: int = 4,
        onnx_path: Path | None = None,
        global_engine_path: Path | None = None,
        local_engine_path: Path | None = None,
    ) -> None:
        """
        作用：加载 YOLO11n 模型并保存单帧推理参数。
        参数：
            model_path：用户提供的三姿态 YOLO11n 权重文件路径。
            confidence：模型输出人物框的最低置信度。
            iou_threshold：YOLO 非极大值抑制使用的交并比阈值。
            image_size：模型推理输入尺寸。
            device：为空或 auto 时优先使用 GPU；CUDA 不可用时回退 CPU。
        返回：无。
        异常：
            模型文件不存在或 Ultralytics 无法加载模型时抛出异常。
        副作用：读取并加载模型权重。
        """
        if backend not in {"auto", "pt", "onnx", "tensorrt"}:
            raise ValueError(f"不支持的推理后端：{backend}")
        if model_path.suffix.lower() != ".pt":
            raise ValueError("--model 需要 PT 权重；导出模型请使用 --onnx-model 或 --global-engine/--local-engine")
        if not model_path.is_file():
            raise FileNotFoundError(f"模型文件不存在：{model_path}")
        self.device = self._select_device(device)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.local_image_size = local_image_size
        self.global_batch_size = global_batch_size
        self.model = None  # PT 在 engine 无法使用时才加载，两个场景共用。
        self._models: dict[str, Any] = {}
        self.backend_by_scope: dict[str, str] = {}
        self.model_paths: dict[str, str] = {}
        default_onnx, default_global, default_local = artifact_paths(model_path)
        paths = {"global": global_engine_path or default_global, "local": local_engine_path or default_local}
        self._initialize_models(model_path, backend, onnx_path or default_onnx, paths)

    def _initialize_models(self, model_path: Path, backend: str, onnx_path: Path, paths: dict) -> None:
        from ultralytics import YOLO

        source_hash = None
        if backend == "onnx":
            import onnxruntime as ort

            # 第一版默认 ORT CPU；仅在实际安装 CUDA provider 时请求 GPU。
            if self.device != "cpu" and "CUDAExecutionProvider" not in ort.get_available_providers():
                logger.warning("ONNX Runtime 未提供 CUDAExecutionProvider，使用 ONNX CPU 推理")
                self.device = "cpu"
        if backend == "tensorrt" and self.device == "cpu":
            raise ValueError("显式指定 TensorRT 时需要可用 CUDA；CPU 回退请使用 --backend auto")
        for scope, size, batch in (("global", self.image_size, self.global_batch_size),
                                    ("local", self.local_image_size, 1)):
            selected = backend
            path = paths[scope] if backend in {"auto", "tensorrt"} else onnx_path
            model = None
            if backend == "pt" or (backend == "auto" and self.device == "cpu"):
                selected = "pt"
            else:
                try:
                    if not path.is_file():
                        raise FileNotFoundError(f"导出模型不存在：{path}")
                    source_hash = source_hash or file_sha256(model_path)
                    if backend == "onnx":
                        metadata = read_onnx_metadata(path)
                        tracking = validate_source(metadata, source_hash)
                        if not tracking.get("dynamic"):
                            raise ValueError("需要支持全局/局部尺寸的动态 ONNX")
                        selected = "onnx"
                    else:
                        metadata = read_engine_metadata(path)
                        validate_engine(metadata, source_hash, size, batch, runtime_info(self.device))
                        selected = "tensorrt"
                    model = YOLO(str(path), task="detect")
                    self._warmup(model, size, batch)
                    if selected == "onnx":
                        # ORT 可能在 provider 初始化失败后退回 CPU，核对实际使用的 provider。
                        self._check_onnx_provider(model)
                except Exception as error:
                    # 仅包围导出模型的加载与预热；正常视频推理不捕获异常。
                    if backend != "auto":
                        raise RuntimeError(f"{scope} 的 {backend} 初始化失败：{error}") from error
                    logger.warning("%s TensorRT 不可用，回退 PT（device=%s）：%s", scope, self.device, error)
                    selected = "pt"
                    model = None
            if selected == "pt":
                if self.model is None:
                    self.model = YOLO(str(model_path), task="detect")
                model, path = self.model, model_path
            self._models[scope] = model
            self.backend_by_scope[scope] = selected
            self.model_paths[scope] = str(path)
            logger.info("%s 推理：%s，device=%s，imgsz=%s，batch=1..%s，模型=%s",
                        scope, "TensorRT FP16" if selected == "tensorrt" else selected.upper(),
                        self.device, size, batch, path)

    def _warmup(self, model: Any, size: int, batch: int) -> None:
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        for count in sorted({1, batch}):
            results = model.predict(source=[frame] * count, imgsz=size, device=self.device,
                                    rect=False, verbose=False, stream=False, agnostic_nms=True)
            if len(results) != count:
                raise RuntimeError("导出模型预热结果数量与输入不一致")

    def _check_onnx_provider(self, model: Any) -> None:
        # 固定的 Ultralytics 8.4.154 使用独立 ONNX 后端。
        session = model.predictor.model.backend.session
        providers = session.get_providers()
        if self.device != "cpu" and "CUDAExecutionProvider" not in providers:
            raise RuntimeError(f"ONNX CUDA provider 初始化失败，实际 providers={providers}；请使用 --device cpu 或修复 ORT")
        logger.info("ONNX Runtime 实际 providers=%s", providers)

    @staticmethod
    def _select_device(device: str | None) -> str:
        """选择推理设备，并在初始化时打印设备或 CPU 回退原因。"""
        import torch

        requested = (device or "auto").strip().lower() or "auto"
        if requested == "cpu":
            logger.info("推理设备：CPU（手动指定）")
            return "cpu"

        automatic = requested == "auto"
        cuda_requested = (
            automatic
            or requested == "cuda"
            or requested.startswith("cuda:")
            or all(part.strip().isdigit() for part in requested.split(","))
        )
        if cuda_requested:
            if not torch.cuda.is_available():
                logger.warning("未检测到可用的 CUDA GPU，自动回退到 CPU 推理")
                return "cpu"
            selected = "0" if automatic or requested == "cuda" else requested.removeprefix("cuda:")
            logger.info("推理设备：GPU（CUDA，device=%s）", selected)
            return selected

        logger.info("推理设备：%s（手动指定）", requested)
        return requested

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
        scope = "global" if isinstance(source, list) else "local"
        expected_size = self.image_size if scope == "global" else self.local_image_size
        size = image_size or expected_size
        count = len(source) if isinstance(source, list) else 1
        if self.backend_by_scope[scope] == "tensorrt":
            if size != expected_size or count > (self.global_batch_size if scope == "global" else 1):
                raise ValueError("请求的输入尺寸或批量超出已校验的 TensorRT 范围")
        predict_arguments: dict[str, Any] = {
            "source": source,
            "conf": self.confidence,
            "iou": self.iou_threshold,
            "imgsz": size,
            "verbose": False,
            "agnostic_nms": True,
            "stream": False,
            "device": self.device,
            "rect": False,
        }
        return self._models[scope].predict(**predict_arguments)

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
                    class_name=YoloPersonDetector._class_name(result.names, int(class_id)),
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
