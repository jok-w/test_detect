from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class PersonTrackingConfig:
    """保存离线和实时人物识别共用的模型、区域和跟踪参数。"""

    model_path: Path
    warmup_detections: int = 4
    prediction_frames: int = 0
    confidence: float = 0.25
    iou_threshold: float = 0.45
    image_size: int = 640
    local_image_size: int = 384
    device: str | None = None
    roi_y_min: int = 0
    roi_y_max: int | None = None
    max_prediction_ms: float = 500.0
    global_recovery_ms: float = 300.0
    recovery_misses: int = 3
    recovery_detections: int = 2
    crop_min_size: int = 320
    crop_max_size: int = 800
    crop_person_height_ratio: float = 0.60
    crop_person_width_ratio: float = 0.35
    crop_center_smoothing: float = 0.30
    crop_size_smoothing: float = 0.20
    crop_max_size_change_ratio: float = 0.10
    crop_edge_margin_ratio: float = 0.08

    def validate(self) -> None:
        """
        作用：校验人物识别模型路径、识别区域和跟踪参数。
        返回：无。
        异常：
            参数无效、文件缺失或输入输出路径相同时抛出 ValueError。
        """
        if not self.model_path.is_file():
            raise ValueError(f"模型文件不存在：{self.model_path}")
        if self.warmup_detections not in {3, 4}:
            raise ValueError("预热连续成功检测次数只能为 3 或 4")
        if self.prediction_frames < 0:
            raise ValueError("模型校正之间的卡尔曼预测帧数不能小于 0")
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("模型置信度阈值必须位于 0.0～1.0")
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ValueError("模型 IoU 阈值必须位于 0.0～1.0")
        if self.image_size <= 0:
            raise ValueError("全局模型输入尺寸必须大于 0")
        if self.local_image_size <= 0:
            raise ValueError("局部模型输入尺寸必须大于 0")
        if self.roi_y_min < 0:
            raise ValueError("纵向裁剪起点不能小于 0")
        if self.roi_y_max is not None and self.roi_y_max <= self.roi_y_min:
            raise ValueError("纵向裁剪终点必须大于裁剪起点")
        if self.max_prediction_ms <= 0.0:
            raise ValueError("卡尔曼纯预测最大可靠时间必须大于 0")
        if self.global_recovery_ms <= 0.0:
            raise ValueError("切换全局恢复搜索的时间必须大于 0")
        if self.global_recovery_ms > self.max_prediction_ms:
            raise ValueError("切换全局恢复搜索的时间不能大于预测框最大可靠时间")
        if self.recovery_misses < 1:
            raise ValueError("切换全局搜索前的局部漏检次数必须大于 0")
        if self.recovery_detections < 1:
            raise ValueError("恢复局部跟踪所需连续检测次数必须大于 0")
        if self.crop_min_size <= 0:
            raise ValueError("动态裁剪最小边长必须大于 0")
        if self.crop_max_size < self.crop_min_size:
            raise ValueError("动态裁剪最大边长不能小于最小边长")
        if not 0.0 < self.crop_person_height_ratio <= 1.0:
            raise ValueError("人物目标高度占比必须位于 0.0～1.0")
        if not 0.0 < self.crop_person_width_ratio <= 1.0:
            raise ValueError("人物目标宽度占比必须位于 0.0～1.0")
        if not 0.0 < self.crop_center_smoothing <= 1.0:
            raise ValueError("裁剪中心平滑系数必须位于 0.0～1.0")
        if not 0.0 < self.crop_size_smoothing <= 1.0:
            raise ValueError("裁剪尺寸平滑系数必须位于 0.0～1.0")
        if not 0.0 < self.crop_max_size_change_ratio < 1.0:
            raise ValueError("裁剪尺寸单帧最大变化比例必须位于 0.0～1.0")
        if not 0.0 <= self.crop_edge_margin_ratio < 0.5:
            raise ValueError("裁剪边缘预警比例必须位于 0.0～0.5")

