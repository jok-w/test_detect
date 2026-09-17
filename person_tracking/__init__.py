"""提供独立的视频人物检测、卡尔曼预测和动态人物裁剪能力。"""

from .processor import (
    PersonVideoProcessor,
    ProcessorConfig,
)

from .engine import PersonTrackingEngine, FrameTrackingResult
from .tracking_config import PersonTrackingConfig

__all__ = ["PersonVideoProcessor", "ProcessorConfig", "PersonTrackingEngine", "PersonTrackingConfig", "FrameTrackingResult"]
