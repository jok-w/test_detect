from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class BoundingBox:
    """表示原始视频画面坐标系中的人物矩形框。"""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """
        作用：计算人物框宽度。
        返回：非负的像素宽度。
        """
        return max(self.x2 - self.x1, 0.0)

    @property
    def height(self) -> float:
        """
        作用：计算人物框高度。
        返回：非负的像素高度。
        """
        return max(self.y2 - self.y1, 0.0)

    @property
    def center(self) -> tuple[float, float]:
        """
        作用：计算人物框中心坐标。
        返回：人物框中心的横向和纵向像素坐标。
        """
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def diagonal(self) -> float:
        """
        作用：计算人物框对角线长度。
        返回：人物框对角线像素长度。
        """
        return hypot(self.width, self.height)

    def shifted(self, x_offset: float = 0.0, y_offset: float = 0.0) -> BoundingBox:
        """
        作用：将人物框从裁剪区域坐标恢复到原始画面坐标。
        参数：
            x_offset：横向坐标偏移量，单位为像素。
            y_offset：纵向坐标偏移量，单位为像素。
        返回：应用偏移后的新人物框。
        """
        return BoundingBox(
            x1=self.x1 + x_offset,
            y1=self.y1 + y_offset,
            x2=self.x2 + x_offset,
            y2=self.y2 + y_offset,
        )

    def clamped(self, frame_width: int, frame_height: int) -> BoundingBox:
        """
        作用：将人物框限制在视频画面范围内。
        参数：
            frame_width：原始视频宽度，单位为像素。
            frame_height：原始视频高度，单位为像素。
        返回：限制边界后的新人物框。
        """
        return BoundingBox(
            x1=min(max(self.x1, 0.0), float(frame_width - 1)),
            y1=min(max(self.y1, 0.0), float(frame_height - 1)),
            x2=min(max(self.x2, 0.0), float(frame_width - 1)),
            y2=min(max(self.y2, 0.0), float(frame_height - 1)),
        )

    def as_int_xyxy(self) -> tuple[int, int, int, int]:
        """
        作用：将浮点人物框转换为 OpenCV 使用的整数坐标。
        返回：按 x1、y1、x2、y2 排列的整数坐标。
        """
        return (
            int(round(self.x1)),
            int(round(self.y1)),
            int(round(self.x2)),
            int(round(self.y2)),
        )


@dataclass(frozen=True)
class PersonDetection:
    """表示模型在单帧中输出的一个人物姿态检测结果。"""

    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str


@dataclass(frozen=True)
class CropWindow:
    """表示原始视频坐标系中的整数裁剪窗口。"""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        """
        作用：计算裁剪窗口宽度。
        返回：非负的像素宽度。
        """
        return max(self.x2 - self.x1, 0)

    @property
    def height(self) -> int:
        """
        作用：计算裁剪窗口高度。
        返回：非负的像素高度。
        """
        return max(self.y2 - self.y1, 0)

    def contains_valid_area(self) -> bool:
        """
        作用：判断裁剪窗口是否包含可供模型处理的像素区域。
        返回：宽度和高度都大于零时返回 True，否则返回 False。
        """
        return self.width > 0 and self.height > 0


def bbox_iou(first: BoundingBox, second: BoundingBox) -> float:
    """
    作用：计算两个人物框的交并比。
    参数：
        first：第一个人物框。
        second：第二个人物框。
    返回：范围为 0.0～1.0 的交并比。
    """
    intersection_x1 = max(first.x1, second.x1)
    intersection_y1 = max(first.y1, second.y1)
    intersection_x2 = min(first.x2, second.x2)
    intersection_y2 = min(first.y2, second.y2)
    intersection_width = max(intersection_x2 - intersection_x1, 0.0)
    intersection_height = max(intersection_y2 - intersection_y1, 0.0)
    intersection_area = intersection_width * intersection_height
    union_area = first.width * first.height + second.width * second.height
    union_area -= intersection_area
    if union_area <= 0.0:
        return 0.0
    return intersection_area / union_area
