from __future__ import annotations

import numpy as np

from .types import BoundingBox


class BoundingBoxKalmanFilter:
    """使用匀速运动模型预测单个人物矩形框。"""

    _STATE_SIZE = 8
    _MEASUREMENT_SIZE = 4

    def __init__(
        self,
        process_variance: float = 300.0,
        center_measurement_variance: float = 16.0,
        size_measurement_variance: float = 36.0,
    ) -> None:
        """
        作用：初始化人物框卡尔曼滤波参数和空状态。
        参数：
            process_variance：人员运动变化的过程噪声强度。
            center_measurement_variance：模型中心坐标测量噪声方差。
            size_measurement_variance：模型人物框尺寸测量噪声方差。
        返回：无。
        """
        self.process_variance = float(process_variance)
        self._measurement_matrix = np.zeros(
            (self._MEASUREMENT_SIZE, self._STATE_SIZE),
            dtype=np.float64,
        )
        self._measurement_matrix[:, : self._MEASUREMENT_SIZE] = np.eye(
            self._MEASUREMENT_SIZE,
            dtype=np.float64,
        )
        self._measurement_noise = np.diag(
            [
                center_measurement_variance,
                center_measurement_variance,
                size_measurement_variance,
                size_measurement_variance,
            ]
        ).astype(np.float64)
        self._state: np.ndarray | None = None
        self._covariance: np.ndarray | None = None
        self._last_timestamp_ms: float | None = None

    @property
    def initialized(self) -> bool:
        """
        作用：判断卡尔曼滤波器是否已经取得首个人物框。
        返回：存在可预测状态时返回 True，否则返回 False。
        """
        return self._state is not None and self._covariance is not None

    def initialize(self, bbox: BoundingBox, timestamp_ms: float) -> BoundingBox:
        """
        作用：使用第一帧模型人物框初始化位置、尺寸和速度状态。
        参数：
            bbox：第一帧模型检测得到的人物框。
            timestamp_ms：检测帧的视频毫秒时间戳。
        返回：初始化后的当前人物框。
        """
        center_x, center_y = bbox.center
        self._state = np.array(
            [
                center_x,
                center_y,
                max(bbox.width, 2.0),
                max(bbox.height, 2.0),
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        self._covariance = np.diag(
            [16.0, 16.0, 36.0, 36.0, 2500.0, 2500.0, 900.0, 900.0]
        ).astype(np.float64)
        self._last_timestamp_ms = float(timestamp_ms)
        return self.current_bbox()

    def predict(self, timestamp_ms: float) -> BoundingBox:
        """
        作用：根据当前帧时间戳将人物框预测到新的时间位置。
        参数：
            timestamp_ms：当前视频帧的毫秒时间戳。
        返回：当前帧的卡尔曼预测人物框。
        异常：
            滤波器未初始化时抛出 RuntimeError。
        副作用：推进内部状态、协方差和最近时间戳。
        """
        state, covariance = self._require_state()
        if self._last_timestamp_ms is None:
            raise RuntimeError("卡尔曼滤波器缺少初始化时间戳")
        elapsed_ms = max(float(timestamp_ms) - self._last_timestamp_ms, 0.001)
        delta_seconds = elapsed_ms / 1000.0
        transition = self._build_transition(delta_seconds)
        process_noise = self._build_process_noise(delta_seconds)
        self._state = transition @ state
        self._state[2:4] = np.maximum(self._state[2:4], 2.0)
        self._covariance = transition @ covariance @ transition.T + process_noise
        self._last_timestamp_ms = float(timestamp_ms)
        return self.current_bbox()

    def correct(self, bbox: BoundingBox) -> BoundingBox:
        """
        作用：使用当前帧模型人物框修正预测位置、尺寸和速度。
        参数：
            bbox：当前帧模型检测得到的人物框。
        返回：模型测量修正后的平滑人物框。
        异常：
            滤波器未初始化时抛出 RuntimeError。
        副作用：更新内部状态和协方差。
        """
        state, covariance = self._require_state()
        center_x, center_y = bbox.center
        measurement = np.array(
            [center_x, center_y, max(bbox.width, 2.0), max(bbox.height, 2.0)],
            dtype=np.float64,
        )
        innovation = measurement - self._measurement_matrix @ state
        innovation_covariance = (
            self._measurement_matrix @ covariance @ self._measurement_matrix.T
            + self._measurement_noise
        )
        projected_covariance = covariance @ self._measurement_matrix.T
        kalman_gain = np.linalg.solve(
            innovation_covariance.T,
            projected_covariance.T,
        ).T
        self._state = state + kalman_gain @ innovation
        self._state[2:4] = np.maximum(self._state[2:4], 2.0)
        identity = np.eye(self._STATE_SIZE, dtype=np.float64)
        residual_transform = identity - kalman_gain @ self._measurement_matrix
        self._covariance = (
            residual_transform @ covariance @ residual_transform.T
            + kalman_gain @ self._measurement_noise @ kalman_gain.T
        )
        return self.current_bbox()

    def current_bbox(self) -> BoundingBox:
        """
        作用：将当前卡尔曼状态转换为人物矩形框。
        返回：当前状态对应的人物框。
        异常：
            滤波器未初始化时抛出 RuntimeError。
        """
        state, _ = self._require_state()
        center_x, center_y, width, height = state[:4]
        half_width = max(width, 2.0) / 2.0
        half_height = max(height, 2.0) / 2.0
        return BoundingBox(
            x1=float(center_x - half_width),
            y1=float(center_y - half_height),
            x2=float(center_x + half_width),
            y2=float(center_y + half_height),
        )

    def reset(self) -> None:
        """
        作用：清除人物跟踪状态并恢复为等待首次检测。
        返回：无。
        副作用：丢弃当前卡尔曼状态、协方差和时间戳。
        """
        self._state = None
        self._covariance = None
        self._last_timestamp_ms = None

    def _require_state(self) -> tuple[np.ndarray, np.ndarray]:
        """
        作用：取得已经初始化的卡尔曼状态和协方差。
        返回：当前状态向量和协方差矩阵。
        异常：
            滤波器未初始化时抛出 RuntimeError。
        """
        if self._state is None or self._covariance is None:
            raise RuntimeError("卡尔曼滤波器尚未初始化")
        return self._state, self._covariance

    @classmethod
    def _build_transition(cls, delta_seconds: float) -> np.ndarray:
        """
        作用：根据相邻视频帧的真实时间差构造匀速状态转移矩阵。
        参数：
            delta_seconds：相邻处理帧的秒级时间差。
        返回：八维人物框状态转移矩阵。
        """
        transition = np.eye(cls._STATE_SIZE, dtype=np.float64)
        transition[: cls._MEASUREMENT_SIZE, cls._MEASUREMENT_SIZE :] = (
            np.eye(cls._MEASUREMENT_SIZE, dtype=np.float64) * delta_seconds
        )
        return transition

    def _build_process_noise(self, delta_seconds: float) -> np.ndarray:
        """
        作用：根据帧时间差构造允许人员加速和尺寸变化的过程噪声。
        参数：
            delta_seconds：相邻处理帧的秒级时间差。
        返回：八维人物框过程噪声矩阵。
        """
        delta_squared = delta_seconds**2
        delta_cubed = delta_seconds**3
        delta_fourth = delta_seconds**4
        process_noise = np.zeros(
            (self._STATE_SIZE, self._STATE_SIZE),
            dtype=np.float64,
        )
        position_slice = slice(0, self._MEASUREMENT_SIZE)
        velocity_slice = slice(self._MEASUREMENT_SIZE, self._STATE_SIZE)
        identity = np.eye(self._MEASUREMENT_SIZE, dtype=np.float64)
        process_noise[position_slice, position_slice] = identity * delta_fourth / 4.0
        process_noise[position_slice, velocity_slice] = identity * delta_cubed / 2.0
        process_noise[velocity_slice, position_slice] = identity * delta_cubed / 2.0
        process_noise[velocity_slice, velocity_slice] = identity * delta_squared
        return process_noise * self.process_variance
