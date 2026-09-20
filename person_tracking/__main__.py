import argparse
import logging
from pathlib import Path

from .processor import (
    PersonVideoProcessor,
    ProcessorConfig,
)


DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "best.pt"


def build_parser() -> argparse.ArgumentParser:
    """
    作用：构建独立人物视频处理命令行参数解析器。
    返回：配置完成的 argparse 参数解析器。
    """
    parser = argparse.ArgumentParser(
        description=("逐帧使用 YOLO11n 三姿态模型和卡尔曼滤波生成人物框标注视频")
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLO11n 权重路径")
    parser.add_argument("--backend", choices=("auto", "pt", "onnx", "tensorrt"), default="auto",
                        help="默认优先 TensorRT，缺失或初始化失败回退 PT；显式 tensorrt 失败则报错")
    parser.add_argument("--onnx-model", type=Path, help="动态 ONNX 路径；默认与 PT 同名")
    parser.add_argument("--global-engine", type=Path, help="默认 <PT名称>.global.engine")
    parser.add_argument("--local-engine", type=Path, help="默认 <PT名称>.local.engine")
    parser.add_argument("--input", type=Path, required=True, help="输入视频路径")
    parser.add_argument("--decoder", choices=("auto", "opencv", "gstreamer"), default="auto",
                        help="读取后端：Jetson H.264/H.265 MP4/MOV 优先 NVDEC；Windows 使用 OpenCV")
    parser.add_argument("--decode-prefetch", type=int, choices=(1, 2), default=2,
                        help="NVDEC appsink 预读队列上限，默认 2 帧，满时等待、不丢帧")
    parser.add_argument("--gst-python", default="/usr/bin/python3",
                        help="提供 JetPack GStreamer GI 的系统 Python，默认 /usr/bin/python3")
    parser.add_argument("--output", type=Path, default=None, help="输出视频路径；默认保存到 outputs 目录")
    parser.add_argument("--output-max-width", type=int, default=1920,
                        help="输出最大宽度，默认 1920；0 保留原尺寸，编码尺寸对齐偶数；识别仍用原图")
    parser.add_argument(
        "--warmup-detections",
        type=int,
        choices=(3, 4),
        default=4,
        help="启动阶段连续成功模型检测次数，默认 4",
    )
    parser.add_argument(
        "--prediction-frames",
        type=int,
        default=5,
        help="局部模型检测之间的卡尔曼纯预测帧数，默认 5；设为 0 时每帧局部检测",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=None,
                        help="整帧全局检测的正方形网络输入尺寸，默认 640；整图缩放补边后推理")
    parser.add_argument(
        "--local-imgsz",
        type=int,
        default=None,
        help="动态裁剪区域的模型输入尺寸，默认 384",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="推理设备，例如 cpu、0；默认优先 GPU，CUDA 不可用时回退 CPU 并打印提示",
    )
    parser.add_argument("--roi-y-min", type=int, default=None)
    parser.add_argument(
        "--roi-y-max",
        type=int,
        default=None,
        help="固定人物纵向区域终点；为空时使用视频底部",
    )
    parser.add_argument("--codec", default="mp4v", help="OpenCV 输出 FourCC；硬件输出固定 H.264")
    parser.add_argument("--encoder", choices=("auto", "opencv", "gstreamer"), default="auto",
                        help="视频输出后端：Jetson MP4 自动优先 NVENC；Windows 使用 OpenCV")
    parser.add_argument("--video-bitrate", type=int, default=8000000,
                        help="硬件 H.264 输出码率，单位 bps，默认 8000000")
    parser.add_argument("--max-prediction-ms", type=float, default=500.0)
    parser.add_argument("--global-recovery-ms", type=float, default=300.0)
    parser.add_argument("--recovery-misses", type=int, default=3)
    parser.add_argument("--recovery-detections", type=int, default=2)
    parser.add_argument("--crop-min-size", type=int, default=320)
    parser.add_argument("--crop-max-size", type=int, default=800)
    parser.add_argument(
        "--crop-person-height-ratio",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--crop-person-width-ratio",
        type=float,
        default=0.35,
    )
    parser.add_argument("--crop-center-smoothing", type=float, default=0.30)
    parser.add_argument("--crop-size-smoothing", type=float, default=0.20)
    parser.add_argument(
        "--crop-max-size-change-ratio",
        type=float,
        default=0.10,
    )
    parser.add_argument("--crop-edge-margin-ratio", type=float, default=0.08)
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="每处理完成一帧立即显示；使用 --no-display 可关闭窗口",
    )
    return parser


def build_config(
    arguments: argparse.Namespace,
) -> ProcessorConfig:
    """
    作用：将离线视频命令行参数转换为处理配置。
    参数：
        arguments：argparse 解析得到的命令行参数。
    返回：可以交给人物视频处理器的完整配置。
    """
    input_path = arguments.input
    output_path = arguments.output or Path("outputs") / f"{input_path.stem}-tracked.mp4"
    return ProcessorConfig(
        model_path=arguments.model,
        backend=arguments.backend,
        onnx_path=arguments.onnx_model,
        global_engine_path=arguments.global_engine,
        local_engine_path=arguments.local_engine,
        input_path=input_path,
        decoder=arguments.decoder,
        decode_prefetch=arguments.decode_prefetch,
        gst_python=arguments.gst_python,
        output_path=output_path,
        output_max_width=arguments.output_max_width,
        warmup_detections=arguments.warmup_detections,
        prediction_frames=arguments.prediction_frames,
        confidence=arguments.confidence,
        iou_threshold=arguments.iou,
        image_size=arguments.imgsz if arguments.imgsz is not None else 640,
        local_image_size=arguments.local_imgsz if arguments.local_imgsz is not None else 384,
        device=arguments.device,
        roi_y_min=arguments.roi_y_min if arguments.roi_y_min is not None else 0,
        roi_y_max=arguments.roi_y_max,
        codec=arguments.codec,
        encoder=arguments.encoder,
        video_bitrate=arguments.video_bitrate,
        max_prediction_ms=arguments.max_prediction_ms,
        global_recovery_ms=arguments.global_recovery_ms,
        recovery_misses=arguments.recovery_misses,
        recovery_detections=arguments.recovery_detections,
        crop_min_size=arguments.crop_min_size,
        crop_max_size=arguments.crop_max_size,
        crop_person_height_ratio=arguments.crop_person_height_ratio,
        crop_person_width_ratio=arguments.crop_person_width_ratio,
        crop_center_smoothing=arguments.crop_center_smoothing,
        crop_size_smoothing=arguments.crop_size_smoothing,
        crop_max_size_change_ratio=arguments.crop_max_size_change_ratio,
        crop_edge_margin_ratio=arguments.crop_edge_margin_ratio,
        display=arguments.display,
    )


def main() -> int:
    """
    作用：解析命令行参数并执行独立人物视频处理任务。
    返回：处理成功时返回进程退出码 0。
    副作用：加载模型、读取输入视频并生成标注输出视频。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = build_parser()
    arguments = parser.parse_args()
    config = build_config(arguments)
    stats = PersonVideoProcessor(config).process()
    logging.getLogger(__name__).info(
        "视频处理完成：处理帧数=%s，写入帧数=%s，模型帧数=%s，"
        "全局模型帧数=%s，局部模型帧数=%s，卡尔曼预测帧数=%s，"
        "无人物框帧数=%s，平均单帧耗时=%.2fms，"
        "等效处理速度=%.2ffps，用户提前停止=%s，输出=%s",
        stats.total_frames,
        stats.written_frames,
        stats.model_frames,
        stats.global_model_frames,
        stats.local_model_frames,
        stats.kalman_only_frames,
        stats.frames_without_box,
        stats.average_frame_time_ms,
        stats.average_processing_fps,
        stats.stopped_by_user,
        stats.output_path,
    )
    logging.getLogger(__name__).info(
        "读取器收尾=%.2fms，编码器收尾及校验=%.2fms，含收尾实际处理速度=%.2ffps（不含初始化）",
        stats.reader_finalize_time_ms,
        stats.encoder_finalize_time_ms,
        stats.processing_fps_with_finalize,
    )
    logging.getLogger(__name__).info(
        "平均分项耗时：读取=%.2fms，检测跟踪=%.2fms，缩放绘图=%.2fms，"
        "显示=%.2fms，编码写入=%.2fms（不含初始化与编码器收尾）",
        stats.average_read_time_ms,
        stats.average_tracking_time_ms,
        stats.average_drawing_time_ms,
        stats.average_display_time_ms,
        stats.average_encoding_time_ms,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
