"""在同一视频上比较 PT 与 ONNX/TensorRT 的检测、跟踪耗时与结果。"""
from __future__ import annotations

import argparse
import gc
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .__main__ import build_config, build_parser as tracking_parser
from .engine import PersonTrackingEngine
from .model_artifacts import artifact_paths
from .types import BoundingBox, bbox_iou


def timing_summary(values: list[float]) -> dict:
    return {"count": len(values), "mean_ms": float(np.mean(values)) if values else None,
            "p95_ms": float(np.percentile(values, 95)) if values else None}


def run_video(config, limit: int, warmup: int) -> dict:
    import torch

    engine = PersonTrackingEngine(config)
    detector = engine.detector
    device = detector.device
    use_cuda = device != "cpu" and device.split(",")[0].isdigit()

    def synchronize():
        if use_cuda:
            torch.cuda.synchronize(int(device.split(",")[0]))

    records = []
    capture = None
    try:
        if warmup:
            # PT 可能在预热视频里一直未找到人物，也应预热局部输入尺寸。
            detector._warmup(detector._models["global"], config.image_size, config.global_tile_batch_size)
            detector._warmup(detector._models["local"], config.local_image_size, 1)
        for warming, count in ((True, warmup), (False, limit)):
            if not count:
                continue
            capture = cv2.VideoCapture(str(config.input_path))
            if not capture.isOpened():
                raise RuntimeError(f"无法读取视频：{config.input_path}")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0:
                raise RuntimeError("输入视频缺少有效帧率")
            engine.reset()
            for index in range(count):
                synchronize()
                started = perf_counter()
                ok, frame = capture.read()
                if not ok:
                    break
                inference_started = perf_counter()
                result, used_model = engine.process_frame(frame, index * 1000.0 / fps)
                synchronize()
                finished = perf_counter()
                if not warming:
                    records.append({"frame": index, "processing_ms": (finished - inference_started) * 1000,
                                    "read_and_processing_ms": (finished - started) * 1000,
                                    "used_model": used_model, **asdict(result)})
            capture.release()
            capture = None
        if not records:
            raise RuntimeError("视频没有可测量的帧")
        processing = [r["processing_ms"] for r in records]
        read_processing = [r["read_and_processing_ms"] for r in records]
        return {
            "requested_backend": config.backend, "actual_backends": detector.backend_by_scope,
            "models": detector.model_paths, "device": device, "frames": len(records),
            "processing": timing_summary(processing), "read_and_processing": timing_summary(read_processing),
            "fps_without_drawing_encoding": 1000.0 / float(np.mean(read_processing)),
            "global_detection_frames": timing_summary([r["processing_ms"] for r in records
                                                         if r["used_model"] and r["model_scope"] == "global"]),
            "local_detection_frames": timing_summary([r["processing_ms"] for r in records
                                                        if r["used_model"] and r["model_scope"] == "local"]),
            "records": records,
        }
    finally:
        if capture is not None:
            capture.release()
        engine.close()
        del detector
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()


def compare_results(baseline: list[dict], candidate: list[dict]) -> dict:
    if len(baseline) != len(candidate):
        raise ValueError("两次运行的视频帧数不同，不能直接比较")
    ious, class_matches = [], []
    presence_mismatches = 0
    for left, right in zip(baseline, candidate):
        if (left["bbox"] is None) != (right["bbox"] is None):
            presence_mismatches += 1
        elif left["bbox"] is not None:
            ious.append(bbox_iou(BoundingBox(**left["bbox"]), BoundingBox(**right["bbox"])))
            class_matches.append(left["class_name"] == right["class_name"])
    return {"both_have_box_frames": len(ious), "presence_mismatch_frames": presence_mismatches,
            "mean_box_iou": float(np.mean(ious)) if ious else None,
            "class_agreement": float(np.mean(class_matches)) if class_matches else None}


def build_parser() -> argparse.ArgumentParser:
    parser = tracking_parser()
    parser.description = __doc__
    parser.add_argument("--candidate", choices=("onnx", "tensorrt"), default="tensorrt")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--benchmark-warmup", type=int, default=8)
    parser.add_argument("--report", type=Path, default=Path("outputs/benchmark.json"))
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args()
    if args.frames < 1 or args.benchmark_warmup < 0:
        raise ValueError("frames 必须大于 0，benchmark-warmup 不能小于 0")
    config = build_config(args)
    model_files = [config.model_path, *artifact_paths(config.model_path), config.onnx_path,
                   config.global_engine_path, config.local_engine_path]
    if args.report.resolve() in {p.resolve() for p in [config.input_path, *model_files] if p is not None}:
        raise ValueError("报告不能覆盖输入视频或模型")
    baseline = run_video(replace(config, backend="pt", display=False), args.frames, args.benchmark_warmup)
    candidate = run_video(replace(config, backend=args.candidate, display=False), args.frames, args.benchmark_warmup)
    report = {
        "note": "相同视频逐帧检测/跟踪比较；不含模型初始化、预热、绘制、显示和编码；不代表标注集精度评估。",
        "comparison": compare_results(baseline["records"], candidate["records"]),
        "processing_speedup": baseline["processing"]["mean_ms"] / candidate["processing"]["mean_ms"],
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "pt": baseline, "candidate": candidate,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("处理阶段加速比=%.3f，结果对比=%s，报告=%s", report["processing_speedup"], report["comparison"], args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
