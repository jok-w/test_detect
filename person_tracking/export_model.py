"""PT → 动态 ONNX → Jetson TensorRT 10 FP16 双 engine。"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .model_artifacts import (
    SCHEMA_VERSION, artifact_paths, file_sha256, read_onnx_metadata, runtime_info, validate_source,
)


logger = logging.getLogger(__name__)
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "best.pt"


def check_size(size: int, batch: int) -> None:
    if size < 32 or size % 32 or batch < 1:
        raise ValueError("输入尺寸必须是正的 32 倍数，batch 必须大于 0")


def export_onnx(model_path: Path, output: Path, image_size: int = 640, batch: int = 4,
                opset: int = 17) -> Path:
    """使用临时目录导出 FP32 ONNX，验证图并保留原 PT，不覆盖已有 ONNX 直到成功。"""
    import onnx
    import onnxruntime  # 预先检查依赖，避免 Ultralytics 导出中途自动安装。
    import onnxslim
    import ultralytics
    from ultralytics import YOLO

    check_size(image_size, batch)
    if model_path.suffix.lower() != ".pt" or not model_path.is_file():
        raise ValueError(f"需要有效的 PT 模型：{model_path}")
    if output.suffix.lower() != ".onnx":
        raise ValueError("ONNX 输出文件必须使用 .onnx 后缀")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="onnx-export-", dir=output.parent) as directory:
        source = Path(directory) / model_path.name
        shutil.copy2(model_path, source)
        model = YOLO(str(source), task="detect")
        if model.task != "detect" or getattr(model.model, "end2end", False):
            raise ValueError("当前导出工具仅支持普通 YOLO 检测模型（如 YOLO11 三姿态模型）")
        exported = Path(model.export(format="onnx", imgsz=image_size, batch=batch, dynamic=True,
                                     simplify=True, opset=opset, nms=False, device="cpu"))
        graph = onnx.load(str(exported))
        entry = graph.metadata_props.add()
        entry.key = "person_tracking"
        entry.value = json.dumps({
            "schema": SCHEMA_VERSION, "source_sha256": file_sha256(model_path),
            "source_name": model_path.name, "precision": "fp32", "dynamic": True,
            "ultralytics": ultralytics.__version__, "created_at": datetime.now(timezone.utc).isoformat(),
        })
        onnx.checker.check_model(graph)
        onnx.save_model(graph, str(exported), save_as_external_data=False)
        # 验证动态 batch 和空间尺寸，防止仅修改符号而图内部仍固定形状。
        session = onnxruntime.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
        for shape in ((1, 3, 384, 384), (batch, 3, image_size, image_size)):
            outputs = session.run(None, {session.get_inputs()[0].name: np.zeros(shape, dtype=np.float32)})
            if not outputs or outputs[0].shape[0] != shape[0] or not np.isfinite(outputs[0]).all():
                raise RuntimeError(f"ONNX 预热结果无效：{shape}")
        del session
        exported.replace(output)
    logger.info("ONNX 导出及动态输入验证完成：%s", output)
    return output


def build_engine(onnx_path: Path, output: Path, image_size: int, max_batch: int,
                 device: str = "0", workspace: float = 2.0) -> Path:
    """由已有 ONNX 构建精确输入范围的 engine，预热成功后才发布文件。"""
    import torch
    import tensorrt as trt
    from ultralytics import YOLO

    check_size(image_size, max_batch)
    if not workspace > 0 or not np.isfinite(workspace):
        raise ValueError("workspace 必须是有限正数，单位 GiB")
    if output.suffix.lower() != ".engine":
        raise ValueError("TensorRT 输出文件必须使用 .engine 后缀")
    environment = runtime_info(device)
    if not trt.__version__.startswith("10."):
        raise ValueError("当前构建工具针对 JetPack 的 TensorRT 10.x；请使用匹配 JetPack 的运行库")
    metadata = read_onnx_metadata(onnx_path)
    tracking = metadata.get("person_tracking", {})
    if not tracking.get("source_sha256"):
        raise ValueError("ONNX 缺少 PT 指纹，请先使用本工具执行 pt-to-onnx")
    validate_source(metadata, tracking["source_sha256"])
    if not tracking.get("dynamic") or tracking.get("precision") != "fp32":
        raise ValueError("需要本工具导出的动态 FP32 ONNX")
    if tracking.get("ultralytics") != environment["ultralytics"]:
        raise ValueError("ONNX 导出和 engine 构建使用的 Ultralytics 版本不同，请统一版本")

    torch.cuda.set_device(int(device))
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(str(onnx_path.resolve())):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT 解析 ONNX 失败：\n{errors}")
    if network.num_inputs != 1 or network.get_input(0).name != "images":
        raise ValueError("只支持单输入 images 的 YOLO 检测 ONNX")
    inp = network.get_input(0)
    if tuple(inp.shape) != (-1, 3, -1, -1) or inp.dtype != trt.float32:
        raise ValueError(f"ONNX 输入应为动态 FP32 NCHW，实际为 {inp.shape} / {inp.dtype}")
    inp.shape = (-1 if max_batch > 1 else 1, 3, image_size, image_size)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace * (1 << 30)))
    if not getattr(builder, "platform_has_fast_fp16", True):
        raise ValueError("当前 GPU 不支持快速 FP16")
    config.set_flag(trt.BuilderFlag.FP16)
    if max_batch > 1:
        profile = builder.create_optimization_profile()
        profile.set_shape("images", (1, 3, image_size, image_size),
                          (max_batch, 3, image_size, image_size), (max_batch, 3, image_size, image_size))
        config.add_optimization_profile(profile)
    logger.info("构建 TensorRT FP16：%sx%s，batch=1..%s，workspace=%s GiB", image_size, image_size, max_batch, workspace)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine 构建失败；请检查解析日志、内存和 workspace")
    metadata.update(batch=max_batch, imgsz=[image_size, image_size], dynamic=max_batch > 1)
    metadata["args"] = {"dynamic": max_batch > 1, "nms": False}
    metadata["person_tracking"] = {
        **tracking, "onnx_sha256": file_sha256(onnx_path), "image_size": image_size,
        "max_batch": max_batch, "dynamic": max_batch > 1, "precision": "fp16", "runtime": environment,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="engine-build-", dir=output.parent) as directory:
        candidate = Path(directory) / output.name
        header = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
        with candidate.open("wb") as stream:
            stream.write(len(header).to_bytes(4, "little"))
            stream.write(header)
            stream.write(serialized)
        # 释放构建资源后再加载，降低 Jetson 内存峰值。
        del serialized, parser, network, config, builder
        model = YOLO(str(candidate), task="detect")
        frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        for count in sorted({1, max_batch}):
            results = model.predict(source=[frame] * count, imgsz=image_size, device=device,
                                    rect=False, verbose=False, agnostic_nms=True, stream=False)
            if len(results) != count:
                raise RuntimeError("TensorRT 预热结果数量与输入不一致")
        torch.cuda.synchronize(int(device))
        del model
        candidate.replace(output)
    logger.info("TensorRT 构建及预热成功：%s", output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    onnx = commands.add_parser("pt-to-onnx", help="导出 FP32 动态 ONNX")
    onnx.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    onnx.add_argument("--output", type=Path)
    onnx.add_argument("--imgsz", type=int, default=640)
    onnx.add_argument("--batch", type=int, default=4)
    onnx.add_argument("--opset", type=int, default=17)
    engine = commands.add_parser("onnx-to-engine", help="由已有 ONNX 构建全局、局部两个 FP16 engine")
    engine.add_argument("--onnx", type=Path, default=DEFAULT_MODEL.with_suffix(".onnx"))
    engine.add_argument("--global-engine", type=Path)
    engine.add_argument("--local-engine", type=Path)
    engine.add_argument("--global-imgsz", type=int, default=640)
    engine.add_argument("--local-imgsz", type=int, default=384)
    engine.add_argument("--batch", type=int, default=4)
    engine.add_argument("--device", default="0")
    engine.add_argument("--workspace", type=float, default=2.0, help="构建 workspace 上限，单位 GiB")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args()
    if args.command == "pt-to-onnx":
        export_onnx(args.model, args.output or args.model.with_suffix(".onnx"), args.imgsz, args.batch, args.opset)
    else:
        _, global_path, local_path = artifact_paths(args.onnx)
        global_path, local_path = args.global_engine or global_path, args.local_engine or local_path
        if global_path.resolve() == local_path.resolve():
            raise ValueError("全局和局部 engine 必须使用不同的输出路径")
        build_engine(args.onnx, global_path, args.global_imgsz, args.batch, args.device, args.workspace)
        build_engine(args.onnx, local_path, args.local_imgsz, 1, args.device, args.workspace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
