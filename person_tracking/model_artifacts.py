"""导出模型的指纹、元数据和输入范围校验；不依赖 CUDA。"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_paths(model: Path) -> tuple[Path, Path, Path]:
    return (
        model.with_suffix(".onnx"),
        model.with_name(f"{model.stem}.global.engine"),
        model.with_name(f"{model.stem}.local.engine"),
    )


def read_engine_metadata(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        size = int.from_bytes(stream.read(4), "little")
        if not 0 < size <= min(1024 * 1024, path.stat().st_size - 4):
            raise ValueError(f"engine 缺少项目元数据，请使用 person_tracking.export_model 重新构建：{path}")
        metadata = json.loads(stream.read(size))
    if not isinstance(metadata, dict):
        raise ValueError("engine 元数据必须是字典")
    return metadata


def read_onnx_metadata(path: Path) -> dict[str, Any]:
    import onnx

    graph = onnx.load(str(path), load_external_data=False)
    metadata = {}
    for item in graph.metadata_props:
        try:
            metadata[item.key] = json.loads(item.value)
        except ValueError:
            try:
                metadata[item.key] = ast.literal_eval(item.value)
            except (ValueError, SyntaxError):
                metadata[item.key] = item.value
    return metadata


def validate_source(metadata: dict, source_hash: str) -> dict:
    tracking = metadata.get("person_tracking", {})
    if not isinstance(tracking, dict) or tracking.get("schema") != SCHEMA_VERSION:
        raise ValueError("缺少受支持的导出元数据，请重新导出模型")
    if tracking.get("source_sha256") != source_hash:
        raise ValueError("导出模型与当前 PT 权重不一致，请重新导出")
    if metadata.get("task") != "detect" or not metadata.get("names"):
        raise ValueError("导出模型缺少人物检测类别信息")
    return tracking


def validate_engine(metadata: dict, source_hash: str, image_size: int, batch_size: int, runtime: dict) -> dict:
    tracking = validate_source(metadata, source_hash)
    max_batch = tracking.get("max_batch")
    if tracking.get("image_size") != image_size or not isinstance(max_batch, int) or not 1 <= batch_size <= max_batch:
        raise ValueError(f"engine 输入范围不支持 imgsz={image_size}、batch={batch_size}，请重新构建")
    if tracking.get("precision") != "fp16":
        raise ValueError("当前只支持项目导出的 FP16 engine")
    environment = tracking.get("runtime")
    if not isinstance(environment, dict):
        raise ValueError("engine 缺少构建环境信息")
    for key in ("tensorrt", "cuda", "gpu_name", "gpu_capability", "ultralytics"):
        if environment.get(key) != runtime.get(key):
            raise ValueError(f"engine 构建环境与当前 {key} 不匹配，请在目标设备重新构建")
    return tracking


def runtime_info(device: str) -> dict:
    import torch
    import tensorrt
    import ultralytics

    if not device.isdigit() or not torch.cuda.is_available():
        raise ValueError("TensorRT 需要一张可用的 CUDA GPU，例如 --device 0")
    return {
        "tensorrt": tensorrt.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(int(device)),
        "gpu_capability": list(torch.cuda.get_device_capability(int(device))),
        "ultralytics": ultralytics.__version__,
    }
