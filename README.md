# 离线人物识别实验

这里是从 `touzhi_service` 复制出来的独立离线实验程序。它逐帧读取视频，使用三姿态 YOLO 模型检测人物，并结合卡尔曼滤波、动态裁剪和重新捕获策略，生成带标注的输出视频。此目录不依赖原仓库、实时相机进程或 Web 服务。

## 安装与运行

在 Jetson 的终端中进入本目录并安装依赖：

```bash
cd /home/mtr/Desktop/test_detect
uv sync --python 3.10
```

`pyproject.toml` 按当前 Jetson 部署配置限定 Linux aarch64、Python 3.10 和 NumPy 1.x，固定 `torch 2.8.0`、`torchvision 0.23.0`，并使用 Jetson AI Lab 的 `jp6/cu126` 索引。部署后可检查安装结果：

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

运行时默认检查 `torch.cuda.is_available()`：CUDA 可用时显式使用首张 GPU（`device=0`），不可用时自动回退 CPU。启动日志会打印实际选择的设备；回退时打印“未检测到可用的 CUDA GPU，自动回退到 CPU 推理”。`--device cpu` 可强制使用 CPU；`--device 0` 显式请求 GPU 时，也会在 CUDA 不可用时回退 CPU。单帧和批量分片推理使用同一设备。

此依赖配置用于 JetPack 6 / CUDA 12.6 部署环境。其他 JetPack 版本需要重新匹配依赖；当前配置不包含 Windows 安装环境。

## PT → ONNX → TensorRT FP16

推理后端默认 `--backend auto`：有 CUDA 时分别尝试全局、局部 TensorRT engine；文件缺失、权重指纹或环境不匹配、初始化/预热失败时，该场景回退 PT GPU 并打印原因。没有 CUDA 或指定 `--device cpu` 时使用 PT CPU。`--backend pt` 强制原始模型；`--backend tensorrt` 强制 engine，初始化失败直接报错。视频推理中的普通异常不会被自动回退掩盖。

`--model` 始终指定原始 PT，作为类别/权重一致性校验和回退依据。默认导出路径为：

| 文件 | 用途 | 默认输入 |
|---|---|---|
| `models/best.pt` | 原始权重及 GPU/CPU 回退 | 按推理参数 |
| `models/best.onnx` | 动态 FP32 中间模型、ONNX 验证 | 动态 batch 和空间尺寸 |
| `models/best.global.engine` | 全局分片 FP16 | 640×640，batch 1～4，优化 batch=4 |
| `models/best.local.engine` | 局部跟踪 FP16 | 384×384，batch=1 |

安装导出依赖（保留 Jetson torch/torchvision 配置，Ultralytics 固定为本项目验证的 8.4.154）：

```bash
uv sync --python 3.10 --extra export
```

先导出 ONNX。工具保留三姿态类别、PT SHA256 指纹，执行 ONNX 图检查和 CPU 动态输入预热，成功后才替换目标文件：

```bash
uv run --extra export python -m person_tracking.export_model pt-to-onnx \
  --model models/best.pt --output models/best.onnx
```

然后在**目标 Jetson** 构建 engine。需要该 JetPack 配套的 TensorRT **10.x**，不通过 PyPI 自动安装或升级 TensorRT。先确认运行项目的 Python 环境可以导入它：

```bash
uv run --extra export python -c "import torch, tensorrt; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), tensorrt.__version__)"
uv run --extra export python -m person_tracking.export_model onnx-to-engine \
  --onnx models/best.onnx --device 0 --workspace 2
```

若系统 Python 可导入 TensorRT，而 uv/Conda 环境不能，请将 JetPack 提供的、匹配 Python 3.10 的 TensorRT Python bindings 配置到项目解释器可见路径；使用与系统兼容的虚拟环境，或按设备上实际 bindings 路径配置 `PYTHONPATH`。不要用通用桌面 CUDA wheel 替代 Jetson 运行库。`workspace` 单位为 GiB，是构建器工作区上限，并非模型运行时总内存限制；内存不足可先降为 `--workspace 1`。

两个 engine 分别构建并预热，通过后才替换各自目标文件；若第二个构建失败，第一个已完成的文件仍可使用。engine 内置 Ultralytics 元数据头，不能直接当作裸 TensorRT plan 交给 `trtexec`。记录并校验 TensorRT、CUDA（PyTorch 报告）、GPU 名称/计算能力及 Ultralytics 版本，环境变化时重新构建。当前工具面向 YOLO11 普通检测模型，使用 FP16 层优化、FP32 输入输出，保留现有 NMS 和跨分片去重。

构建后运行：

```bash
uv run python -m person_tracking --input /path/to/video.mp4 --backend auto --no-display
# 强制 TensorRT，确认运行没有回退
uv run python -m person_tracking --input /path/to/video.mp4 --backend tensorrt --device 0 --no-display
# 使用 ONNX CPU 对照；默认采用 --model 对应的同名 .onnx
uv run --extra export python -m person_tracking --input /path/to/video.mp4 --backend onnx --device cpu --no-display
```

可使用 `--onnx-model`、`--global-engine`、`--local-engine` 指定导出文件。ONNX 使用两个独立预测器，默认 export 依赖安装的是 CPU ONNX Runtime；只有已安装的 ORT 提供 CUDA provider 时才尝试 ONNX GPU，并打印实际 providers。TensorRT 不依赖 ONNX Runtime GPU 包。

需要改变尺寸或批量时，重新构建并使用相同推理参数，例如：

```bash
uv run --extra export python -m person_tracking.export_model onnx-to-engine \
  --onnx models/best.onnx --global-imgsz 640 --local-imgsz 384 --batch 2
uv run python -m person_tracking --input /path/to/video.mp4 --backend tensorrt \
  --imgsz 640 --local-imgsz 384 --global-tile-batch-size 2 --no-display
```

全局末批不足最大 batch 时直接按实际数量运行。预处理统一 `rect=False`，将原图裁剪缩放、补边至对应正方形输入；原始裁剪尺寸变化无需重建 engine。尺寸必须为 32 的倍数。

## 性能与结果对照

在同一视频上顺序运行 PT 和 TensorRT，预热后报告平均/P95 延迟、全局/局部模型帧处理耗时、结果框 IoU、类别一致率和检测框有无差异：

```bash
uv run python -m person_tracking.benchmark --input /path/to/video.mp4 \
  --candidate tensorrt --device 0 --frames 300 --benchmark-warmup 8 \
  --report outputs/benchmark-trt.json
```

也可以用 `--candidate onnx --device cpu` 配合 `uv run --extra export` 验证 ONNX。报告写明实际后端和设备；TensorRT 对照强制使用 engine，失败会报错。测试不显示、绘制或编码视频，报告的 FPS 包含视频读取和检测/跟踪，不是完整输出视频的端到端 FPS；结果框比较包含跟踪产生的框，不代表标注集上的 mAP。没有进入局部跟踪的测试，其局部统计为 null，应改用包含人物、遮挡和重捕获的代表性视频。

完整读写流程的速度请另外使用普通 `person_tracking --backend pt/tensorrt --no-display` 处理同一视频，比较已有完成日志。保持设备功耗模式、温度、阈值、输入尺寸和批量一致；本地 CPU 验证不能代表 Jetson TensorRT 提速。

处理任意本地视频。`models/best.pt` 已随实验目录复制，默认输出为 `outputs/输入文件名-tracked.mp4`：

```bash
uv run python -m person_tracking --input '/path/to/video.mp4'
```

处理时默认实时显示已标注画面；按 `Q` 或 `Esc` 可提前结束。没有桌面窗口时使用 `--no-display`：

```bash
uv run python -m person_tracking --input '/path/to/video.mp4' --no-display
```

可以指定输出、其他模型和推理设备：

```bash
uv run python -m person_tracking \
  --input '/path/to/video.mp4' \
  --output 'outputs/trial-01.mp4' \
  --model 'models/best.pt' \
  --device cpu \
  --no-display
```

## 调整策略

所有处理参数可用 `uv run python -m person_tracking --help` 查看。常用参数：

| 参数 | 用途 | 默认值 |
|---|---|---:|
| `--prediction-frames` | 两次局部模型检测之间只使用卡尔曼预测的帧数；设为 `0` 可每帧检测 | 5 |
| `--warmup-detections` | 初始阶段连续检测成功次数 | 4 |
| `--confidence` | 模型置信度阈值 | 0.25 |
| `--imgsz` | 全局检测输入尺寸 | 640 |
| `--global-tile-size` | 全局搜索分片在原图中的边长，单位为像素 | 640 |
| `--global-tile-overlap` | 相邻全局分片的最小重叠，单位为像素 | 128 |
| `--global-tile-batch-size` | 每批送入模型的全局分片数量 | 4 |
| `--local-imgsz` | 动态裁剪后检测输入尺寸 | 384 |
| `--roi-y-min` / `--roi-y-max` | 人物搜索区域的上下边界，单位为像素；默认整个画面 | 0 / 视频底部 |
| `--crop-min-size` / `--crop-max-size` | 动态裁剪尺寸范围，单位为像素 | 320 / 800 |
| `--recovery-misses` | 局部检测连续漏检后切回全局搜索的次数 | 3 |
| `--global-recovery-ms` | 无成功测量后切回全局搜索的时间，单位为毫秒 | 300 |
| `--max-prediction-ms` | 最长绘制纯预测框的时间，单位为毫秒 | 500 |

全局搜索（首次捕获、预热和丢失后重捕获）现在会对固定纵向 ROI 分片推理。默认 640×640 原图像素分片、至少重叠 128 像素；3840×2160 且搜索区域为整帧时共 8 列×4 行，即 32 个分片，分 8 批运行。检测框会映射回原始画面坐标，并进行跨分片去重。局部跟踪仍使用卡尔曼预测的动态裁剪。若画面中的人高于约 640 像素，可增大 `--global-tile-size`，模型仍会把该分片缩放到 `--imgsz` 指定的输入尺寸；应结合实际视频检查检出率与速度。

处理逻辑集中在 `person_tracking/engine.py`，模型调用在 `person_tracking/detector.py`，运动预测在 `person_tracking/kalman.py`；可以直接在这里修改策略，不会影响服务中的代码。输出视频保留输入帧率和尺寸，但不包含原视频音轨。模型权重和生成的视频请勿提交到 Git。
