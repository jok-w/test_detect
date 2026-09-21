# 离线人物识别实验

这里是从 `touzhi_service` 复制出来的独立离线实验程序。它逐帧读取视频，使用三姿态 YOLO 模型检测人物，并结合卡尔曼滤波、动态裁剪和重新捕获策略，生成带标注的输出视频。此目录不依赖原仓库、实时相机进程或 Web 服务。

## 安装与运行

在 Jetson 的终端中进入本目录并安装依赖：

```bash
cd /home/mtr/Desktop/test_detect
uv sync --python 3.10
```

`pyproject.toml` 支持 Jetson（Linux aarch64）和 Windows x64（AMD64），共用 Python 3.10、NumPy 1.x、`torch 2.8.0`、`torchvision 0.23.0`。uv 按平台选择安装源：Jetson 使用 Jetson AI Lab 的 `jp6/cu126`，Windows 使用 PyTorch 官方 `cu126`，无需手动修改依赖。Windows 的版本组合见 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/#v280)。部署后可检查安装结果：

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

运行时默认检查 `torch.cuda.is_available()`：CUDA 可用时显式使用首张 GPU（`device=0`），不可用时自动回退 CPU。启动日志会打印实际选择的设备；回退时打印“未检测到可用的 CUDA GPU，自动回退到 CPU 推理”。`--device cpu` 可强制使用 CPU；`--device 0` 显式请求 GPU 时，也会在 CUDA 不可用时回退 CPU。全局整帧和局部裁剪推理使用同一设备，每次仅输入一张图。

Jetson 配置用于 JetPack 6 / CUDA 12.6，其他 JetPack 版本需要重新匹配依赖。

### Windows 安装与运行

在 PowerShell 中执行（将项目和视频路径替换为实际位置）：

```powershell
cd E:\detect_test
uv sync --python 3.10
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
uv run python -m person_tracking `
  --model models/best.pt `
  --input "D:\videos\test.mp4" `
  --backend pt `
  --no-display
```

默认安装 CUDA 12.6 版 PyTorch，有兼容的 NVIDIA GPU 和驱动时使用 GPU，无可用 CUDA 时回退 CPU；也可添加 `--device cpu` 强制 CPU。CUDA 版安装包较大，CPU 回退无需更换安装源。需要显示窗口时去掉 `--no-display`。

Windows 可安装 ONNX 导出依赖，使用同一导出及推理工具：

```powershell
uv sync --python 3.10 --extra export
uv run --extra export python -m person_tracking.export_model pt-to-onnx
uv run --extra export python -m person_tracking `
  --input "D:\videos\test.mp4" --backend onnx --device cpu --no-display
```

本配置在 Windows 支持 PT/ONNX，`tensorrt` extra 仅声明 Jetson 系统依赖，不会在 Windows 安装 TensorRT。Jetson 生成的 engine 不用于 Windows；TensorRT 构建及以下 APT 安装步骤在目标 Jetson 执行。

## PT → ONNX → TensorRT FP16

推理后端默认 `--backend auto`：有 CUDA 时分别尝试全局、局部 TensorRT engine；文件缺失、权重指纹或环境不匹配、初始化/预热失败时，该场景回退 PT GPU 并打印原因。没有 CUDA 或指定 `--device cpu` 时使用 PT CPU。`--backend pt` 强制原始模型；`--backend tensorrt` 强制 engine，初始化失败直接报错。视频推理中的普通异常不会被自动回退掩盖。

`--model` 始终指定原始 PT，作为类别/权重一致性校验和回退依据。默认导出路径为：

| 文件 | 用途 | 默认输入 |
|---|---|---|
| `models/best.pt` | 原始权重及 GPU/CPU 回退 | 按推理参数 |
| `models/best.onnx` | 动态 FP32 中间模型、ONNX 验证 | 保留动态维度供两个尺寸共用，运行时 batch=1 |
| `models/best.global.engine` | 全局整帧 FP16 | 固定 1×3×640×640 |
| `models/best.local.engine` | 局部跟踪 FP16 | 固定 1×3×384×384 |

全局检测已取消分片，每个全局检测帧只调用一次模型。TensorRT 针对固定 batch=1、固定空间尺寸构建，不再配置 batch 1～4 的动态优化范围；两个 engine 分别优化全局和局部输入，继续采用 FP16 层优化。整图仍会等比例缩放并补边到 `--imgsz`，不是按原始 4K 分辨率直接计算。若缩小后人物过小，可用 `--imgsz 960` 或 `1280` 比较检出率与延迟，并按同一尺寸重建全局 engine。

升级后需要重新构建两个 engine。旧动态 ONNX 仍可作为构建输入；旧 engine 缺少单图策略标记或输入为动态 batch 时，自动模式会打印原因并回退 PT，强制 TensorRT 模式会报错要求重建。原来的 `--global-tile-size`、`--global-tile-overlap`、`--global-tile-batch-size` 及导出工具的 `--batch` 参数已移除。

安装导出依赖（保留 Jetson torch/torchvision 配置，Ultralytics 固定为本项目验证的 8.4.154）：

```bash
uv sync --python 3.10 --extra export
```

先导出 ONNX。工具保留三姿态类别、PT SHA256 指纹，执行 ONNX 图检查和 CPU 动态输入预热，成功后才替换目标文件：

```bash
uv run --extra export python -m person_tracking.export_model pt-to-onnx \
  --model models/best.pt --output models/best.onnx
```

然后在**目标 Jetson** 构建 engine。`pyproject.toml` 的 `tensorrt` extra 声明 `tensorrt>=10,<11`，并通过 `tool.uv.exclude-dependencies` 将它交给 JetPack/APT 管理。**`uv sync --extra tensorrt` 不会下载或安装 TensorRT，也不会校验系统包版本**；需要先安装设备配套的 TensorRT 10.x，不能仅靠 extra 完成安装。[uv 排除依赖说明](https://docs.astral.sh/uv/reference/settings/#exclude-dependencies)、[NVIDIA JetPack 6.2 安装说明](https://forums.developer.nvidia.com/t/how-to-install-tensorrt-in-jetpack-6-2/335222)。

使用已配置的 JetPack APT 源安装 Python bindings 及构建所需组件：

```bash
sudo apt install tensorrt python3-libnvinfer python3-libnvinfer-dev
/usr/bin/python3.10 -c "import tensorrt; print(tensorrt.__version__)"
uv sync --python 3.10 --extra export --extra tensorrt
```

如果系统 Python 可以导入，但项目虚拟环境不能导入，可将系统 bindings 所在目录添加到项目环境的 `.pth` 文件中。以下命令从实际安装位置读取路径，保留虚拟环境自身依赖的优先级：

```bash
TRT_SITE=$(/usr/bin/python3.10 -c "import pathlib, tensorrt; print(pathlib.Path(tensorrt.__file__).resolve().parent.parent)")
uv run python -c "import pathlib, site, sys; assert sys.prefix != sys.base_prefix, '需要虚拟环境'; pathlib.Path(site.getsitepackages()[0], 'jetson-tensorrt.pth').write_text(sys.argv[1] + '\\n')" "$TRT_SITE"
```

确认项目解释器能导入且构建器可用，再执行转换：

```bash
uv run --extra export python -c "import torch, tensorrt as trt; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), trt.__version__); assert trt.__version__.startswith('10.'); assert trt.Builder(trt.Logger()) is not None"
uv run --extra export python -m person_tracking.export_model onnx-to-engine \
  --onnx models/best.onnx --device 0 --workspace 2
```

Python bindings 需要匹配 Python 3.10 和系统原生 TensorRT 库；不要用通用桌面 CUDA wheel 替代 Jetson 运行库。`workspace` 单位为 GiB，是构建器工作区上限，并非模型运行时总内存限制；内存不足可先降为 `--workspace 1`。

两个 engine 分别构建并预热，通过后才替换各自目标文件；若第二个构建失败，第一个已完成的文件仍可使用。engine 内置 Ultralytics 元数据头，不能直接当作裸 TensorRT plan 交给 `trtexec`。记录并校验 TensorRT、CUDA（PyTorch 报告）、GPU 名称/计算能力及 Ultralytics 版本，环境变化时重新构建。当前工具面向 YOLO11 普通检测模型，使用 FP16 层优化、FP32 输入输出，保留模型后处理的类别无关 NMS，已移除跨分片去重。

构建后运行：

```bash
uv run python -m person_tracking --input /path/to/video.mp4 --backend auto --no-display
# 强制 TensorRT，确认运行没有回退
uv run python -m person_tracking --input /path/to/video.mp4 --backend tensorrt --device 0 --no-display
# 使用 ONNX CPU 对照；默认采用 --model 对应的同名 .onnx
uv run --extra export python -m person_tracking --input /path/to/video.mp4 --backend onnx --device cpu --no-display
```

可使用 `--onnx-model`、`--global-engine`、`--local-engine` 指定导出文件。ONNX 使用两个独立预测器，默认 export 依赖安装的是 CPU ONNX Runtime；只有已安装的 ORT 提供 CUDA provider 时才尝试 ONNX GPU，并打印实际 providers。TensorRT 不依赖 ONNX Runtime GPU 包。

需要改变网络输入尺寸时，重新构建并使用相同推理参数，例如：

```bash
uv run --extra export python -m person_tracking.export_model onnx-to-engine \
  --onnx models/best.onnx --global-imgsz 960 --local-imgsz 384
uv run python -m person_tracking --input /path/to/video.mp4 --backend tensorrt \
  --imgsz 960 --local-imgsz 384 --no-display
```

预处理统一 `rect=False`，将全局原始整帧或局部裁剪等比例缩放、补边至对应正方形输入。原始视频分辨率或裁剪大小变化无需重建 engine；改变网络输入尺寸才需要重建，尺寸必须为 32 的倍数。全局和局部通过显式场景参数选择模型，即使两个网络输入尺寸相同也不会混用。

## 性能与结果对照

### Jetson 硬件视频输出

`--encoder` 控制视频输出，与模型推理的 `--backend` 独立：

- `auto`（默认）：检测到 Jetson Linux 且输出为 MP4 时，检查 GStreamer 所需插件并使用 `nvv4l2h264enc` 硬件 H.264 编码。插件缺失时打印原因并回退 OpenCV；其他平台及非 MP4 输出使用 OpenCV。
- `gstreamer`：强制 Jetson 硬件 H.264 / MP4 输出，依赖不可用或编码失败时直接报错。
- `opencv`：使用原有 OpenCV 输出，`--codec` 默认 `mp4v`。

在目标 Jetson 执行（读取默认按下一节自动选择，也可用 `--decoder opencv` 做读取对照）：

```bash
uv run python -m person_tracking \
  --input camera_20260910_143934.mp4 \
  --output outputs/camera_20260910_143934-nvenc.mp4 \
  --encoder gstreamer --video-bitrate 8000000 --no-display
```

`--video-bitrate` 单位为 bps，默认 8000000（8 Mbps），仅用于硬件输出；它与原 mp4v 编码的质量设置并不等价，应检查细节和标注清晰度。可调大码率改善画质。`--output-max-width` 默认 1920，设为 0 保留原始输出尺寸。输出缩放不改变模型收到的原始图像。

硬件路径使用独立 `gst-launch-1.0` 进程，不依赖项目 OpenCV 的 `GStreamer: YES` 或 Python GI bindings，也不需要替换现有 OpenCV。需要系统的 `fdsrc`、`rawvideoparse`、`nvvidconv`、`nvv4l2h264enc`、`h264parse`、`qtmux` 和 `filesink`。实现将 BGR 转为 BGRx，通过有背压的管道依次提交所有帧，再由 `nvvidconv` 转 NV12 并硬件编码。[NVIDIA 编解码组件](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/Multimedia/AcceleratedGstreamer.html)、[GStreamer rawvideoparse](https://gstreamer.freedesktop.org/documentation/rawparse/rawvideoparse.html)。

每帧写入表示已提交到编码管道，不等于该帧已完成落盘。结束或按 Q/Esc 时关闭输入，等待 EOS 和 MP4 收尾，然后校验输出元数据的帧数、帧率和尺寸；此校验不逐帧解码。编码启动后的错误不会中途切换后端，避免生成混合或缺帧输出；写入或收尾连续等待超过 30 秒时报错，异常中断的文件可能不完整。

日志会打印实际视频输出后端、分项耗时，以及新增的“编码器收尾及校验”和“含收尾实际处理速度”。比较整体速度时优先使用含收尾 FPS，它包括读取、检测、绘图、显示、管道提交、编码收尾及硬件输出元数据检查，不含初始化。原来的平均单帧分项不含收尾，不能把管道提交耗时当成独立硬件编码延迟。

Windows 本地测试不验证 NVENC。将修改同步到 Jetson 后，可运行真实管道集成测试，它会验证包含空格和中文的路径、非整数帧率、输出尺寸、逐帧解码后的数量与顺序：

```bash
RUN_JETSON_GSTREAMER_TEST=1 uv run python -m unittest discover -s tests -p test_video_writer.py -v
```

### Jetson NVDEC 硬件读取及有限预读

`--decoder auto|opencv|gstreamer` 控制读取，与模型 `--backend`、输出 `--encoder` 独立。默认 `auto` 在 Jetson 上对 H.264/H.265 的 MP4/MOV 文件优先使用 NVDEC；其他平台、格式或缺少系统依赖时打印原因并使用 OpenCV。强制 `gstreamer` 不可用时报错。解码辅助进程启动后发生错误不会中途回退或重新从头读视频。

```bash
uv run python -m person_tracking \
  --input camera_20260910_143934.mp4 \
  --output outputs/camera_20260910_143934-nvdec-nvenc.mp4 \
  --decoder gstreamer --decode-prefetch 2 --read-ahead 2 \
  --encoder gstreamer --video-bitrate 8000000 --no-display
```

实现通过 `/usr/bin/python3` 运行独立 GI 辅助进程，使用 `nvv4l2decoder → nvvidconv → appsink`。`--gst-python` 可指定其他提供 JetPack GI 的系统解释器；需要 `Gst`、`GstApp`、`GstVideo` 1.0 命名空间，不要求项目虚拟环境安装 GI，也不需要重新编译 OpenCV。OpenCV 只在启动时读取输入元数据，之后图像由 NVDEC 解码。

保持原始分辨率；例如 4K 视频不会在读取阶段缩为 1080p。`--decode-prefetch 1|2` 设置 appsink 队列上限，默认 2；满时阻塞、不丢帧。解码器内部的参考帧、转换中正在处理的帧不包含在这个队列上限中。像素经 `/dev/shm` 中的一块匿名共享内存传输（4K BGRx 约 33 MB），主进程转为独立 BGR 数组后才请求下一帧；控制 socket 单独传帧序和纳秒 PTS，NVIDIA 的控制台日志不会混入像素。没有在 Python 中无限堆积帧或预读整个视频。

跟踪使用原始 PTS 间隔并把首个有效帧对齐视频起点；缺失或不递增的时间戳沿用单调回退策略。不会将约 25.0353 FPS 的平均帧率直接当成整数 25 来重建正常 PTS。输出仍保持已有固定帧率 MP4 策略，不保留变帧率容器的逐帧 PTS；硬件颜色转换或实际时间戳差异可能使跟踪结果与旧路径略有差异，应验收结果。

`--read-ahead 0|1|2` 控制主程序的完整 BGR 帧预读，默认 2。上一版虽然在 appsink 提前解码，但共享内存复制和 BGRx→BGR 转换仍串行阻塞主线程。现在一个后台线程提前完成取帧、复制和转换，让这些工作与当前帧的检测、绘图及编码重叠。已准备和正在准备的额外帧合计不超过指定数量，按顺序交付、满时等待，不丢帧。每帧独立拥有 BGR 数组，并携带自己的 PTS 和计时快照；后台推进不会把未来帧的时间戳用于当前帧。

该上限独立于 `--decode-prefetch` 的 appsink 缓冲。默认 2 帧的完整帧预读额外需要约 50 MB 的 4K BGR 像素内存，此外仍有原共享内存、appsink 队列和解码器内部缓冲。提前停止时先唤醒阻塞读取，再由后台线程释放共享内存，避免转换还在读取时关闭映射。`--read-ahead 0` 保留上一版同步准备方式，便于原地对照或回退。

新增日志包括“主线程取帧等待”“后台准备”“等待解码样本”“共享内存复制”“BGR 转换”和缺失 PTS 帧数。只统计已交付给检测流程的帧，提前停止时的预读帧不混入平均数。后台准备包含 IPC 和复制/转换；等待样本反映缓冲与消费情况，不能当成纯 NVDEC 解码延迟。开启完整帧预读后，主流程的读取耗时主要反映等待准备好的 BGR 帧。阶段重叠，不能把后台耗时加到主线程分项；性能验收使用包含线程、读取器和编码器收尾的“含收尾实际处理速度”。此优化没有消除物理复制，也没有更改分辨率、颜色转换算法或模型策略，收益需在目标设备测量。

对照最新优化时，两次运行保持同一视频、模型、`--decode-prefetch 2` 和 NVENC 输出，仅分别指定 `--read-ahead 0` 与 `--read-ahead 2`，并使用不同输出文件名。比较含收尾 FPS、798 帧完整性和输出人物框；也可测试 `--read-ahead 1`。不要仅根据读取分项变小判断整体已经加速。

真机测试包含真实 NVENC→NVDEC 往返、非均匀 PTS、帧序/尺寸和提前停止后的进程清理：

```bash
RUN_JETSON_GSTREAMER_TEST=1 uv run python -m unittest discover -s tests -p 'test_video_*.py' -v
```

Windows 会跳过 NVDEC/NVENC 真机测试，但执行共享内存、分段控制消息、行填充、帧序/PTS、错误/超时及原 OpenCV 输出测试。部署后对同一原始视频分别运行 `--decoder opencv --encoder gstreamer` 和 `--decoder gstreamer --encoder gstreamer`，再比较预读 1/2 帧。保持模型和其他参数一致，各运行三次；检查总帧数、时长、颜色和检测结果，并比较含收尾 FPS。输入完整读完时还会校验读取帧数与输入容器元数据，异常截断不会作为正常结束报告。

在同一视频上顺序运行 PT 和 TensorRT，预热后报告平均/P95 延迟、全局/局部模型帧处理耗时、结果框 IoU、类别一致率和检测框有无差异：

```bash
uv run python -m person_tracking.benchmark --input /path/to/video.mp4 \
  --candidate tensorrt --device 0 --frames 300 --benchmark-warmup 8 \
  --report outputs/benchmark-trt.json
```

也可以用 `--candidate onnx --device cpu` 配合 `uv run --extra export` 验证 ONNX。报告写明实际后端和设备；TensorRT 对照强制使用 engine，失败会报错。测试不显示、绘制或编码视频，报告的 FPS 包含视频读取和检测/跟踪，不是完整输出视频的端到端 FPS；结果框比较包含跟踪产生的框，不代表标注集上的 mAP。没有进入局部跟踪的测试，其局部统计为 null，应改用包含人物、遮挡和重捕获的代表性视频。

完整读写流程的速度请另外使用普通 `person_tracking --backend pt/tensorrt --no-display` 处理同一视频，比较已有完成日志。保持设备功耗模式、温度、阈值和输入尺寸一致；两种后端均采用 batch=1。本地 CPU 验证不能代表 Jetson TensorRT 提速。

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

输出视频默认最大宽度为 1920：4K 横屏输入保存为 1920×1080，小视频不放大；按比例缩放后，宽高向下对齐偶数（最小为 2）以适配编码器。检测与卡尔曼跟踪始终使用原始画面，人物框、裁剪框和 ROI 仅在输出绘图时映射到新尺寸，文字在缩小后的画面上绘制。帧率、帧数保持不变。

可用 `--output-max-width 1280` 进一步减少输出编码量，或用 `--output-max-width 0` 保留原尺寸（奇数边长仍对齐偶数）。例如：

```bash
uv run python -m person_tracking --input camera_20260910_143934.mp4 --output-max-width 1920 --no-display
```

启动日志显示输入和输出尺寸；完成日志分别报告读取、检测跟踪、缩放绘图、显示、编码写入的平均耗时。分项及等效处理速度不含模型初始化、编码器收尾；编码写入统计同步 `writer.write` 的耗时。仅减小输出尺寸不会减少原视频的解码开销。

所有处理参数可用 `uv run python -m person_tracking --help` 查看。常用参数：

| 参数 | 用途 | 默认值 |
|---|---|---:|
| `--output-max-width` | 保存和显示画面的最大宽度；0 保留原尺寸，编码尺寸对齐偶数 | 1920 |
| `--prediction-frames` | 两次局部模型检测之间只使用卡尔曼预测的帧数；设为 `0` 可每帧检测 | 5 |
| `--warmup-detections` | 初始阶段连续检测成功次数 | 4 |
| `--confidence` | 模型置信度阈值 | 0.25 |
| `--imgsz` | 全局整帧缩放补边后的正方形网络输入尺寸 | 640 |
| `--local-imgsz` | 动态裁剪后检测输入尺寸 | 384 |
| `--roi-y-min` / `--roi-y-max` | 全局检测后按框中心筛选的纵向范围，也是局部裁剪的边界 | 0 / 视频底部 |
| `--crop-min-size` / `--crop-max-size` | 动态裁剪尺寸范围，单位为像素 | 320 / 800 |
| `--recovery-misses` | 局部检测连续漏检后切回全局搜索的次数 | 3 |
| `--global-recovery-ms` | 无成功测量后切回全局搜索的时间，单位为毫秒 | 300 |
| `--max-prediction-ms` | 最长绘制纯预测框的时间，单位为毫秒 | 500 |

全局搜索（首次捕获、预热和丢失后重捕获）每次直接传入完整原始帧，模型预处理按 `--imgsz` 缩放补边，后处理返回原图坐标的检测框，不再裁分片或叠加分片偏移。指定纵向 ROI 时，仍传入整帧，仅保留检测框中心满足 `roi_y_min <= center_y < roi_y_max` 的候选人物。局部跟踪继续使用卡尔曼预测的动态裁剪及坐标映射。

处理逻辑集中在 `person_tracking/engine.py`，模型调用在 `person_tracking/detector.py`，运动预测在 `person_tracking/kalman.py`；可以直接在这里修改策略，不会影响服务中的代码。输出视频保留输入帧率，尺寸由 `--output-max-width` 决定，不包含原视频音轨。模型权重和生成的视频请勿提交到 Git。
