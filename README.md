# 离线人物识别实验

这里是从 `touzhi_service` 复制出来的独立离线实验程序。它逐帧读取视频，使用三姿态 YOLO 模型检测人物，并结合卡尔曼滤波、动态裁剪和重新捕获策略，生成带标注的输出视频。此目录不依赖原仓库、实时相机进程或 Web 服务。

## 安装与运行

在 PowerShell 中进入本目录并安装依赖：

```powershell
cd D:\detect\_test
uv sync
```

`pyproject.toml` 显式固定 `torch 2.14.0` 和 `torchvision 0.29.0`。在 Linux 或 Windows 上，`uv` 会从 PyTorch 的 CUDA 12.6 索引安装对应构建；GPU 平台需提供兼容 CUDA 12.6 的 NVIDIA 驱动。部署后可检查安装结果：

```powershell
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

运行时如需指定首张 GPU，使用 `--device 0`。

处理任意本地视频。`models/best.pt` 已随实验目录复制，默认输出为 `outputs/输入文件名-tracked.mp4`：

```powershell
uv run python -m person_tracking --input 'D:\path\to\video.mp4'
```

处理时默认实时显示已标注画面；按 `Q` 或 `Esc` 可提前结束。没有桌面窗口时使用 `--no-display`：

```powershell
uv run python -m person_tracking --input 'D:\path\to\video.mp4' --no-display
```

可以指定输出、其他模型和推理设备：

```powershell
uv run python -m person_tracking `
  --input 'D:\path\to\video.mp4' `
  --output 'outputs\trial-01.mp4' `
  --model 'models\best.pt' `
  --device cpu `
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
