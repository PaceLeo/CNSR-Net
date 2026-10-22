# CNSR-Net 最小推理包

本包只保留当前 CNSR-Net Balanced 推理所需的源码、配置、冻结 CLIP 权重和校准文件。

## 目录

- `CNSR-Net/`：CNSR-Net 推理源码及其 CLIP 加载依赖
- `repro_assets/balanced_phase12/weights/ViT-L-14-336px.pt`：冻结 CLIP 权重
- `repro_assets/balanced_phase12/balanced_calibration.pt`：Balanced 运行校准文件

## 数据集位置

如果运行后续 FIgLib 数据处理或实验脚本，将数据集放在包根目录的：

```text
dataset/FigLib/
```

即与 `CNSR-Net-main` 同级的 `dataset/FigLib`。当前最小包未包含这些脚本

## 基础用法

从 `CNSR-Net/` 目录运行，并确保运行环境已经安装 PyTorch、TorchVision、Pillow、PyYAML、NumPy、SciPy、scikit-learn、tqdm、packaging、ftfy、regex 等依赖：

```python
from wildfire_detection import CNSRNet

detector = CNSRNet(
    config_path="configs/phase12_balanced.yaml",
    calibration_path="../repro_assets/balanced_phase12/balanced_calibration.pt",
    device="cuda",
)
```

初始化后，需要先用每个相机的正常历史帧调用 `calibrate_camera`，再按时间顺序调用 `process_frame`。

