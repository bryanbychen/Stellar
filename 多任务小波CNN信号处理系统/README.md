# Multitask Wavelet-CNN Signal Processing System

**[English](#english) | [中文](#中文)**

<a id="english"></a>

A lightweight multi-task deep learning system that combines **Wavelet Transform** with **Convolutional Neural Networks (CNN)** for simultaneous signal denoising and waveform classification.

## Highlights

- **Two-Stage Denoising Pipeline** — Wavelet soft-threshold pre-denoising followed by CNN deep denoising, significantly outperforming either method alone.
- **Multi-Task Learning** — Joint signal denoising (regression) and waveform classification (5 classes) in a single forward pass, with uncertainty-weighted automatic loss balancing (Kendall et al., 2018).
- **Multi-Scale Feature Fusion** — Parallel convolution kernels (3/5/7) capture patterns at different temporal scales; Squeeze-and-Excitation (SE) channel attention adaptively recalibrates feature responses.
- **Lightweight & CPU-Friendly** — ~900 lines of code, trains in minutes on a standard CPU. No GPU required.
- **Real Signal Support** — Import your own CSV/Excel data for automatic denoising and classification.

## Architecture

```
Raw Signal (2, L)
       │
       ▼
┌──────────────────────┐
│  Wavelet Preprocessor│  DWT (db4, 4-level) → Soft Threshold → IDWT
│  14-ch Feature Tensor│  [Original, A4, D4~D1, Denoised] × 2 channels
└──────────┬───────────┘
           │ (14, L)
           ▼
┌──────────────────────┐
│   Shared Encoder     │  MultiScaleConv → ResBlocks → SE Attention
│   (64, L)            │
└─────┬──────────┬─────┘
      │          │
      ▼          ▼
┌───────────┐ ┌────────────────┐
│ Task A:   │ │ Task B:        │
│ Denoising │ │ Classification │
│ Decoder   │ │ Head (FC)      │
│ (2, L)    │ │ (5 classes)    │
└───────────┘ └────────────────┘
```

## Project Structure

```
├── main.py        # Entry point: training pipeline, evaluation & visualization
├── model.py       # MultitaskWaveletCNN architecture & multi-task loss
├── utils.py       # Data I/O, synthetic signal generator, wavelet feature extraction
└── README.md
```

## Supported Waveform Types

| Class | Waveform    | Description                       |
|-------|-------------|-----------------------------------|
| 0     | Sine        | Standard periodic sinusoidal wave |
| 1     | Sawtooth    | Linear ramp with abrupt reset     |
| 2     | Pulse       | Narrow Gaussian spike(s)          |
| 3     | Square      | Rectangular wave with duty cycle  |
| 4     | Triangle    | Symmetric linear ramp up & down   |

## Requirements

- Python 3.8 – 3.10
- PyTorch >= 1.12
- PyWavelets (pywt) >= 1.4
- NumPy >= 1.21
- Pandas >= 1.3
- Matplotlib >= 3.5
- Seaborn >= 0.11
- scikit-learn >= 1.0

## Installation

```bash
git clone https://github.com/<your-username>/multitask-wavelet-cnn.git
cd multitask-wavelet-cnn
pip install torch pywt numpy pandas matplotlib seaborn scikit-learn
```

## Quick Start

Run with default settings (generates synthetic data, trains, evaluates, and produces visualizations):

```bash
python main.py
```

### Command-Line Options

```bash
python main.py --epochs 200 --lr 0.001 --wavelet db4 --level 4 --batch-size 32
```

| Argument              | Default | Description                            |
|-----------------------|---------|----------------------------------------|
| `--epochs`            | 300     | Maximum training epochs                |
| `--lr`                | 1e-3    | Initial learning rate                  |
| `--batch-size`        | 32      | Batch size                             |
| `--wavelet`           | db4     | Wavelet basis function                 |
| `--level`             | 4       | Wavelet decomposition levels           |
| `--samples-per-class` | 200     | Synthetic samples per waveform class   |
| `--noise-min`         | 0.05    | Minimum noise standard deviation       |
| `--noise-max`         | 0.25    | Maximum noise standard deviation       |
| `--file`              | None    | Path to your own CSV/Excel signal data |
| `--columns`           | None    | Column names to use (space-separated)  |
| `--no-real-inference` | False   | Skip real signal inference stage       |

### Inference on Your Own Data

```bash
python main.py --file data.csv --columns current voltage
```

The system will automatically extract wavelet features, denoise, and classify your signal.

## Training Pipeline

The system executes four sequential stages:

1. **Synthetic Dataset Construction** — Generates 5 × 200 = 1000 dual-channel noisy waveform samples with randomized frequency, amplitude, phase, and noise level.
2. **Batch Wavelet Feature Extraction** — Performs 4-level DWT decomposition on each sample, producing a 14-channel feature tensor per sample.
3. **Multi-Task Joint Training** — Trains with AdamW optimizer, ReduceLROnPlateau scheduler, gradient clipping, and early stopping (patience=60).
4. **Evaluation & Visualization** — Generates training curves, denoising comparison plots, V-I characteristic curves, and confusion matrices.

## Output Files

After training, the following files are saved in the project directory:

| File                          | Description                                |
|-------------------------------|--------------------------------------------|
| `multitask_model.pth`        | Trained model checkpoint                   |
| `training_curves.png`        | Loss curves (total, denoising, classification) |
| `denoise_result.png`         | Denoising comparison (noisy → wavelet → CNN → clean) |
| `classification_result.png`  | Confusion matrix & per-class accuracy      |
| `vi_characteristic.png`      | V-I characteristic curve comparison        |
| `real_signal_result.png`     | Real signal inference result (if data provided) |

## Key Technical Details

- **Wavelet Denoising**: Daubechies-4 wavelet, 4-level decomposition, VisuShrink adaptive soft thresholding with MAD noise estimation (\(\sigma = \text{median}(|D_1|) / 0.6745\)).
- **CNN Encoder**: Multi-scale convolution (kernel sizes 3/5/7) → 3 residual blocks with SE channel attention → 64-channel encoded features.
- **Loss Function**: Uncertainty-weighted multi-task loss — learnable \(\log(\sigma^2)\) parameters automatically balance MSE + smoothness regularization (denoising) and cross-entropy (classification).
- **Training Strategy**: AdamW (weight decay 1e-4), ReduceLROnPlateau (factor=0.5, patience=25), gradient clipping (max norm 1.0), early stopping (patience=60).

## License

This project is licensed under the [Apache License 2.0](../LICENSE).

---

<a id="中文"></a>

# 多任务小波CNN信号处理系统

**[English](#english) | [中文](#中文)**

一个轻量级多任务深度学习系统，将**小波变换**与**卷积神经网络（CNN）**相结合，同时实现信号去噪和波形分类。

## 亮点

- **二级去噪流水线** — 小波软阈值一次去噪 + CNN深度二次去噪，效果显著优于单一方法。
- **多任务学习** — 单次前向传播同时完成信号去噪重构（回归任务）和波形类型分类（5类），采用基于不确定性的自适应损失加权（Kendall et al., 2018）。
- **多尺度特征融合** — 并行卷积核（3/5/7）捕获不同时间尺度的信号模式；SE通道注意力机制自适应校准特征响应。
- **轻量化设计，CPU友好** — 全部代码约900行，普通CPU上数分钟即可完成训练，无需GPU。
- **支持真实信号** — 可导入CSV/Excel格式的实测数据，自动完成去噪和分类。

## 系统架构

```
原始信号 (2, L)
       │
       ▼
┌──────────────────────┐
│   小波预处理层        │  DWT (db4, 4级分解) → 软阈值去噪 → IDWT
│   14通道特征张量      │  [原始信号, A4, D4~D1, 去噪信号] × 2通道
└──────────┬───────────┘
           │ (14, L)
           ▼
┌──────────────────────┐
│    共享编码器         │  多尺度卷积 → 残差块 → SE注意力
│    (64, L)           │
└─────┬──────────┬─────┘
      │          │
      ▼          ▼
┌───────────┐ ┌────────────────┐
│ 任务A:    │ │ 任务B:          │
│ 去噪解码器│ │ 分类头          │
│ (2, L)    │ │ (5类)           │
└───────────┘ └────────────────┘
```

## 项目结构

```
├── main.py        # 主入口：训练流程、评估与可视化
├── model.py       # MultitaskWaveletCNN 网络架构与多任务损失
├── utils.py       # 数据读取、合成信号生成、小波特征提取
└── README.md
```

## 支持的波形类型

| 类别 | 波形   | 说明                         |
|------|--------|------------------------------|
| 0    | 正弦波 | 标准周期正弦信号             |
| 1    | 斜波   | 线性上升后瞬间回落的锯齿波   |
| 2    | 脉冲波 | 短持续时间的高斯尖脉冲       |
| 3    | 方波   | 在高低电平间交替的矩形波     |
| 4    | 三角波 | 线性上升与线性下降交替的对称波 |

## 环境要求

- Python 3.8 – 3.10
- PyTorch >= 1.12
- PyWavelets (pywt) >= 1.4
- NumPy >= 1.21
- Pandas >= 1.3
- Matplotlib >= 3.5
- Seaborn >= 0.11
- scikit-learn >= 1.0

## 安装

```bash
git clone https://github.com/<your-username>/multitask-wavelet-cnn.git
cd multitask-wavelet-cnn
pip install torch pywt numpy pandas matplotlib seaborn scikit-learn
```

## 快速开始

使用默认参数运行（自动生成合成数据、训练模型、评估并输出可视化图表）：

```bash
python main.py
```

### 命令行参数

```bash
python main.py --epochs 200 --lr 0.001 --wavelet db4 --level 4 --batch-size 32
```

| 参数                  | 默认值 | 说明                         |
|-----------------------|--------|------------------------------|
| `--epochs`            | 300    | 最大训练轮数                 |
| `--lr`                | 1e-3   | 初始学习率                   |
| `--batch-size`        | 32     | 批处理大小                   |
| `--wavelet`           | db4    | 小波基函数                   |
| `--level`             | 4      | 小波分解层数                 |
| `--samples-per-class` | 200    | 每类合成样本数               |
| `--noise-min`         | 0.05   | 最小噪声标准差               |
| `--noise-max`         | 0.25   | 最大噪声标准差               |
| `--file`              | 无     | 自有CSV/Excel信号数据文件路径 |
| `--columns`           | 无     | 指定使用的列名（空格分隔）    |
| `--no-real-inference` | False  | 跳过真实信号推理阶段          |

### 对自有数据进行推理

```bash
python main.py --file data.csv --columns current voltage
```

系统将自动对你的信号进行小波特征提取、去噪重构和波形分类。

## 训练流程

系统按四个阶段顺序执行：

1. **合成数据集构建** — 自动生成 5类 × 200 = 1000 个双通道含噪波形样本，频率、幅度、相位、噪声水平均随机变化。
2. **批量小波特征提取** — 对每个样本进行4级DWT分解，生成14通道特征张量。
3. **多任务联合训练** — 采用AdamW优化器、ReduceLROnPlateau学习率调度、梯度裁剪和早停策略（patience=60）。
4. **评估与可视化** — 生成训练曲线、去噪对比图、V-I特性曲线和分类混淆矩阵。

## 输出文件

训练完成后，以下文件保存在项目目录中：

| 文件                          | 说明                                   |
|-------------------------------|----------------------------------------|
| `multitask_model.pth`        | 训练好的模型检查点                      |
| `training_curves.png`        | 损失曲线（总损失、去噪损失、分类损失）   |
| `denoise_result.png`         | 去噪效果对比（含噪→小波→CNN→干净信号）   |
| `classification_result.png`  | 混淆矩阵与各类准确率                    |
| `vi_characteristic.png`      | V-I特性曲线对比                         |
| `real_signal_result.png`     | 真实信号推理结果（如提供数据文件）        |

## 核心技术细节

- **小波去噪**：Daubechies-4小波基，4级分解，VisuShrink自适应软阈值，MAD噪声估计（σ = median(|D₁|) / 0.6745）。
- **CNN编码器**：多尺度卷积（卷积核3/5/7）→ 3组残差块 + SE通道注意力 → 64通道编码特征。
- **损失函数**：不确定性加权多任务损失 — 可学习的 log(σ²) 参数自动平衡 MSE+平滑正则（去噪）与交叉熵（分类）。
- **训练策略**：AdamW（权重衰减1e-4）、ReduceLROnPlateau（factor=0.5, patience=25）、梯度裁剪（最大范数1.0）、早停（patience=60）。

## 许可证

本项目基于 [Apache License 2.0](../LICENSE) 开源。
