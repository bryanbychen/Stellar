#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py - 数据读取、合成信号生成与小波多尺度特征提取工具模块

本模块提供以下核心功能：
    1. 鲁棒性数据读取：自动检测 CSV/Excel 格式，自动筛选有效数值列
    2. 合成信号生成器：生成多种标准波形（正弦波、斜波、脉冲波、方波、三角波）
       并注入可控高斯噪声，同时自动标注信号类型标签（用于分类任务）
    3. 多尺度小波分解：利用 PyWavelets 进行多级小波分解，生成多通道特征张量
    4. 二次去噪管线：小波软阈值一次去噪 → CNN 二次深度去噪
    5. 数据归一化与预处理管线

作者: Aurora (AI Assistant)
日期: 2026-03-07
版本: 2.0 - 多任务学习版（去噪 + 分类）
"""

import os
import sys
import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional, Dict


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_FILE = "data.csv"

# 小波分解默认参数
DEFAULT_WAVELET = "db4"          # Daubechies-4 小波基函数
DEFAULT_DECOMP_LEVEL = 4         # 分解层数
DEFAULT_MODE = "symmetric"       # 边界延拓模式

# 信号类型标签映射（分类任务核心常量）
SIGNAL_TYPE_MAP = {
    "sine":      0,   # 正弦波
    "sawtooth":  1,   # 斜波（锯齿波）
    "pulse":     2,   # 脉冲波
    "square":    3,   # 方波
    "triangle":  4,   # 三角波
}

# 反向映射：索引 → 中文名称
SIGNAL_TYPE_NAMES_CN = {
    0: "正弦波 (Sine)",
    1: "斜波 (Sawtooth)",
    2: "脉冲波 (Pulse)",
    3: "方波 (Square)",
    4: "三角波 (Triangle)",
}

NUM_SIGNAL_CLASSES = len(SIGNAL_TYPE_MAP)


# ============================================================================
#  数据读取模块
# ============================================================================

def load_signal_data(
    file_path: Optional[str] = None,
    target_columns: Optional[List[str]] = None  # 修改为支持多个列名
) -> Tuple[np.ndarray, List[str]]:
    if file_path is None: file_path = _auto_discover_data_file()
    if not os.path.exists(file_path): raise FileNotFoundError(f"未找到: {file_path}")
    df = _read_dataframe(file_path)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        raise ValueError("【错误】文件中的数值列少于两列，无法进行双通道处理。")

    if target_columns is not None:
        cols = target_columns
    else:
        cols = numeric_cols[:2] # 默认取前两列（例如电流和电压）
        print(f"[信息] 自动选取前两列: {cols}")

    # 修改点：提取两列并转置为 (2, Length)
    signal = df[cols].values.astype(np.float64).T 
    
    # 简单的 NaN 填充
    if np.isnan(signal).any():
        df[cols] = df[cols].interpolate(method="linear").bfill().ffill()
        signal = df[cols].values.astype(np.float64).T
        
    return signal, cols


def _auto_discover_data_file() -> str:
    """自动扫描项目目录，寻找可用数据文件。

    按优先级扫描：data.csv > *.csv > *.xlsx > *.xls

    Returns:
        str: 找到的数据文件的完整路径。

    Raises:
        FileNotFoundError: 当项目目录中无任何数据文件时抛出。
    """
    default_path = os.path.join(PROJECT_DIR, DEFAULT_DATA_FILE)
    if os.path.exists(default_path):
        return default_path

    supported_extensions = [".csv", ".xlsx", ".xls"]
    candidates = []
    for fname in os.listdir(PROJECT_DIR):
        _, ext = os.path.splitext(fname.lower())
        if ext in supported_extensions:
            candidates.append(fname)

    if not candidates:
        raise FileNotFoundError(
            f"【错误】在 {PROJECT_DIR}/ 下未找到任何数据文件。\n"
            f"支持的格式: {supported_extensions}\n"
            f"请将数据文件（如 data.csv）放入该目录。"
        )

    selected = sorted(candidates, key=lambda f: supported_extensions.index(
        os.path.splitext(f.lower())[1]
    ))[0]
    full_path = os.path.join(PROJECT_DIR, selected)
    print(f"[信息] 未找到 data.csv，自动选取: {selected}")
    return full_path


def _read_dataframe(file_path: str) -> pd.DataFrame:
    """智能读取数据文件为 DataFrame。

    根据文件扩展名和内容自动选择合适的读取引擎。
    支持 CSV 和 Excel 格式（包括伪装扩展名的情况）。

    Args:
        file_path: 数据文件路径。

    Returns:
        pd.DataFrame: 读取的数据框。
    """
    ext = os.path.splitext(file_path.lower())[1]

    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(file_path, engine="openpyxl")

    try:
        df = pd.read_csv(file_path)
        if len(df.columns) >= 1 and len(df) > 0:
            return df
    except Exception:
        pass

    try:
        print("[信息] CSV 读取失败，尝试以 Excel 格式读取...")
        return pd.read_excel(file_path, engine="openpyxl")
    except Exception as e:
        raise ValueError(
            f"【错误】无法读取文件 '{os.path.basename(file_path)}'。\n"
            f"已尝试 CSV 和 Excel 格式，均失败。\n"
            f"详细信息: {str(e)}"
        )


# ============================================================================
#  合成信号生成器（用于多任务训练数据构建）
# ============================================================================

class SyntheticSignalGenerator:
    """合成信号生成器。

    为多任务学习（去噪 + 分类）生成标准波形训练集。
    每个样本包含：干净信号（去噪目标）、含噪信号（模型输入）、类型标签（分类目标）。

    支持的波形类型：
        - 正弦波 (Sine): 标准周期正弦信号
        - 斜波 (Sawtooth): 线性上升后瞬间回落的锯齿波
        - 脉冲波 (Pulse): 短持续时间的尖锐脉冲
        - 方波 (Square): 在高低电平间交替的矩形波
        - 三角波 (Triangle): 线性上升与线性下降交替的对称波

    Attributes:
        signal_length: 生成信号的采样点数。
        noise_levels: 噪声强度范围 [min, max]。
        rng: 随机数生成器。
    """

    def __init__(
        self,
        signal_length: int = 329,
        noise_range: Tuple[float, float] = (0.05, 0.25),
        seed: Optional[int] = None
    ):
        """初始化合成信号生成器。

        Args:
            signal_length: 信号采样点数。
            noise_range: 噪声标准差范围，(最小值, 最大值)。
            seed: 随机种子，用于可复现的实验。
        """
        self.signal_length = signal_length
        self.noise_range = noise_range
        self.rng = np.random.RandomState(seed)

    def generate_dataset(
        self,
        samples_per_class: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """批量生成多类型信号数据集。

        为每种信号类型生成指定数量的样本，每个样本在频率、幅度、
        相位等参数上随机变化，以增加数据多样性。

        Args:
            samples_per_class: 每种类型生成的样本数量。

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - noisy_signals: 含噪信号集，形状 (N, signal_length)。
                - clean_signals: 干净信号集（去噪目标），形状 (N, signal_length)。
                - labels: 类型标签集，形状 (N,)，值域 [0, NUM_SIGNAL_CLASSES-1]。
        """
        all_noisy = []
        all_clean = []
        all_labels = []

        generators = {
            "sine":     self._generate_sine,
            "sawtooth": self._generate_sawtooth,
            "pulse":    self._generate_pulse,
            "square":   self._generate_square,
            "triangle": self._generate_triangle,
        }

        for sig_type, gen_func in generators.items():
            label = SIGNAL_TYPE_MAP[sig_type]
            print(f"[合成数据] 生成 {sig_type} 双列信号 × {samples_per_class}...")

            for _ in range(samples_per_class):
                # 修改点：生成两通道数据模拟（电流与电压）
                clean1 = gen_func()
                clean2 = gen_func() 
                clean = np.stack([clean1, clean2], axis=0) # 形状变为 (2, L)

                noise_level = self.rng.uniform(*self.noise_range)
                noise = self.rng.randn(2, self.signal_length) * noise_level
                noisy = clean + noise

                all_clean.append(clean)
                all_noisy.append(noisy)
                all_labels.append(label)

        # 打乱顺序
        indices = self.rng.permutation(len(all_labels))
        noisy_signals = np.array(all_noisy)[indices]
        clean_signals = np.array(all_clean)[indices]
        labels = np.array(all_labels)[indices]

        print(f"[合成数据] 数据集构建完成：共 {len(labels)} 样本，"
              f"{NUM_SIGNAL_CLASSES} 类别")
        return noisy_signals, clean_signals, labels

    def _generate_sine(self) -> np.ndarray:
        """生成随机参数的正弦波信号。

        参数随机范围：频率 1~8 个周期，幅度 0.3~1.0，相位 0~2π。

        Returns:
            np.ndarray: 归一化后的干净正弦波信号。
        """
        t = np.linspace(0, 1, self.signal_length)
        freq = self.rng.uniform(1, 8)
        amplitude = self.rng.uniform(0.3, 1.0)
        phase = self.rng.uniform(0, 2 * np.pi)
        signal = amplitude * np.sin(2 * np.pi * freq * t + phase)
        return self._normalize(signal)

    def _generate_sawtooth(self) -> np.ndarray:
        """生成随机参数的斜波（锯齿波）信号。

        线性上升段占比随机（0.7~1.0），模拟不同斜率的锯齿波。

        Returns:
            np.ndarray: 归一化后的干净斜波信号。
        """
        t = np.linspace(0, 1, self.signal_length)
        freq = self.rng.uniform(1, 6)
        amplitude = self.rng.uniform(0.3, 1.0)
        # 用取模实现锯齿：(freq*t) mod 1 产生 0→1 的锯齿
        signal = amplitude * (2 * (freq * t % 1) - 1)
        return self._normalize(signal)

    def _generate_pulse(self) -> np.ndarray:
        """生成随机参数的脉冲波信号。

        在平坦基线上叠加若干个高斯脉冲，模拟真实脉冲信号。
        脉冲数量 1~5 个，宽度和位置随机。

        Returns:
            np.ndarray: 归一化后的干净脉冲波信号。
        """
        t = np.linspace(0, 1, self.signal_length)
        signal = np.zeros(self.signal_length)
        n_pulses = self.rng.randint(1, 6)

        for _ in range(n_pulses):
            center = self.rng.uniform(0.05, 0.95)
            width = self.rng.uniform(0.005, 0.04)
            amplitude = self.rng.uniform(0.5, 1.0)
            signal += amplitude * np.exp(-((t - center) ** 2) / (2 * width ** 2))

        return self._normalize(signal)

    def _generate_square(self) -> np.ndarray:
        """生成随机参数的方波信号。

        占空比在 30%~70% 间随机变化，模拟不同形态的方波。

        Returns:
            np.ndarray: 归一化后的干净方波信号。
        """
        t = np.linspace(0, 1, self.signal_length)
        freq = self.rng.uniform(1, 6)
        duty_cycle = self.rng.uniform(0.3, 0.7)
        amplitude = self.rng.uniform(0.3, 1.0)
        # 方波：周期内前 duty_cycle 部分为高，其余为低
        signal = amplitude * (((freq * t) % 1) < duty_cycle).astype(float)
        signal = 2 * signal - amplitude  # 居中对称
        return self._normalize(signal)

    def _generate_triangle(self) -> np.ndarray:
        """生成随机参数的三角波信号。

        对称三角波：线性上升半周期 + 线性下降半周期。

        Returns:
            np.ndarray: 归一化后的干净三角波信号。
        """
        t = np.linspace(0, 1, self.signal_length)
        freq = self.rng.uniform(1, 6)
        amplitude = self.rng.uniform(0.3, 1.0)
        # 三角波：利用 arcsin(sin(...)) 的线性化效果
        signal = amplitude * (2 * np.abs(2 * (freq * t % 1) - 1) - 1)
        return self._normalize(signal)

    @staticmethod
    def _normalize(signal: np.ndarray) -> np.ndarray:
        """将信号归一化到 [-1, 1] 范围。

        Args:
            signal: 输入信号。

        Returns:
            np.ndarray: 归一化后的信号。
        """
        s_min, s_max = signal.min(), signal.max()
        if s_max - s_min < 1e-10:
            return np.zeros_like(signal)
        return 2 * (signal - s_min) / (s_max - s_min) - 1


# ============================================================================
#  小波多尺度特征提取模块
# ============================================================================

class WaveletFeatureExtractor:
    """多尺度小波特征提取器。

    利用 PyWavelets 进行多级小波分解（Multi-level Decomposition），
    将一维信号拆解为近似分量（Approximation）和多级细节分量（Details），
    并将其重组为多通道特征张量，用作 CNN 的输入。

    核心原理：
        信号 S = A_n + D_n + D_{n-1} + ... + D_1
        其中 A_n 为第 n 级近似分量，D_i 为第 i 级细节分量。
        所有分量被对齐到原始信号长度后堆叠为多通道张量。

    Attributes:
        wavelet: 小波基函数名称（如 'db4', 'sym6', 'coif3'）。
        level: 分解层数。
        mode: 边界延拓模式。
        scaler: 数据归一化器（MinMaxScaler）。
    """

    def __init__(
        self,
        wavelet: str = DEFAULT_WAVELET,
        level: int = DEFAULT_DECOMP_LEVEL,
        mode: str = DEFAULT_MODE
    ):
        """初始化小波特征提取器。

        Args:
            wavelet: 小波基函数名称。推荐 'db4'（信号拟合）或 'sym6'（对称性好）。
            level: 分解层数。层数越高，提取的低频特征越粗糙。
                   建议范围: 3-6，取决于信号长度。
            mode: 边界延拓模式。可选 'symmetric', 'zero', 'constant' 等。
        """
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        available_wavelets = pywt.wavelist()
        if wavelet not in available_wavelets:
            raise ValueError(
                f"【错误】不支持的小波基函数 '{wavelet}'。\n"
                f"可用的小波族: {pywt.families()}"
            )

        self._original_level = level

    def extract_features(self, signal: np.ndarray, normalize: bool = True):
        # 修改点：signal 现在的形状是 (2, L)
        # 分别处理通道 1 (电流) 和 通道 2 (电压)
        feats1, den1, names1 = self._extract_single_channel(signal[0], normalize)
        feats2, den2, names2 = self._extract_single_channel(signal[1], normalize)

        # 在通道维度拼接：7 + 7 = 14 个通道
        features = np.concatenate([feats1, feats2], axis=0) 
        # 去噪目标也堆叠起来：变成 2 个通道
        denoised = np.stack([den1, den2], axis=0) 
        
        channel_names = [f"CH1_{n}" for n in names1] + [f"CH2_{n}" for n in names2]
        return features, denoised, channel_names
        
    def _extract_single_channel(self, signal_1d: np.ndarray, normalize: bool):
        # 把你原本 extract_features 里的核心逻辑原封不动移到这里
        signal_length = len(signal_1d)
        max_level = pywt.dwt_max_level(signal_length, self.wavelet)
        current_level = min(self.level, max_level)

        if normalize:
            signal_processed = self.scaler.fit_transform(signal_1d.reshape(-1, 1)).flatten()
        else:
            signal_processed = signal_1d.copy()

        coefficients = pywt.wavedec(signal_processed, wavelet=self.wavelet, level=current_level, mode=self.mode)
        denoised_coefficients = self._apply_soft_threshold(coefficients)
        denoised = pywt.waverec(denoised_coefficients, self.wavelet)[:signal_length]

        aligned_components, channel_names = self._align_and_stack(
            coefficients, signal_processed, denoised, signal_length
        )
        return aligned_components, denoised, channel_names

    def extract_features_batch(
        self,
        signals: np.ndarray,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """批量提取多条信号的小波多尺度特征。

        对每条信号独立执行小波分解与去噪，并返回批量特征张量。
        此方法在多任务训练数据准备阶段使用，静默模式无逐条日志。

        Args:
            signals: 信号批次，形状 (batch_size, signal_length)。
            normalize: 是否归一化。

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - batch_features: 批量特征张量，形状 (batch, n_channels, signal_length)。
                - batch_denoised: 批量一次去噪信号，形状 (batch, signal_length)。
        """
        batch_features = []
        batch_denoised = []

        for i in range(len(signals)):
            feats, denoised, _ = self.extract_features(
                signals[i], normalize=normalize
            )
            batch_features.append(feats)
            batch_denoised.append(denoised)

        return np.array(batch_features), np.array(batch_denoised)

    def _apply_soft_threshold(
        self,
        coefficients: List[np.ndarray]
    ) -> List[np.ndarray]:
        """对小波系数应用自适应软阈值去噪（一次去噪）。

        使用 VisuShrink（通用阈值）方法对细节分量进行去噪，
        保留近似分量不变。阈值公式: λ = σ * √(2 * ln(N))

        Args:
            coefficients: 小波分解系数列表 [cA_n, cD_n, ..., cD_1]。

        Returns:
            List[np.ndarray]: 阈值处理后的系数列表。
        """
        denoised_coeffs = [coefficients[0].copy()]

        finest_detail = coefficients[-1]
        sigma = np.median(np.abs(finest_detail)) / 0.6745
        n_samples = sum(len(c) for c in coefficients)
        threshold = sigma * np.sqrt(2 * np.log(max(n_samples, 2)))

        for i in range(1, len(coefficients)):
            level_threshold = threshold / (2 ** (len(coefficients) - 1 - i) * 0.5 + 1)
            denoised_coeffs.append(
                pywt.threshold(coefficients[i], level_threshold, mode="soft")
            )

        return denoised_coeffs

    def _align_and_stack(
        self,
        coefficients: List[np.ndarray],
        original: np.ndarray,
        denoised: np.ndarray,
        target_length: int
    ) -> Tuple[np.ndarray, List[str]]:
        """将各级小波分量对齐到原始信号长度并堆叠为多通道张量。

        Args:
            coefficients: 小波分解系数列表。
            original: 归一化后的原始信号。
            denoised: 去噪后的重构信号。
            target_length: 目标对齐长度。

        Returns:
            Tuple[np.ndarray, List[str]]: (多通道特征张量, 通道名称列表)。
        """
        channels = []
        names = []

        channels.append(original)
        names.append("Original")

        approx = self._interpolate_to_length(coefficients[0], target_length)
        channels.append(approx)
        names.append(f"Approx_L{self.level}")

        for i in range(1, len(coefficients)):
            detail = self._interpolate_to_length(coefficients[i], target_length)
            channels.append(detail)
            level_idx = self.level - i + 1
            names.append(f"Detail_L{level_idx}")

        channels.append(denoised)
        names.append("Denoised_L1")

        feature_tensor = np.stack(channels, axis=0)
        return feature_tensor, names

    @staticmethod
    def _interpolate_to_length(
        array: np.ndarray,
        target_length: int
    ) -> np.ndarray:
        """将一维数组通过线性插值调整到目标长度。

        Args:
            array: 输入数组。
            target_length: 目标长度。

        Returns:
            np.ndarray: 调整长度后的数组。
        """
        if len(array) == target_length:
            return array
        x_original = np.linspace(0, 1, len(array))
        x_target = np.linspace(0, 1, target_length)
        return np.interp(x_target, x_original, array)


# ============================================================================
#  辅助工具函数
# ============================================================================

def get_device_info() -> str:
    """检测并返回可用的计算设备信息。

    优先使用 CUDA GPU，不可用时回退到 CPU。

    Returns:
        str: 设备标识符（'cuda' 或 'cpu'）。
    """
    import torch
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[设备] 使用 GPU: {device_name} (显存: {memory_total:.1f} GB)")
        return "cuda"
    else:
        print("[设备] CUDA 不可用，使用 CPU 进行计算。")
        return "cpu"


def classify_real_signal(signal: np.ndarray) -> Tuple[int, str, Dict[str, float]]:
    """基于统计特征对真实信号进行启发式波形类型预判。

    通过分析信号的统计特性（峰度、过零率、平坦度、脉冲因子等）
    给出一个初步的波形类型估计，作为分类训练的软标签参考。

    注意：此函数仅作为无标签真实数据的辅助标注，最终分类
    以 CNN 模型输出为准。

    Args:
        signal: 一维输入信号。

    Returns:
        Tuple[int, str, Dict[str, float]]:
            - 预测类别索引。
            - 预测类别中文名称。
            - 各类别的置信度得分字典。
    """
    from scipy import stats

    # 归一化到 [-1, 1]
    s = signal - np.mean(signal)
    s_max = np.max(np.abs(s))
    if s_max > 1e-10:
        s = s / s_max

    # 特征计算
    kurtosis = stats.kurtosis(s, fisher=True)        # 峰度（高斯=0）
    zero_crossings = np.sum(np.diff(np.sign(s)) != 0) / len(s)  # 过零率
    rms = np.sqrt(np.mean(s ** 2))                    # 均方根
    crest_factor = np.max(np.abs(s)) / (rms + 1e-10)  # 波峰因子
    diff_std = np.std(np.diff(s))                     # 一阶差分标准差

    # 平坦度：信号在 |s| > 0.8 区域的时间占比
    flatness = np.mean(np.abs(s) > 0.8)

    # 简单评分模型
    scores = {}
    scores["正弦波 (Sine)"] = 1.0 / (1.0 + abs(kurtosis + 1.5))
    scores["斜波 (Sawtooth)"] = min(diff_std * 5, 1.0) * (1.0 - flatness)
    scores["脉冲波 (Pulse)"] = min(kurtosis / 5.0, 1.0) if kurtosis > 2 else 0.1
    scores["方波 (Square)"] = flatness * (1.0 + abs(kurtosis + 1.2))
    scores["三角波 (Triangle)"] = (
        1.0 / (1.0 + abs(kurtosis + 1.2))
        * min(diff_std * 3, 1.0)
        * (1.0 - flatness)
    )

    # 归一化得分
    total = sum(scores.values())
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}

    best_type = max(scores, key=scores.get)
    # 映射回索引
    name_to_idx = {v: k for k, v in SIGNAL_TYPE_NAMES_CN.items()}
    pred_idx = name_to_idx.get(best_type, 0)

    return pred_idx, best_type, scores


# ============================================================================
#  模块测试入口
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  小波特征提取 + 合成信号生成 - 模块测试")
    print("=" * 60)

    try:
        # 测试合成信号生成
        generator = SyntheticSignalGenerator(signal_length=329, seed=42)
        noisy, clean, labels = generator.generate_dataset(samples_per_class=5)
        print(f"\n[结果] 合成数据集形状: noisy={noisy.shape}, "
              f"clean={clean.shape}, labels={labels.shape}")
        print(f"[结果] 类别分布: {dict(zip(*np.unique(labels, return_counts=True)))}")

        # 测试小波特征提取
        extractor = WaveletFeatureExtractor()
        feats, denoised, names = extractor.extract_features(noisy[0], normalize=False)
        print(f"[结果] 单样本特征形状: {feats.shape}")
        print(f"[结果] 通道名称: {names}")

        print("\n[测试通过] ✓ 所有模块正常工作")

    except Exception as e:
        print(f"\n[测试失败] ✗ {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
