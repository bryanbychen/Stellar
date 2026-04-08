#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.py - 多任务小波 CNN 模型：二次去噪重构 + 信号类型分类

本模块定义了 MultitaskWaveletCNN 网络架构，核心设计要素包括：
    1. 共享特征编码器（Shared Encoder）：提取小波多尺度特征的深层表征
    2. 任务 A - 去噪解码器（Denoising Decoder）：基于转置卷积的信号重构头
    3. 任务 B - 分类头（Classification Head）：信号类型识别（5 类波形）
    4. 不确定性加权多任务损失（Uncertainty-weighted Multi-Task Loss）

网络架构概览:
                           Input (C, L)
                               │
                    ┌──────────┴──────────┐
                    │   Shared Encoder     │
                    │  Conv → Res → Attn   │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │                      │
           ┌───────┴───────┐    ┌─────────┴─────────┐
           │  Task A Head  │    │   Task B Head      │
           │  Denoising    │    │   Classification   │
           │  Decoder      │    │   FC Layers        │
           └───────┬───────┘    └─────────┬─────────┘
                   │                      │
           Reconstructed (L)        Logits (num_classes)

作者: Aurora (AI Assistant)
日期: 2026-03-08
版本: 2.0 - 多任务学习版
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


# ============================================================================
#  基础构建块
# ============================================================================

class ConvBlock(nn.Module):
    """标准一维卷积块：Conv1d + BatchNorm + GELU + Dropout。

    Attributes:
        conv: 一维卷积层。
        bn: 批归一化层。
        act: GELU 激活函数。
        dropout: Dropout 正则化层。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout_rate: float = 0.0,
        dilation: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(self.bn(self.conv(x))))


class ResidualBlock(nn.Module):
    """一维残差块：两层卷积 + 跳跃连接。

    结构:
        x → Conv → BN → GELU → Dropout → Conv → BN → (+) → GELU → out
        └───────────────── shortcut ─────────────────┘
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=1, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.act = nn.GELU()

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation 通道注意力模块。

    GAP + GMP 双分支 → 共享 MLP → Sigmoid 加权。
    """

    def __init__(self, channels: int, reduction_ratio: int = 4):
        super().__init__()
        reduced = max(channels // reduction_ratio, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.GELU(),
            nn.Linear(reduced, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _ = x.size()
        avg_out = self.mlp(F.adaptive_avg_pool1d(x, 1).view(B, C))
        max_out = self.mlp(F.adaptive_max_pool1d(x, 1).view(B, C))
        attn = self.sigmoid(avg_out + max_out).view(B, C, 1)
        return x * attn


class MultiScaleConv(nn.Module):
    """多尺度卷积模块：并行使用不同大小的卷积核提取多尺度模式。

    结构:
        x → [Conv_k3 | Conv_k5 | Conv_k7] → Concat → 1x1 Conv → out
    """

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        super().__init__()
        branch_ch = out_channels // 3

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(branch_ch), nn.GELU()
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_channels, branch_ch, 5, padding=2, bias=False),
            nn.BatchNorm1d(branch_ch), nn.GELU()
        )
        self.branch7 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels - 2 * branch_ch, 7, padding=3, bias=False),
            nn.BatchNorm1d(out_channels - 2 * branch_ch), nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels), nn.GELU(),
            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        out = torch.cat([b3, b5, b7], dim=1)
        return self.fusion(out)


# ============================================================================
#  共享编码器
# ============================================================================

class SharedEncoder(nn.Module):
    """共享特征编码器。

    从多通道小波特征中提取高级表征，供下游任务头使用。

    架构:
        Input(C, L) → MultiScaleConv(C → 48)
                    → ConvBlock(48 → 64)
                    → ResBlock(64 → 64) + Attn
                    → ResBlock(64 → 128) + Attn
                    → ResBlock(128 → 128) + Attn
                    → ConvBlock(128 → 64)
        输出: (B, 64, L) — 序列级特征

    Attributes:
        out_channels: 编码器输出通道数。
    """

    def __init__(
        self,
        input_channels: int,
        base_filters: int = 48,
        dropout_rate: float = 0.15
    ):
        super().__init__()
        self.out_channels = 64

        # 入口：多尺度卷积
        self.entry = MultiScaleConv(input_channels, base_filters, dropout_rate * 0.5)

        # 过渡卷积
        self.conv_up = ConvBlock(base_filters, 64, kernel_size=5, padding=2,
                                 dropout_rate=dropout_rate * 0.5)

        # 残差堆叠 + 通道注意力
        self.res1 = ResidualBlock(64, 64, kernel_size=3, dropout_rate=dropout_rate)
        self.attn1 = ChannelAttention(64)

        self.res2 = ResidualBlock(64, 128, kernel_size=3, dropout_rate=dropout_rate)
        self.attn2 = ChannelAttention(128)

        self.res3 = ResidualBlock(128, 128, kernel_size=3, dropout_rate=dropout_rate)
        self.attn3 = ChannelAttention(128)

        # 特征压缩
        self.conv_down = ConvBlock(128, 64, kernel_size=3, padding=1,
                                   dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L)

        Returns:
            (B, 64, L) 序列级编码特征
        """
        out = self.entry(x)         # (B, 48, L)
        out = self.conv_up(out)     # (B, 64, L)

        out = self.attn1(self.res1(out))   # (B, 64, L)
        out = self.attn2(self.res2(out))   # (B, 128, L)
        out = self.attn3(self.res3(out))   # (B, 128, L)

        out = self.conv_down(out)   # (B, 64, L)
        return out


# ============================================================================
#  任务 A: 去噪重构解码器
# ============================================================================

class DenoisingDecoder(nn.Module):
    def __init__(self, encoder_channels: int = 64, out_channels: int = 2, dropout_rate: float = 0.1):
        super().__init__()
        self.decoder = nn.Sequential(
            ResidualBlock(encoder_channels, encoder_channels, kernel_size=5, dropout_rate=dropout_rate),
            ConvBlock(encoder_channels, 32, kernel_size=3, padding=1, dropout_rate=dropout_rate * 0.5),
            nn.Conv1d(32, out_channels, kernel_size=1, bias=True)
        )
    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.decoder(encoded)


# ============================================================================
#  任务 B: 信号类型分类头
# ============================================================================

class ClassificationHead(nn.Module):
    """信号类型分类头。

    将编码器特征通过全局池化 + 全连接层映射到类别空间。

    架构:
        encoded(B, 64, L) → GAP → (B, 64)
                           + GMP → (B, 64)
                           = Concat → (B, 128)
                           → FC(128→64) → GELU → Dropout
                           → FC(64→num_classes) → Logits
    """

    def __init__(
        self,
        encoder_channels: int = 64,
        num_classes: int = 5,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(encoder_channels * 2, encoder_channels),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(encoder_channels, num_classes)
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoded: (B, 64, L)

        Returns:
            (B, num_classes) 分类 logits
        """
        avg_pool = F.adaptive_avg_pool1d(encoded, 1).squeeze(-1)  # (B, 64)
        max_pool = F.adaptive_max_pool1d(encoded, 1).squeeze(-1)  # (B, 64)
        pooled = torch.cat([avg_pool, max_pool], dim=1)           # (B, 128)
        return self.classifier(pooled)


# ============================================================================
#  多任务主网络
# ============================================================================

class MultitaskWaveletCNN(nn.Module):
    """多任务小波 CNN 主网络。

    整合共享编码器、去噪重构解码器、分类头三大组件，
    支持单独或联合执行两个任务。

    架构:
        ┌─────────────────────────────┐
        │     SharedEncoder           │
        │  MultiScaleConv → ResBlocks │
        │  + ChannelAttention         │
        └──────────┬──────────────────┘
                   │  (B, 64, L)
          ┌────────┴────────┐
          │                 │
    DenoisingDecoder  ClassificationHead
      (B, L)            (B, num_classes)

    Attributes:
        encoder: 共享特征编码器。
        denoiser: 去噪重构解码器。
        classifier: 信号分类头。
    """

    def __init__(
        self,
        input_channels: int,
        signal_length: int,
        num_classes: int = 5,
        output_channels: int = 2, # 新增输出通道参数
        base_filters: int = 48,
        dropout_rate: float = 0.15
    ):
        super().__init__()
        self.input_channels = input_channels
        self.signal_length = signal_length
        self.num_classes = num_classes

        self.encoder = SharedEncoder(input_channels, base_filters, dropout_rate)
        # 修改点：将 output_channels 传给 DenoisingDecoder
        self.denoiser = DenoisingDecoder(self.encoder.out_channels, output_channels, dropout_rate)
        self.classifier = ClassificationHead(
            self.encoder.out_channels, num_classes, dropout_rate * 2
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming 初始化策略。"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        task: str = "both"
    ) -> Dict[str, torch.Tensor]:
        """前向传播。

        Args:
            x: (B, C, L) 输入特征张量。
            task: 执行哪些任务。
                  "both" — 同时执行去噪 + 分类（训练阶段）。
                  "denoise" — 仅去噪重构（推理阶段）。
                  "classify" — 仅分类。

        Returns:
            Dict[str, Tensor]:
                "denoised": (B, L) 去噪重构信号（若 task 包含去噪）。
                "logits": (B, num_classes) 分类 logits（若 task 包含分类）。
                "encoded": (B, 64, L) 编码器中间特征（始终返回）。
        """
        encoded = self.encoder(x)  # (B, 64, L)

        result = {"encoded": encoded}

        if task in ("both", "denoise"):
            result["denoised"] = self.denoiser(encoded)

        if task in ("both", "classify"):
            result["logits"] = self.classifier(encoded)

        return result

    def count_parameters(self) -> int:
        """可训练参数总量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters_by_component(self) -> Dict[str, int]:
        """分组件统计参数量。"""
        return {
            "encoder": sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
            "denoiser": sum(p.numel() for p in self.denoiser.parameters() if p.requires_grad),
            "classifier": sum(p.numel() for p in self.classifier.parameters() if p.requires_grad),
        }

    def summary(self) -> str:
        """模型结构摘要。"""
        total = self.count_parameters()
        by_comp = self.count_parameters_by_component()
        lines = [
            "=" * 60,
            "  MultitaskWaveletCNN 模型结构摘要",
            "=" * 60,
            f"  输入通道数:       {self.input_channels}",
            f"  信号长度:         {self.signal_length}",
            f"  分类类别数:       {self.num_classes}",
            "-" * 60,
            f"  共享编码器参数:   {by_comp['encoder']:>10,}",
            f"  去噪解码器参数:   {by_comp['denoiser']:>10,}",
            f"  分类头参数:       {by_comp['classifier']:>10,}",
            "-" * 60,
            f"  总可训练参数:     {total:>10,}",
            f"  模型大小 (FP32):  {total * 4 / 1024:.1f} KB",
            "=" * 60,
        ]
        return "\n".join(lines)


# ============================================================================
#  多任务损失函数
# ============================================================================

class MultitaskLoss(nn.Module):
    """不确定性加权多任务损失函数。

    基于 Kendall et al. (2018) "Multi-Task Learning Using Uncertainty to
    Weigh Losses" 方法，通过可学习的 log(σ²) 参数自动平衡两个任务的损失量级。

    总损失 = (1/2σ_A²) * L_denoise + (1/2σ_B²) * L_classify
             + log(σ_A) + log(σ_B)

    其中 L_denoise 和 L_classify 分别为去噪重构损失和分类损失，
    σ_A 和 σ_B 为各任务的可学习噪声参数。

    Attributes:
        log_var_denoise: 去噪任务的 log(σ²) 参数。
        log_var_classify: 分类任务的 log(σ²) 参数。
        smooth_weight: 平滑正则项权重。
    """

    def __init__(self, smooth_weight: float = 0.1):
        super().__init__()
        # 可学习的 log(σ²) 参数，初始化为 0 → σ² = 1
        self.log_var_denoise = nn.Parameter(torch.zeros(1))
        self.log_var_classify = nn.Parameter(torch.zeros(1))
        self.smooth_weight = smooth_weight

    def forward(
        self,
        denoised: torch.Tensor,
        clean_target: torch.Tensor,
        logits: torch.Tensor,
        label_target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        mse_loss = F.mse_loss(denoised, clean_target)

        # 修改点：适配三维张量 (Batch, Channels, Length) 的差分计算
        pred_diff = denoised[:, :, 1:] - denoised[:, :, :-1]
        target_diff = clean_target[:, :, 1:] - clean_target[:, :, :-1]
        smooth_loss = F.l1_loss(pred_diff, target_diff)

        denoise_loss = mse_loss + self.smooth_weight * smooth_loss
        classify_loss = F.cross_entropy(logits, label_target)

        precision_denoise = torch.exp(-self.log_var_denoise)
        precision_classify = torch.exp(-self.log_var_classify)

        total_loss = (
            precision_denoise * denoise_loss + self.log_var_denoise
            + precision_classify * classify_loss + self.log_var_classify
        )

        return {
            "total": total_loss.squeeze(),
            "denoise": denoise_loss.detach(),
            "classify": classify_loss.detach(),
            "weight_denoise": precision_denoise.detach().squeeze(),
            "weight_classify": precision_classify.detach().squeeze(),
        }


# ============================================================================
#  旧版兼容：单任务 WaveletCNN（保留向后兼容）
# ============================================================================

class WaveletCNN(nn.Module):
    """单任务信号拟合网络（向后兼容旧版 model.py）。

    内部委托给 MultitaskWaveletCNN 的 denoise 分支。
    """

    def __init__(self, input_channels: int, signal_length: int,
                 base_filters: int = 48, dropout_rate: float = 0.15):
        super().__init__()
        self.backbone = MultitaskWaveletCNN(
            input_channels=input_channels,
            signal_length=signal_length,
            num_classes=5,
            base_filters=base_filters,
            dropout_rate=dropout_rate
        )
        self.input_channels = input_channels
        self.signal_length = signal_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.backbone(x, task="denoise")
        return result["denoised"]

    def count_parameters(self) -> int:
        return self.backbone.count_parameters()

    def summary(self) -> str:
        return self.backbone.summary()


# ============================================================================
#  模块测试
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  MultitaskWaveletCNN 模型 - 独立测试")
    print("=" * 60)

    B, C, L = 8, 7, 329
    num_classes = 5

    model = MultitaskWaveletCNN(
        input_channels=C,
        signal_length=L,
        num_classes=num_classes
    )
    print(model.summary())

    # 测试前向传播
    dummy_input = torch.randn(B, C, L)
    result = model(dummy_input, task="both")

    print(f"\n  输入形状:      {dummy_input.shape}")
    print(f"  去噪输出形状:  {result['denoised'].shape}")
    print(f"  分类输出形状:  {result['logits'].shape}")
    print(f"  编码特征形状:  {result['encoded'].shape}")

    assert result["denoised"].shape == (B, L)
    assert result["logits"].shape == (B, num_classes)
    print("\n  [测试通过] ✓ 双任务前向传播正常")

    # 测试多任务损失
    criterion = MultitaskLoss()
    clean = torch.randn(B, L)
    labels = torch.randint(0, num_classes, (B,))
    losses = criterion(result["denoised"], clean, result["logits"], labels)

    print(f"\n  总损失:         {losses['total'].item():.6f}")
    print(f"  去噪损失:       {losses['denoise'].item():.6f}")
    print(f"  分类损失:       {losses['classify'].item():.6f}")
    print(f"  去噪权重:       {losses['weight_denoise'].item():.4f}")
    print(f"  分类权重:       {losses['weight_classify'].item():.4f}")
    print("\n  [测试通过] ✓ 多任务损失计算正常")

    # 测试向后兼容
    legacy = WaveletCNN(input_channels=C, signal_length=L)
    legacy_out = legacy(dummy_input)
    assert legacy_out.shape == (B, L)
    print("  [测试通过] ✓ 旧版 WaveletCNN 兼容正常")
