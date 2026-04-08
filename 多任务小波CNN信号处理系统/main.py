#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - 多任务小波CNN：二次去噪 + 波形分类 · 主训练脚本
v2.0 - 合成数据训练 → 真实信号推理
"""

import os, sys, time, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report

from utils import (
    load_signal_data, WaveletFeatureExtractor, SyntheticSignalGenerator,
    get_device_info, classify_real_signal, PROJECT_DIR,
    DEFAULT_WAVELET, DEFAULT_DECOMP_LEVEL,
    SIGNAL_TYPE_MAP, SIGNAL_TYPE_NAMES_CN, NUM_SIGNAL_CLASSES
)
from model import MultitaskWaveletCNN, MultitaskLoss

# ============================================================================
#  配置
# ============================================================================

class Config:
    def __init__(self):
        self.epochs = 300
        self.lr = 1e-3
        self.batch_size = 32
        self.weight_decay = 1e-4
        self.patience = 60
        self.min_delta = 1e-7
        self.scheduler_factor = 0.5
        self.scheduler_patience = 25
        self.min_lr = 1e-6
        self.wavelet = DEFAULT_WAVELET
        self.level = DEFAULT_DECOMP_LEVEL
        self.samples_per_class = 200
        self.noise_min = 0.05
        self.noise_max = 0.25
        self.val_ratio = 0.2
        self.output_dir = PROJECT_DIR
        self.model_path = os.path.join(PROJECT_DIR, "multitask_model.pth")

# ============================================================================
#  数据集
# ============================================================================

class MultitaskDataset(Dataset):
    """多任务数据集：特征→(去噪目标, 分类标签)"""

    def __init__(self, features, clean_signals, labels):
        self.features = torch.FloatTensor(features)
        self.clean = torch.FloatTensor(clean_signals)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.clean[idx], self.labels[idx]

# ============================================================================
#  训练引擎
# ============================================================================

class MultitaskTrainer:
    def __init__(self, model, criterion, config, device):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.config = config
        self.device = device

        all_params = list(model.parameters()) + list(criterion.parameters())
        self.optimizer = optim.AdamW(all_params, lr=config.lr,
                                     weight_decay=config.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=config.scheduler_factor,
            patience=config.scheduler_patience, min_lr=config.min_lr)

        self.history = {"train_total": [], "train_denoise": [], "train_classify": [],
                        "val_total": [], "val_denoise": [], "val_classify": [],
                        "val_acc": []}
        self.best_loss = float("inf")
        self.wait = 0
        self.best_state = None

    def train(self, train_loader, val_loader):
        print(f"\n{'='*55}")
        print(f"  多任务训练开始 | 设备: {self.device}")
        print(f"  Epochs: {self.config.epochs} | LR: {self.config.lr}")
        print(f"{'='*55}")
        t0 = time.time()

        for epoch in range(1, self.config.epochs + 1):
            tr = self._train_epoch(train_loader)
            vl = self._val_epoch(val_loader)

            for k in ["total", "denoise", "classify"]:
                self.history[f"train_{k}"].append(tr[k])
                self.history[f"val_{k}"].append(vl[k])
            self.history["val_acc"].append(vl["acc"])

            self.scheduler.step(vl["total"])

            if vl["total"] < self.best_loss - self.config.min_delta:
                self.best_loss = vl["total"]
                self.wait = 0
                self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                self.wait += 1

            if epoch % 20 == 0 or epoch <= 3 or epoch == self.config.epochs:
                lr = self.optimizer.param_groups[0]["lr"]
                print(f"  Ep {epoch:>3d}/{self.config.epochs} | "
                      f"T:{tr['total']:.5f} D:{tr['denoise']:.5f} C:{tr['classify']:.4f} | "
                      f"V:{vl['total']:.5f} Acc:{vl['acc']:.1%} | "
                      f"LR:{lr:.1e} P:{self.wait}/{self.config.patience}")

            if self.wait >= self.config.patience:
                print(f"  [早停] Epoch {epoch}")
                break

        if self.best_state:
            self.model.load_state_dict(self.best_state)

        torch.save({"model": self.model.state_dict(),
                     "config": vars(self.config),
                     "history": self.history,
                     "best_loss": self.best_loss},
                    self.config.model_path)
        print(f"  模型已保存: {self.config.model_path}")
        print(f"  总用时: {time.time()-t0:.1f}s | 最佳Loss: {self.best_loss:.6f}")
        return self.history

    def _train_epoch(self, loader):
        self.model.train()
        sums = {"total": 0, "denoise": 0, "classify": 0}
        n = 0
        for feat, clean, label in loader:
            feat, clean, label = feat.to(self.device), clean.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(feat, task="both")
            losses = self.criterion(out["denoised"], clean, out["logits"], label)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            bs = feat.size(0)
            for k in sums: sums[k] += losses[k].item() * bs
            n += bs
        return {k: v/n for k, v in sums.items()}

    @torch.no_grad()
    def _val_epoch(self, loader):
        self.model.eval()
        sums = {"total": 0, "denoise": 0, "classify": 0}
        correct, n = 0, 0
        for feat, clean, label in loader:
            feat, clean, label = feat.to(self.device), clean.to(self.device), label.to(self.device)
            out = self.model(feat, task="both")
            losses = self.criterion(out["denoised"], clean, out["logits"], label)
            bs = feat.size(0)
            for k in sums: sums[k] += losses[k].item() * bs
            correct += (out["logits"].argmax(1) == label).sum().item()
            n += bs
        res = {k: v/n for k, v in sums.items()}
        res["acc"] = correct / n
        return res

    @torch.no_grad()
    def predict(self, features):
        self.model.eval()
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        out = self.model(x, task="both")
        denoised = out["denoised"].squeeze(0).cpu().numpy()
        probs = torch.softmax(out["logits"], dim=1).squeeze(0).cpu().numpy()
        return denoised, probs

# ============================================================================
#  可视化
# ============================================================================

def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=120)

    for ax, key, title in zip(axes, ["total", "denoise", "classify"],
                               ["Total Loss", "Denoise Loss", "Classify Loss"]):
        ax.semilogy(history[f"train_{key}"], label="Train", alpha=0.8)
        ax.semilogy(history[f"val_{key}"], label="Val", alpha=0.8)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log)")
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [图] 训练曲线: {save_path}")


def plot_denoise_result(noisy, wavelet_dn, cnn_dn, clean, idx, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=120)
    x = np.arange(len(noisy))

    ax = axes[0]
    ax.plot(x, noisy, color="#3498db", alpha=0.4, lw=0.8, label="Noisy")
    ax.plot(x, wavelet_dn, color="#2ecc71", lw=1.2, ls="--", label="Wavelet 1st Denoise")
    ax.plot(x, cnn_dn, color="#e74c3c", lw=1.8, label="CNN 2nd Denoise")
    ax.plot(x, clean, color="#2c3e50", lw=1.0, ls=":", label="Clean (Ground Truth)")
    ax.set_title("Secondary Denoising Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    res_w = noisy - wavelet_dn
    res_c = noisy - cnn_dn
    ax.plot(x, res_w, color="#2ecc71", alpha=0.6, lw=0.8, label=f"Wavelet Residual (RMSE={np.sqrt(np.mean((wavelet_dn-clean)**2)):.4f})")
    ax.plot(x, res_c, color="#e74c3c", alpha=0.8, lw=1.0, label=f"CNN Residual (RMSE={np.sqrt(np.mean((cnn_dn-clean)**2)):.4f})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Residual Analysis", fontsize=12); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [图] 去噪对比: {save_path}")

def plot_vi_characteristic(noisy, cnn_dn, clean, save_path):
    """
    绘制电压-电流 (V-I) 特性曲线视图
    假设 index 0 是电流 (I)，index 1 是电压 (V)
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    
    # 1. 原始含噪轨迹 (散点，淡色)
    ax.scatter(noisy[1], noisy[0], color="#3498db", alpha=0.2, s=2, label="Noisy V-I")
    
    # 2. 干净信号轨迹 (参考线)
    ax.plot(clean[1], clean[0], color="#2c3e50", lw=1.5, ls=":", label="Clean Target", zorder=5)
    
    # 3. CNN 去噪后的轨迹 (实线)
    ax.plot(cnn_dn[1], cnn_dn[0], color="#e74c3c", lw=2, label="CNN Denoised V-I", zorder=10)
    
    ax.set_title("Current vs Voltage (V-I Characteristic)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Voltage (V)", fontsize=12)
    ax.set_ylabel("Current (I)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [图] V-I 特性曲线: {save_path}")

def plot_classification(y_true, y_pred, save_path):
    names = [SIGNAL_TYPE_NAMES_CN[i] for i in range(NUM_SIGNAL_CLASSES)]
    cm = confusion_matrix(y_true, y_pred)
    acc = np.sum(y_true == y_pred) / len(y_true)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names,
                yticklabels=names, ax=axes[0])
    axes[0].set_title(f"Confusion Matrix (Acc: {acc:.1%})", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

    per_class = cm.diagonal() / cm.sum(axis=1).clip(1)
    colors = ["#3498db", "#e67e22", "#e74c3c", "#2ecc71", "#9b59b6"]
    axes[1].barh(names, per_class, color=colors)
    axes[1].set_xlim(0, 1.05)
    for i, v in enumerate(per_class):
        axes[1].text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=10)
    axes[1].set_title("Per-class Accuracy", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [图] 分类结果: {save_path}")

# ============================================================================
#  主函数
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="多任务小波CNN：去噪+分类")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--wavelet", type=str, default=DEFAULT_WAVELET)
    p.add_argument("--level", type=int, default=DEFAULT_DECOMP_LEVEL)
    p.add_argument("--samples-per-class", type=int, default=200)
    p.add_argument("--noise-min", type=float, default=0.05)
    p.add_argument("--noise-max", type=float, default=0.25)
    p.add_argument("--file", type=str, default=None)
    p.add_argument("--columns", nargs='+', default=None, help="指定两列数据的列名，空格隔开")
    p.add_argument("--no-real-inference", action="store_true")
    return p.parse_args()


def main():
    print("╔════════════════════════════════════════════════════════╗")
    print("║  多任务小波 CNN 信号处理系统 v2.0                      ║")
    print("║  Task A: 二次去噪重构 | Task B: 波形类型分类           ║")
    print("╚════════════════════════════════════════════════════════╝")

    args = parse_args()
    cfg = Config()
    cfg.epochs = args.epochs
    cfg.lr = args.lr
    cfg.batch_size = args.batch_size
    cfg.wavelet = args.wavelet
    cfg.level = args.level
    cfg.samples_per_class = args.samples_per_class
    cfg.noise_min = args.noise_min
    cfg.noise_max = args.noise_max

    device = get_device_info()

    # ---- 阶段 1: 合成数据集 ----
    print(f"\n{'─'*55}")
    print("  阶段 1/4: 合成数据集构建")
    print(f"{'─'*55}")

    gen = SyntheticSignalGenerator(
        signal_length=329, noise_range=(cfg.noise_min, cfg.noise_max), seed=42)
    noisy_all, clean_all, labels_all = gen.generate_dataset(cfg.samples_per_class)

    # ---- 阶段 2: 小波特征提取 ----
    print(f"\n{'─'*55}")
    print("  阶段 2/4: 批量小波特征提取")
    print(f"{'─'*55}")

    extractor = WaveletFeatureExtractor(wavelet=cfg.wavelet, level=cfg.level)
    features_all, wavelet_dn_all = extractor.extract_features_batch(
        noisy_all, normalize=False)

    n_channels = features_all.shape[1]
    signal_length = features_all.shape[2]
    print(f"  特征形状: {features_all.shape} | 通道数: {n_channels}")

    # ---- 构建数据集、划分训练/验证 ----
    dataset = MultitaskDataset(features_all, clean_all, labels_all)
    val_size = int(len(dataset) * cfg.val_ratio)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device=="cuda"))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0)

    print(f"  训练集: {train_size} | 验证集: {val_size}")

    # ---- 阶段 3: 多任务训练 ----
    print(f"\n{'─'*55}")
    print("  阶段 3/4: 多任务联合训练")
    print(f"{'─'*55}")

    model = MultitaskWaveletCNN(
    input_channels=n_channels, # 这里运行会自动推断为 14
    signal_length=signal_length,
    num_classes=NUM_SIGNAL_CLASSES,
    output_channels=2 # 新增：去噪双通道输出
)

    criterion = MultitaskLoss(smooth_weight=0.1)
    trainer = MultitaskTrainer(model, criterion, cfg, device)
    history = trainer.train(train_loader, val_loader)

    # ---- 阶段 4: 评估 + 可视化 ----
    print(f"\n{'─'*55}")
    print("  阶段 4/4: 评估与可视化")
    print(f"{'─'*55}")

    # 4a. 验证集分类评估
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for feat, clean, label in val_loader:
            out = model(feat.to(device), task="classify")
            all_preds.extend(out["logits"].argmax(1).cpu().tolist())
            all_labels.extend(label.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean()
    print(f"\n  验证集分类准确率: {acc:.1%}")

    # 4b. 去噪效果展示（取验证集第一个样本）
    sample_idx = val_ds.indices[0]
    s_feat = features_all[sample_idx]
    s_noisy = noisy_all[sample_idx]
    s_clean = clean_all[sample_idx]
    s_wdn = wavelet_dn_all[sample_idx]

    cnn_dn, probs = trainer.predict(s_feat)
    pred_cls = probs.argmax()
    true_cls = labels_all[sample_idx]

    print(f"  样本去噪 RMSE: wavelet={np.sqrt(np.mean((s_wdn-s_clean)**2)):.5f} → "
          f"CNN={np.sqrt(np.mean((cnn_dn-s_clean)**2)):.5f}")
    print(f"  样本分类: 真实={SIGNAL_TYPE_NAMES_CN[true_cls]} | "
          f"预测={SIGNAL_TYPE_NAMES_CN[pred_cls]} ({probs[pred_cls]:.1%})")

    # 4c. 画图
    plot_training_curves(history,
        os.path.join(cfg.output_dir, "training_curves.png"))
    plot_denoise_result(
        s_noisy[0], s_wdn[0], cnn_dn[0], s_clean[0], sample_idx,
        os.path.join(cfg.output_dir, "denoise_result.png")
    )
    plot_classification(all_labels, all_preds,
        os.path.join(cfg.output_dir, "classification_result.png"))
    plot_vi_characteristic(
    s_noisy, cnn_dn, s_clean,
    os.path.join(cfg.output_dir, "vi_characteristic.png")
)
    # ---- 真实信号推理（可选）----
    if not args.no_real_inference:
        try:
            print(f"\n{'─'*55}")
            print("  [附加] 真实信号推理")
            print(f"{'─'*55}")

            signal, cols = load_signal_data(file_path=args.file, target_columns=args.columns)
            feat, wdn, names = extractor.extract_features(signal)

            cnn_dn_real, probs_real = trainer.predict(feat)
            pred_real = probs_real.argmax()

            # 启发式参考
            heur_idx, heur_name, heur_scores = classify_real_signal(signal[0])

            print(f"  CNN 分类:  {SIGNAL_TYPE_NAMES_CN[pred_real]} ({probs_real[pred_real]:.1%})")
            print(f"  启发式参考: {heur_name}")

            # 画真实信号去噪图
            orig_norm = feat[0]  # feat[0] 是通道1的原始信号
            fig, ax = plt.subplots(figsize=(14, 5), dpi=120)
            x = np.arange(len(orig_norm))
            ax.plot(x, orig_norm, color="#3498db", alpha=0.5, lw=0.8, label=f"Original ({cols[0]})")
            ax.plot(x, wdn[0], color="#2ecc71", lw=1.2, ls="--", label="Wavelet Denoise")
            ax.plot(x, cnn_dn_real[0], color="#e74c3c", lw=1.8,
                    label=f"CNN Denoise → {SIGNAL_TYPE_NAMES_CN[pred_real]}")
            ax.set_title(f"Real Signal: '{cols[0]}&{cols[1]}' | Classified as: {SIGNAL_TYPE_NAMES_CN[pred_real]}",
                        fontsize=13, fontweight="bold")
            ax.legend(); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            real_path = os.path.join(cfg.output_dir, "real_signal_result.png")
            plt.savefig(real_path, bbox_inches="tight", facecolor="white")
            plt.close()
            print(f"  [图] 真实信号: {real_path}")

        except FileNotFoundError:
            print("  [跳过] 未找到数据文件，跳过真实信号推理。")
        except Exception as e:
            print(f"  [跳过] 真实信号推理失败: {e}")

    # ---- 完成 ----
    print(f"\n{'═'*55}")
    print("  ✓ 全部任务完成！")
    print(f"  ├── 模型:     {cfg.model_path}")
    print(f"  ├── 训练曲线: training_curves.png")
    print(f"  ├── 去噪对比: denoise_result.png")
    print(f"  └── 分类结果: classification_result.png")
    print(f"{'═'*55}")


if __name__ == "__main__":
    main()
