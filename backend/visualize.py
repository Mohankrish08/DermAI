# ==========================================
# visualize.py
# Visualization & Plotting Utilities
# ==========================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                     # headless backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import label_binarize

import config


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


# ==========================================
# 1. Training History Curves
# ==========================================

def plot_training_history(history, tag="FL"):
    """
    Plot loss, accuracy, precision, recall, F1 over rounds.
    """
    rounds  = [h["round"] for h in history]
    losses  = [h["loss"] for h in history]
    accs    = [h["accuracy"] for h in history]
    precs   = [h["precision"] for h in history]
    recalls = [h["recall"] for h in history]
    f1s     = [h["f1"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{tag} — Training Metrics per Round", fontsize=15)

    # Loss
    axes[0].plot(rounds, losses, "o-", color="#e74c3c", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(rounds, accs, "s-", color="#2ecc71", linewidth=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)

    # Precision / Recall / F1
    axes[2].plot(rounds, precs, "^-", label="Precision", linewidth=2)
    axes[2].plot(rounds, recalls, "v-", label="Recall", linewidth=2)
    axes[2].plot(rounds, f1s, "D-", label="F1", linewidth=2)
    axes[2].set_title("Precision / Recall / F1")
    axes[2].set_xlabel("Round")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(RESULTS_DIR, f"{tag}_training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Saved: {path}")


# ==========================================
# 2. Confusion Matrix
# ==========================================

def plot_confusion_matrix(cm, tag="FL"):
    """
    Plot and save a confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{tag} — Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{tag}_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Saved: {path}")


# ==========================================
# 3. Per-Class Accuracy Bar Chart
# ==========================================

def plot_per_class_accuracy(cm, tag="FL"):
    per_class = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(CLASS_NAMES, per_class, color=sns.color_palette("viridis", len(CLASS_NAMES)))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{tag} — Per-Class Accuracy")
    for bar, val in zip(bars, per_class):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{tag}_per_class_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Saved: {path}")


# ==========================================
# 4. Comparison Bar Chart (FL vs FL+DP)
# ==========================================

def plot_comparison(fl_metrics, dp_metrics):
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    fl_vals = [fl_metrics[k] for k in ["accuracy", "precision", "recall", "f1"]]
    dp_vals = [dp_metrics[k] for k in ["accuracy", "precision", "recall", "f1"]]

    x = np.arange(len(labels))
    width = 0.30

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, fl_vals, width, label="FL (No DP)", color="#3498db")
    ax.bar(x + width/2, dp_vals, width, label="FL + DP", color="#e67e22")

    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("FL vs FL+DP — Final Metrics Comparison")
    ax.legend()

    for i, (fv, dv) in enumerate(zip(fl_vals, dp_vals)):
        ax.text(i - width/2, fv + 0.02, f"{fv:.3f}", ha="center", fontsize=8)
        ax.text(i + width/2, dv + 0.02, f"{dv:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "comparison_fl_vs_dp.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Saved: {path}")
