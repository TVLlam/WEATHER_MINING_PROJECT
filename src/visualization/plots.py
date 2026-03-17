"""
Module: visualization/plots.py
Hàm vẽ biểu đồ dùng chung cho toàn bộ dự án.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# Thiết lập style chung
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
})


def plot_distribution(df, columns, ncols=3, title="Phân bố các biến", save_path=None):
    """Vẽ histogram cho nhiều cột."""
    nrows = (len(columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, col in enumerate(columns):
        if i < len(axes):
            df[col].hist(bins=50, ax=axes[i], alpha=0.7, edgecolor="white", color="#2196F3")
            axes[i].set_title(col, fontsize=11, fontweight="bold")
            axes[i].grid(alpha=0.3)

    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_correlation_heatmap(df, columns, title="Ma trận Tương quan", save_path=None):
    """Vẽ heatmap tương quan."""
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[columns].corr()
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0,
                fmt=".2f", linewidths=0.5, square=True, ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_bar_comparison(labels, values_dict, title="So sánh", ylabel="Score",
                        save_path=None):
    """Vẽ biểu đồ bar nhóm so sánh."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.8 / len(values_dict)
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    for i, (name, values) in enumerate(values_dict.items()):
        offset = (i - len(values_dict) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name,
                      color=colors[i % len(colors)], alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_time_series(series, title="Chuỗi thời gian", ylabel="Giá trị",
                     save_path=None):
    """Vẽ biểu đồ chuỗi thời gian."""
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(series.index, series.values, linewidth=1.2, color="#2196F3")
    ax.set_xlabel("Thời gian")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def save_figure(fig, save_path, dpi=150):
    """Lưu figure vào file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path
