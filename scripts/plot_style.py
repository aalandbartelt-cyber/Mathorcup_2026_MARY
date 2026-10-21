"""论文出图统一样式（matplotlib）。

三种风格：
  paper  英文/数字 Times New Roman，中文宋体，浅网格（通用论文风）
  sans   英文/数字 Arial，中文黑体
  m杯    基于 40 篇 M 杯优秀论文 615 张图表实测：Times+宋体、白底、黑框、
         无网格、图例内嵌、宽高比 3:2（最贴近参赛论文观感）

用法：
    import plot_style
    plot_style.set_style("m杯")

    fig, ax = plot_style.fig_ax()                       # 6x4 英寸
    plot_style.plot_confusion_matrix(cm, classes)       # Blues + 数值标注
    plot_style.plot_pr_curves({"dent": (r, p, 0.804)})  # 各类 PR + 粗体总曲线
    plot_style.plot_corr_heatmap(df)                    # hot 配色 + 全格数值
    plot_style.plot_feature_importance(names, values)   # 降序横向彩条
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

_AVAILABLE = {f.name for f in font_manager.fontManager.ttflist}


def _pick(cands, default):
    return next((c for c in cands if c in _AVAILABLE), default)


# 40 篇论文中最常见的四色（训练曲线/多模型对比）
CYCLE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
NAVY = "#00007f"
FIGSIZE = (6, 4)


def set_style(mode: str = "paper") -> None:
    """mode: 'paper' / 'sans' / 'm杯'。"""
    if mode == "sans":
        en = _pick(["Arial"], "DejaVu Sans")
        cn = _pick(["SimHei", "Noto Sans CJK SC"], "DejaVu Sans")
        chain = [en, cn, "DejaVu Sans"]
    else:
        en = _pick(["Times New Roman"], "DejaVu Serif")
        cn = _pick(["SimSun", "NSimSun", "Noto Serif CJK SC"], "DejaVu Serif")
        chain = [en, cn, "DejaVu Serif"]

    m_cup = mode == "m杯"
    plt.rcParams.update(
        {
            "font.family": chain,                  # 逐字形回退：英文优先、中文回退
            "mathtext.fontset": "custom",
            "mathtext.rm": en,
            "mathtext.it": f"{en}:italic",
            "mathtext.bf": f"{en}:bold",
            "mathtext.fallback": "cm",
            "axes.unicode_minus": False,
            "axes.formatter.use_mathtext": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.grid": not m_cup,                # M 杯风格：默认无网格
            "grid.alpha": 0.3,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "black",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": True,
            "legend.loc": "best",
        }
    )
    if m_cup:
        plt.rcParams.update({
            "axes.spines.top": True,
            "axes.spines.right": True,
            "legend.framealpha": 0.9,
            "legend.fancybox": False,
            "legend.edgecolor": "0.7",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        })


set_style()


def fig_ax(width=FIGSIZE[0], height=FIGSIZE[1], **kwargs):
    fig, ax = plt.subplots(figsize=(width, height), **kwargs)
    return fig, ax


def save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_confusion_matrix(cm, classes, ax=None, normalize=False, cmap="Blues",
                          title=None, save_path=None):
    """Blues 配色 + 每格数值标注（M 杯最常见的混淆矩阵画法）。"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
    else:
        fig = ax.figure
    cm = np.asarray(cm)
    data = cm.astype(float) / cm.sum(axis=1, keepdims=True) if normalize else cm
    im = ax.imshow(data, cmap=cmap)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    if title:
        ax.set_title(title)
    thresh = data.max() / 2.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            txt = f"{data[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if data[i, j] > thresh else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if save_path:
        save(fig, save_path)
    return ax


def plot_pr_curves(curves, ax=None, save_path=None):
    """curves: {name: (recall, precision, ap)}；总曲线用粗深蓝，图例带 AP。"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
    else:
        fig = ax.figure
    for i, (name, (recall, precision, ap)) in enumerate(curves.items()):
        if name.lower() in {"all", "all classes", "micro", "总体"}:
            ax.plot(recall, precision, color=NAVY, lw=2.2,
                    label=f"all classes {ap:.3f} mAP@0.5")
        else:
            ax.plot(recall, precision, color=CYCLE[i % len(CYCLE)], lw=1.2,
                    label=f"{name} {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.01)
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    if save_path:
        save(fig, save_path)
    return ax


def plot_corr_heatmap(df, ax=None, cmap="hot", save_path=None, annot=True):
    """相关性热力图：hot 配色 + 全格数值（2024-2 同款）。"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
    else:
        fig = ax.figure
    corr = df.corr(numeric_only=True) if hasattr(df, "corr") else np.asarray(df)
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    labels = list(df.columns) if hasattr(df, "columns") else [str(i) for i in range(len(corr))]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    if annot:
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f"{corr[i][j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(corr[i][j]) < 0.5 else "white")
    ax.set_title("相关性热力图")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if save_path:
        save(fig, save_path)
    return ax


def plot_feature_importance(names, values, ax=None, save_path=None, top_n=15):
    """横向彩条（按重要性降序），2021-1 同款。"""
    order = np.argsort(values)[::-1][:top_n]
    names = [names[i] for i in order]
    vals = [values[i] for i in order]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, max(3.2, 0.32 * len(vals) + 1)))
    else:
        fig = ax.figure
    colors = plt.cm.Spectral(np.linspace(0.15, 0.85, len(vals)))
    ax.barh(range(len(vals))[::-1], vals, color=colors, edgecolor="none")
    ax.set_yticks(range(len(vals))[::-1])
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Relative Importance")
    ax.set_title("特征重要性")
    for y, v in zip(range(len(vals))[::-1], vals):
        ax.text(v, y, f" {v:.2f}", va="center", fontsize=7)
    if save_path:
        save(fig, save_path)
    return ax
