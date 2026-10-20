"""论文出图统一样式：中文字体 + 300dpi + 清晰字号。

任何绘图脚本开头 import 本模块即可获得统一的论文级样式：

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    import plot_style  # noqa: F401
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体：优先 SimHei（安装在 ~/.local/share/fonts），回退 Noto CJK
_candidates = ["SimHei", "Noto Sans CJK SC", "Microsoft YaHei", "DejaVu Sans"]
_available = {f.name for f in font_manager.fontManager.ttflist}
_family = next((f for f in _candidates if f in _available), "DejaVu Sans")

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [_family],
        "axes.unicode_minus": False,
        "mathtext.fontset": "dejavusans",
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)
