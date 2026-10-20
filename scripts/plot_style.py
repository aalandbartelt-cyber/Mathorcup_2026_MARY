"""论文出图统一样式。

paper 模式(默认，写论文用):
  英文/数字 -> Times New Roman，中文 -> 宋体(SimSun)，公式用 Times 数学字体
sans 模式:
  英文/数字 -> Arial，中文 -> 黑体(SimHei)

注意(2026-10-20 实测): 含中文的文本不要与 $...$ 数学模式混排——mathtext 会接管整串字符串,而其字体(Times)没有中文字形,中文会显示成方框。
需要上标时用 Unicode 字符(如 px²、10⁻¹),或拆成两个独立文本对象。

用法:
    import plot_style                 # 导入即应用 paper 样式
    plot_style.set_style("sans")      # 需要黑体风格时切换
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_AVAILABLE = {f.name for f in font_manager.fontManager.ttflist}


def _pick(cands, default):
    return next((c for c in cands if c in _AVAILABLE), default)


def set_style(mode: str = "paper") -> None:
    """mode: 'paper' (Times+宋体) 或 'sans' (Arial+黑体)。"""
    if mode == "sans":
        en = _pick(["Arial"], "DejaVu Sans")
        cn = _pick(["SimHei", "Noto Sans CJK SC"], "DejaVu Sans")
        family, chain, math_fallback = "sans-serif", [en, cn, "DejaVu Sans"], "cm"
    else:
        en = _pick(["Times New Roman"], "DejaVu Serif")
        cn = _pick(["SimSun", "NSimSun", "Noto Serif CJK SC"], "DejaVu Serif")
        family, chain, math_fallback = "serif", [en, cn, "DejaVu Serif"], "cm"

    plt.rcParams.update(
        {
            "font.family": chain,
            "mathtext.fontset": "custom",
            "mathtext.rm": en,
            "mathtext.it": f"{en}:italic",
            "mathtext.bf": f"{en}:bold",
            "mathtext.fallback": math_fallback,
            "axes.unicode_minus": False,
            "axes.formatter.use_mathtext": False,
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


set_style()
