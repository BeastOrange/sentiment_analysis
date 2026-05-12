"""可视化工具：中文字体注册 + 现代风格主题。

任何脚本调用 `apply_style()` 后，matplotlib/seaborn 自动支持中文。
所有标题、图例尽量使用中文；保留英文仅在中文不存在的术语处。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns

# macOS / Linux / Windows 常见中文字体（按优先级）
_CN_FONT_CANDIDATES = (
    "PingFang SC",
    "PingFang HK",
    "Heiti SC",
    "Hiragino Sans GB",
    "STHeiti",
    "Songti SC",
    "Source Han Sans CN",
    "Source Han Sans SC",
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "WenQuanYi Zen Hei",
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
)

PALETTE_PRIMARY = ["#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899"]
PALETTE_DIVERGE = ["#1E40AF", "#3B82F6", "#93C5FD", "#FDE68A", "#FB923C", "#DC2626"]
PALETTE_SEQ = ["#EFF6FF", "#BFDBFE", "#60A5FA", "#2563EB", "#1E3A8A"]


def _find_chinese_font() -> str | None:
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _CN_FONT_CANDIDATES:
        if name in available:
            return name
    for path in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
    ):
        if Path(path).exists():
            try:
                fm.fontManager.addfont(path)
                prop = fm.FontProperties(fname=path)
                return prop.get_name()
            except Exception:
                continue
    return None


_STYLE_APPLIED = False
_CN_FONT_NAME: str | None = None


def apply_style() -> str | None:
    """应用现代化样式 + 中文字体。返回所用字体名（若未找到则 None）。"""
    global _STYLE_APPLIED, _CN_FONT_NAME
    if _STYLE_APPLIED:
        return _CN_FONT_NAME

    sns.set_theme(style="whitegrid", context="notebook")
    cn_font = _find_chinese_font()
    _CN_FONT_NAME = cn_font

    family_list = [cn_font] if cn_font else []
    family_list += ["DejaVu Sans", "Arial", "sans-serif"]

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": family_list,
        "axes.unicode_minus": False,
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#374151",
        "axes.labelcolor": "#1F2937",
        "xtick.color": "#4B5563",
        "ytick.color": "#4B5563",
        "grid.color": "#E5E7EB",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.facecolor": "white",
    })
    _STYLE_APPLIED = True
    return cn_font


def palette(n: int = 6, kind: str = "primary") -> list[str]:
    base = {"primary": PALETTE_PRIMARY, "diverge": PALETTE_DIVERGE, "seq": PALETTE_SEQ}[kind]
    if n <= len(base):
        return base[:n]
    cmap = sns.color_palette("husl", n)
    return [mpl.colors.to_hex(c) for c in cmap]


def save_fig(fig: plt.Figure, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def annotate_bars(ax: plt.Axes, fmt: str = "{:.0f}") -> None:
    for p in ax.patches:
        h = p.get_height()
        if h <= 0:
            continue
        ax.annotate(
            fmt.format(h),
            (p.get_x() + p.get_width() / 2, h),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1F2937",
            xytext=(0, 2),
            textcoords="offset points",
        )
