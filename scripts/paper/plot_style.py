"""
Shared matplotlib style for all paper figures.

Usage:
    from plot_style import apply_style, COLORS, single_col_fig, double_col_fig

All figures use LaTeX text rendering, Computer Modern fonts,
consistent sizing for a two-column journal layout.
"""
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib as mpl

# ============================================================================
# Journal dimensions (inches)
# ============================================================================
SINGLE_COL = 3.375   # single-column width (typical for JFM/JCP/PoF)
DOUBLE_COL = 7.0     # double-column width
GOLDEN = (1 + 5**0.5) / 2  # golden ratio ≈ 1.618

# ============================================================================
# Color palette — user-specified, print-friendly
# ============================================================================
# Base colors (from LaTeX \definecolor)
_lightblue = (0.63, 0.74, 0.78)
_seagreen  = (0.18, 0.42, 0.41)
_orange    = (0.85, 0.55, 0.13)
_silver    = (0.69, 0.67, 0.66)
_rust      = (0.72, 0.26, 0.06)
_purp      = (68/255, 14/255, 156/255)

# Viridis-inspired sequential (for heatmaps / continuous)
_c1 = (0.267, 0.004, 0.329)
_c2 = (0.283, 0.141, 0.458)
_c3 = (0.254, 0.265, 0.530)
_c4 = (0.207, 0.372, 0.553)
_c5 = (0.164, 0.471, 0.558)
_c6 = (0.129, 0.566, 0.551)
_c7 = (0.134, 0.659, 0.517)
SEQUENTIAL = [_c1, _c2, _c3, _c4, _c5, _c6, _c7]

def _darken(c, f=0.85):
    return tuple(ci * f for ci in c)

def _lighten(c, f=0.5):
    return tuple(ci + (1 - ci) * f for ci in c)

COLORS = {
    'tbrf':      _rust,          # red-brown — best offline accuracy
    'tbnn':      _seagreen,      # dark teal — best NN
    'pi_tbnn':   _lightblue,     # muted blue — variant of TBNN
    'mlp':       _orange,        # orange — scalar closure
    'mlp_large': _purp,          # purple — larger scalar closure
    'sst':       _silver,        # silver-gray — classical RANS
    'komega':    _darken(_silver),  # darker gray
    'earsm':     _darken(_seagreen, 0.65),  # dark teal
    'baseline':  _lighten(_silver, 0.4),    # light gray
    'dns':       (0.0, 0.0, 0.0),           # black — ground truth
    'gray':      (0.7, 0.7, 0.7),
    'black':     (0.0, 0.0, 0.0),
}

# Ordered for legends
MODEL_ORDER = ['tbrf', 'tbnn', 'pi_tbnn', 'mlp_large', 'mlp',
               'sst', 'earsm', 'komega', 'baseline']

MODEL_LABELS = {
    'tbrf': 'TBRF',
    'tbnn': 'TBNN',
    'pi_tbnn': 'PI-TBNN',
    'mlp': 'MLP',
    'mlp_large': 'MLP-Large',
    'sst': r'SST $k$-$\omega$',
    'komega': r'$k$-$\omega$',
    'earsm': 'EARSM',
    'baseline': 'Baseline',
    'dns': 'DNS',
    'gep': 'GEP',
}

# ============================================================================
# Style application
# ============================================================================
def _has_latex():
    """Check if LaTeX is available for usetex rendering."""
    import shutil
    return shutil.which('latex') is not None and shutil.which('dvipng') is not None


def apply_style():
    """Apply publication-quality matplotlib style.

    Uses full LaTeX rendering if available, otherwise falls back to
    matplotlib's mathtext with Computer Modern fonts (still high quality).
    """
    use_tex = _has_latex()
    tex_settings = {
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
    } if use_tex else {
        'text.usetex': False,
    }
    plt.rcParams.update(tex_settings)
    plt.rcParams.update({
        # Fonts — Computer Modern via mathtext (works without LaTeX install)
        'font.family': 'serif',
        'font.serif': ['CMU Serif', 'Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
        'mathtext.fontset': 'cm',

        # Font sizes (match 10pt article class)
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,

        # Lines and markers
        'lines.linewidth': 1.0,
        'lines.markersize': 3,
        'patch.linewidth': 0.5,

        # Axes
        'axes.linewidth': 0.5,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,

        # Ticks
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.3,
        'ytick.minor.width': 0.3,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.minor.size': 1.5,
        'ytick.minor.size': 1.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',

        # Legend
        'legend.frameon': False,
        'legend.borderpad': 0.2,
        'legend.handlelength': 1.5,
        'legend.handletextpad': 0.4,
        'legend.columnspacing': 1.0,

        # Figure
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,

        # PDF backend
        'pdf.fonttype': 42,  # TrueType (editable in Illustrator)
    })


def single_col_fig(height_ratio=1.0/GOLDEN):
    """Create a single-column figure. Returns (fig, ax)."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * height_ratio))
    return fig, ax


def double_col_fig(height_ratio=0.4):
    """Create a double-column figure. Returns (fig, ax)."""
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * height_ratio))
    return fig, ax


def single_col_grid(nrows, ncols, height_ratio=None):
    """Create a single-column figure with subplot grid."""
    if height_ratio is None:
        height_ratio = nrows * 0.7 / ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(SINGLE_COL, SINGLE_COL * height_ratio))
    return fig, axes


def double_col_grid(nrows, ncols, height_ratio=None):
    """Create a double-column figure with subplot grid."""
    if height_ratio is None:
        height_ratio = nrows * 0.35 / ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(DOUBLE_COL, DOUBLE_COL * height_ratio))
    return fig, axes


def save_fig(fig, path, close=True):
    """Save figure as PDF with tight layout."""
    fig.savefig(path, format='pdf')
    if close:
        plt.close(fig)
    print(f"  Saved: {path}")
