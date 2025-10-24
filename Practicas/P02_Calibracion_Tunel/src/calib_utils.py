import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def velocity_from_dp(dp, rho: float = 1.225):
    """Compute flow speed from differential pressure using Bernoulli.

    Parameters
    - dp: array-like differential pressure [Pa]
    - rho: air density [kg/m^3]

    Returns
    - speed [m/s] as numpy array
    """
    dp = np.asarray(dp)
    return np.sqrt(2 * dp / rho)


# ---- File discovery and parsing ----

FREQ_RE = re.compile(r"(\d+)Hz", re.IGNORECASE)


def find_p2_data_root(start: Optional[Path] = None) -> Path:
    """Locate the 'files/p2' folder starting from 'start' upward.

    Returns absolute Path to the folder. Raises FileNotFoundError if not found.
    """
    start = Path.cwd() if start is None else Path(start)
    for cand in [start, *start.parents]:
        p2 = cand / "files" / "p2"
        if p2.exists() and p2.is_dir():
            return p2.resolve()
    # Also try from repo root relative to this utils file
    here = Path(__file__).resolve()
    for cand in [here.parent.parent.parent, *here.parents]:
        p2 = cand / "files" / "p2"
        if p2.exists() and p2.is_dir():
            return p2.resolve()
    raise FileNotFoundError("Could not find 'files/p2' directory from current or parent paths.")


def extract_frequency_from_name(name: str) -> Optional[int]:
    """Extract integer frequency in Hz from a file or folder name like '35Hz.000001.txt' or '121112151144.35Hz'."""
    m = FREQ_RE.search(name)
    return int(m.group(1)) if m else None


def iter_lda_files(p2_root: Path) -> Iterable[Tuple[int, Path]]:
    """Yield (frequency_hz, file_path) for all .txt files under the provided p2 root.

    The expected structure is files/p2/<timestamp>.<freq>Hz/<freq>Hz.<seq>.txt
    This function is robust: it extracts frequency from either folder or file name.
    """
    for sub in p2_root.rglob("*.txt"):
        if not sub.is_file():
            continue
        # Try file name first, then parent
        freq = extract_frequency_from_name(sub.name) or extract_frequency_from_name(sub.parent.name)
        if freq is None:
            continue
        yield freq, sub


def _detect_header_rows(path: Path) -> int:
    """Detect how many header lines to skip before tabular data starts.

    Looks for a line containing 'Row#' (case-insensitive) and returns its index as skiprows value.
    """
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "Row#" in line:
            return i
    # Fallback: assume first 5 lines are header (as per practice doc)
    return 5


def read_lda_speeds(path: Path, abs_values: bool = True) -> pd.Series:
    """Read LDA1 [m/s] column from a DXEX-like .txt file.

    - Automatically detects header lines to skip.
    - Attempts common delimiters (tab and comma/semicolon).
    - Returns a pandas Series of speeds (optionally absolute values), dropping NaNs.
    """
    skip = _detect_header_rows(path)
    # Try typical delimiters
    for sep in ["\t", ",", ";", None]:  # None lets pandas infer
        try:
            df = pd.read_csv(path, skiprows=skip, sep=sep, engine="python")
            break
        except Exception:
            df = None
    if df is None:
        raise ValueError(f"Failed to read data from {path}")

    # Normalize column names whitespace
    df.columns = [str(c).strip() for c in df.columns]
    # Find the velocity column by common names
    candidates = [
        "LDA1 [m/s]",
        "LDA1[m/s]",
        "LDA1",
        "Velocity [m/s]",
    ]
    col = None
    for c in candidates:
        if c in df.columns:
            col = c
            break
    if col is None:
        # Heuristic: last column that can be cast to float sensibly
        for c in reversed(df.columns.tolist()):
            try:
                pd.to_numeric(df[c])
                col = c
                break
            except Exception:
                continue
    if col is None:
        raise KeyError(f"Velocity column not found in {path}; available columns: {df.columns.tolist()}")

    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if abs_values:
        s = s.abs()
    return s


def summarize_frequencies(p2_root: Optional[Path] = None, abs_values: bool = True) -> pd.DataFrame:
    """Compute summary statistics per frequency across all files.

    Returns DataFrame with columns:
    - frequency_hz
    - mean_speed_ms
    - std_speed_ms
    - n_samples
    - n_files
    - file_means_ms (list)
    - file_stds_ms (list)
    - file_paths (list of strings)
    """
    p2_root = find_p2_data_root() if p2_root is None else p2_root
    buckets: Dict[int, List[Tuple[float, float, int, str]]] = {}
    for freq, fpath in iter_lda_files(p2_root):
        try:
            s = read_lda_speeds(fpath, abs_values=abs_values)
            mu = float(s.mean())
            sd = float(s.std(ddof=1))
            n = int(s.shape[0])
        except Exception:
            continue
        buckets.setdefault(freq, []).append((mu, sd, n, str(fpath)))

    rows = []
    for freq, items in sorted(buckets.items()):
        file_means = [it[0] for it in items]
        file_stds = [it[1] for it in items]
        file_ns = [it[2] for it in items]
        total_n = int(sum(file_ns))
        # Weighted mean by sample count
        if total_n > 0:
            wmean = float(np.average(file_means, weights=file_ns))
        else:
            wmean = float(np.mean(file_means))
        # Pooled std (approx): sqrt(sum((n-1)s^2)/sum(n-1))
        denom = sum(max(n - 1, 1) for n in file_ns)
        if denom > 0:
            pooled_var = sum(max(n - 1, 1) * (sd ** 2) for sd, n in zip(file_stds, file_ns)) / denom
            pooled_std = float(np.sqrt(pooled_var))
        else:
            pooled_std = float(np.std(file_means, ddof=1)) if len(file_means) > 1 else 0.0
        rows.append(
            {
                "frequency_hz": int(freq),
                "mean_speed_ms": wmean,
                "std_speed_ms": pooled_std,
                "n_samples": total_n,
                "n_files": len(items),
                "file_means_ms": file_means,
                "file_stds_ms": file_stds,
                "file_paths": [it[3] for it in items],
            }
        )

    df = pd.DataFrame(rows).sort_values("frequency_hz").reset_index(drop=True)
    return df


# ---- Regression and plotting ----

def fit_linear(x_hz: np.ndarray, y_ms: np.ndarray) -> Tuple[float, float, float]:
    """Fit y = m x + b using least squares; return (m, b, r2)."""
    x = np.asarray(x_hz, dtype=float)
    y = np.asarray(y_ms, dtype=float)
    m, b = np.polyfit(x, y, deg=1)
    yhat = m * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(m), float(b), float(r2)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, out_dir: Path, name: str, dpi: int = 150) -> Tuple[Path, Path]:
    """Save matplotlib figure as PNG and SVG under out_dir with base name 'name'."""
    ensure_dir(out_dir)
    png_path = out_dir / f"{name}.png"
    svg_path = out_dir / f"{name}.svg"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    return png_path, svg_path


def plot_calibration(df: pd.DataFrame, include_errorbars: bool = True, title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Create a scatter plot of mean speed vs frequency with optional error bars.

    Returns (fig, ax).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x = df["frequency_hz"].to_numpy()
    y = df["mean_speed_ms"].to_numpy()
    if include_errorbars and "std_speed_ms" in df.columns:
        yerr = df["std_speed_ms"].to_numpy()
        ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=4, label="Datos (media ± std)")
    else:
        ax.plot(x, y, "o", label="Datos (media)")

    # Fit and add regression line
    m, b, r2 = fit_linear(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = m * x_line + b
    ax.plot(x_line, y_line, "r-", label=f"Ajuste lineal: V = {m:.3f} f + {b:.3f}\nR² = {r2:.4f}")

    ax.set_xlabel("Frecuencia del motor (Hz)")
    ax.set_ylabel("Velocidad media del aire (m/s)")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_frequency_distribution(freq: int, speeds: pd.Series, bins: int = 50) -> Tuple[plt.Figure, plt.Axes]:
    """Plot histogram of instantaneous LDA speeds for a given frequency."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(speeds, bins=bins, alpha=0.7, edgecolor="k")
    ax.set_title(f"Distribución de velocidades LDA | {freq} Hz")
    ax.set_xlabel("|Velocidad| (m/s)")
    ax.set_ylabel("Conteo")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def invert_calibration(speed_ms: float, m: float, b: float) -> float:
    """Given V = m f + b, return required frequency f for desired speed V."""
    if m == 0:
        return float("nan")
    return (float(speed_ms) - float(b)) / float(m)

