import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

from priors import FHNMultivariateUniform, UniformND


def _dimension_bounds(prior, dim: int) -> Sequence[float]:
    if isinstance(prior, UniformND):
        return float(prior.lows[dim]), float(prior.highs[dim])
    if isinstance(prior, FHNMultivariateUniform):
        if dim == 0:
            low, high = prior.bounds[0]
            return float(low), float(high)
        if dim == 1:
            low = float(prior.bounds[0, 0] / 4.0)
            high = float(prior.second_upper)
            return low, high
        low, high = prior.bounds[dim - 1]
        return float(low), float(high)
    raise TypeError(f"Unsupported prior type {type(prior)!r} for fixed KDE grid generation")


def build_kde_grids(prior, dims: int, num_points: int = 400) -> np.ndarray:
    grids = []
    for dim in range(dims):
        low, high = _dimension_bounds(prior, dim)
        grids.append(np.linspace(low, high, num_points, dtype=float))
    return np.stack(grids, axis=0)


def evaluate_weighted_kdes(samples: np.ndarray, weights: np.ndarray, grids: np.ndarray) -> np.ndarray:
    dims = samples.shape[1]
    evaluations = np.zeros_like(grids)
    for dim in range(dims):
        vals = samples[:, dim]
        try:
            kde = gaussian_kde(vals, weights=weights)
            evaluations[dim] = kde(grids[dim])
        except Exception:
            evaluations[dim] = np.full_like(grids[dim], np.nan)
    return evaluations


def evaluate_mcmc_kdes(mcmc_chain: Optional[np.ndarray], grids: np.ndarray) -> Optional[np.ndarray]:
    if mcmc_chain is None:
        return None
    chain = np.asarray(mcmc_chain, dtype=float)
    if chain.ndim == 1:
        chain = chain[:, None]
    if chain.shape[0] <= 1:
        return None
    burn = max(1000, chain.shape[0] // 5)
    if burn >= chain.shape[0] - 1:
        burn = chain.shape[0] // 2
    chain = chain[burn:]
    if chain.shape[0] <= 1:
        return None
    dims = chain.shape[1]
    evaluations = np.full_like(grids, np.nan)
    for dim in range(dims):
        vals = chain[:, dim]
        if not np.isfinite(vals).any():
            continue
        try:
            kde = gaussian_kde(vals)
            evaluations[dim] = kde(grids[dim])
        except Exception:
            evaluations[dim] = np.full_like(grids[dim], np.nan)
    return evaluations


def save_kde_results(
    output_path: Path,
    grids: np.ndarray,
    smcabc_kdes: np.ndarray,
    metadata: dict,
    mcmc_kdes: Optional[np.ndarray] = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if mcmc_kdes is None:
        mcmc_kdes = np.full_like(grids, np.nan)
    payload = {
        "grid": grids,
        "smcabc": smcabc_kdes,
        "mcmc": mcmc_kdes,
        "metadata": json.dumps(metadata),
    }
    np.savez_compressed(output_path, **payload)
    return output_path


def load_kde_file(path: Path) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        return {
            "path": path,
            "grid": data["grid"],
            "smcabc": data["smcabc"],
            "mcmc": data["mcmc"],
            "metadata": metadata,
        }


def _build_label(record: dict) -> str:
    meta = record["metadata"]
    parts = [
        meta.get("model", "run"),
        meta.get("summary", ""),
        meta.get("timestamp", ""),
    ]
    label = " ".join([p for p in parts if p])
    if not label:
        label = record["path"].stem
    return label


def plot_saved_kde_runs(paths: Iterable[Path], true_theta: Optional[Sequence[float]] = None) -> None:
    records = [load_kde_file(Path(p)) for p in paths]
    if not records:
        raise ValueError("No KDE files provided for plotting")
    dims = records[0]["grid"].shape[0]
    for record in records[1:]:
        if record["grid"].shape != records[0]["grid"].shape:
            raise ValueError("All KDE files must share the same grid shape")

    if true_theta is None:
        first_theta = records[0]["metadata"].get("true_theta")
        if first_theta is not None:
            true_theta = first_theta

    fig, ax = plt.subplots()
    current_dim = [0]

    def plot_dim(dim_idx: int) -> None:
        ax.clear()
        handles = []
        labels = []
        for idx, record in enumerate(records):
            grid = record["grid"][dim_idx]
            smc = record["smcabc"][dim_idx]
            label = _build_label(record)
            (line,) = ax.plot(grid, smc, lw=2, label=f"{label} SMCABC")
            handles.append(line)
            labels.append(f"{label} SMCABC")
            mcmc = record["mcmc"][dim_idx]
            if np.isfinite(mcmc).any():
                (line_mcmc,) = ax.plot(grid, mcmc, lw=2, linestyle="--", label=f"{label} MCMC")
                handles.append(line_mcmc)
                labels.append(f"{label} MCMC")
        if true_theta is not None and len(true_theta) > dim_idx:
            theta_val = float(true_theta[dim_idx])
            line_theta = ax.axvline(theta_val, color="k", linestyle="--", linewidth=2)
            handles.append(line_theta)
            labels.append("True value")
        ax.set_title(f"Dimension {dim_idx + 1}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(handles, labels)
        fig.canvas.draw_idle()

    def on_scroll(event) -> None:
        if event.button == "up":
            current_dim[0] = (current_dim[0] + 1) % dims
        elif event.button == "down":
            current_dim[0] = (current_dim[0] - 1) % dims
        plot_dim(current_dim[0])

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    plot_dim(current_dim[0])
    plt.show()


__all__ = [
    "build_kde_grids",
    "evaluate_weighted_kdes",
    "evaluate_mcmc_kdes",
    "save_kde_results",
    "plot_saved_kde_runs",
    "load_kde_file",
]
