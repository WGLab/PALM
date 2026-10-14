#!/usr/bin/env python3
"""Plot PALM occurrence probabilities and stacked class probabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


#: tab20 alone gives only 20 colors, so a figure showing more than 20 classes was
#: guaranteed to draw two different InterPro entries in the same color. Combining
#: the three qualitative maps gives 60.
PALETTE = [
    plt.get_cmap(name)(index)
    for name in ("tab20", "tab20b", "tab20c")
    for index in range(20)
]


def palette_index(label: str) -> int:
    """Deterministic starting slot, so a class keeps its color across figures."""
    return int(hashlib.md5(label.encode()).hexdigest()[:8], 16) % len(PALETTE)


def stable_color(label: str) -> tuple[float, float, float, float]:
    return PALETTE[palette_index(label)]


def assign_color(label: str, taken: set[int]) -> tuple[float, float, float, float]:
    """Pick this label's color, stepping past slots already used in the figure.

    Keeps the hash-derived color whenever it is free, so colors stay stable
    between figures, while guaranteeing that no two classes drawn together are
    given the same color.
    """
    start = palette_index(label)
    for offset in range(len(PALETTE)):
        index = (start + offset) % len(PALETTE)
        if index not in taken:
            taken.add(index)
            return PALETTE[index]
    return PALETTE[start]


def load_result(tsv: Path, npz: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict, dict]:
    table = pd.read_csv(tsv, sep="\t")
    arrays = np.load(npz, allow_pickle=False)
    domain_labels = {int(key): value for key, value in json.loads(str(arrays["domain_labels"])).items()}
    site_labels = {int(key): value for key, value in json.loads(str(arrays["site_labels"])).items()}
    return table, arrays, domain_labels, site_labels


# A probability of exactly 1.0 drawn with ylim (0, 1) lands on the top spine and
# disappears, which is the common case for confidently annotated residues. Pad
# the occurrence axes so the curve stays inside the panel.
OCCURRENCE_YLIM = (-0.04, 1.04)
OCCURRENCE_YTICKS = (0.0, 0.25, 0.5, 0.75, 1.0)


def selected_classes(probabilities: np.ndarray, threshold: float, maximum: int) -> np.ndarray:
    maxima = probabilities.max(axis=0)
    selected = np.flatnonzero(maxima >= threshold)
    if len(selected) > maximum:
        selected = selected[np.argsort(maxima[selected])[-maximum:]]
    return selected[np.argsort(maxima[selected])[::-1]]


def plot_stack(
    axis,
    positions: np.ndarray,
    probabilities: np.ndarray,
    labels: dict[int, dict[str, str]],
    threshold: float,
    maximum: int,
    title: str,
    taken: set[int],
) -> list[Patch]:
    selected = selected_classes(probabilities, threshold, maximum)
    bottom = np.zeros(len(positions))
    handles = []
    for class_id in selected:
        info = labels[int(class_id)]
        label = f"{info['name']} ({info['ipr']})"
        color = assign_color(info["ipr"], taken)
        values = probabilities[:, class_id]
        axis.fill_between(positions, bottom, bottom + values, color=color, linewidth=0)
        bottom += values
        handles.append(Patch(facecolor=color, label=label))
    residual = np.clip(1.0 - bottom, 0.0, 1.0)
    axis.fill_between(positions, bottom, bottom + residual, color="0.90", linewidth=0)
    handles.append(Patch(facecolor="0.90", label="Other classes"))
    axis.set_ylim(0, 1)
    axis.set_ylabel("Class probability")
    axis.set_title(title, loc="left")
    return handles


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tsv", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="PALM residue annotation")
    parser.add_argument("--class-threshold", type=float, default=0.05)
    parser.add_argument("--max-classes", type=int, default=20)
    parser.add_argument("--legend-output", type=Path)
    args = parser.parse_args()

    table, arrays, domain_labels, site_labels = load_result(args.tsv, args.npz)
    positions = table["position"].to_numpy()
    taken: set[int] = set()
    fig, axes = plt.subplots(4, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(
        positions, table["domain_occurrence_probability"], color="#22543D", linewidth=1.6
    )
    axes[0].set_ylim(OCCURRENCE_YLIM)
    axes[0].set_yticks(OCCURRENCE_YTICKS)
    axes[0].set_ylabel("Occurrence")
    axes[0].set_title("Domain occurrence", loc="left")
    domain_handles = plot_stack(
        axes[1],
        positions,
        arrays["domain_class_probabilities"],
        domain_labels,
        args.class_threshold,
        args.max_classes,
        "Domain classes",
        taken,
    )
    axes[2].plot(
        positions, table["site_occurrence_probability"], color="#1E3A5F", linewidth=1.6
    )
    axes[2].set_ylim(OCCURRENCE_YLIM)
    axes[2].set_yticks(OCCURRENCE_YTICKS)
    axes[2].set_ylabel("Occurrence")
    axes[2].set_title("Site occurrence", loc="left")
    site_handles = plot_stack(
        axes[3],
        positions,
        arrays["site_class_probabilities"],
        site_labels,
        args.class_threshold,
        args.max_classes,
        "Site classes",
        taken,
    )
    axes[3].set_xlabel("Residue position")
    fig.suptitle(args.title)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    legend_path = args.legend_output or args.output.with_name(args.output.stem + "_legend" + args.output.suffix)
    legend_figure = plt.figure(figsize=(8, max(2, 0.25 * (len(domain_handles) + len(site_handles)))))
    legend_figure.legend(
        domain_handles + site_handles,
        [f"Domain: {item.get_label()}" for item in domain_handles]
        + [f"Site: {item.get_label()}" for item in site_handles],
        loc="center",
        frameon=False,
        ncol=2,
    )
    legend_figure.savefig(legend_path, dpi=300, bbox_inches="tight")
    plt.close(legend_figure)


if __name__ == "__main__":
    main()

