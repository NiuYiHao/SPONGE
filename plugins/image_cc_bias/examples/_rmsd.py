"""CPU-only RMSD figure reproduction from archived plugin measurements.

Both entry points share this implementation. Each plotted pair is one target
structure, evaluated by proper Kabsch alignment of corresponding C-alpha atoms.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
GRAY = "#656B70"
BLUE = "#2379A8"


def load_records(system):
    source = BASE / "data" / "rmsd_records.csv"
    provenance = json.loads((BASE / "data" / "provenance.json").read_text())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != provenance["rmsd_records_sha256"]:
        raise ValueError("RMSD source checksum differs from the archived manifest")
    with source.open(newline="") as handle:
        all_records = list(csv.DictReader(handle))
    if len(all_records) != 60:
        raise ValueError("The archive requires 50 1AKE and 10 FUS target records")
    controls = {int(row["target"]) for row in all_records
                if row["system"] == "1AKE" and row["seen_in_library"] == "true"}
    if controls != {1, 13, 25, 37, 50}:
        raise ValueError("1AKE template-control IDs differ from the frozen split")
    records = [row for row in all_records
               if row["system"] == system and row["seen_in_library"] == "false"]
    records.sort(key=lambda row: int(row["target"]))
    expected = {"1AKE": set(range(1, 51)) - controls, "FUS": set(range(1, 11))}[system]
    if {int(row["target"]) for row in records} != expected or len(records) != len(expected):
        raise ValueError(f"{system} target IDs differ from the frozen split")
    values = np.array([[float(row[key]) for key in
                        ("baseline_rmsd_A", "flexible_rmsd_A")] for row in records])
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("RMSD values must be finite and nonnegative")
    return records, values, provenance


def main(system, selected_target):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/image_cc_bias"),
                        help="Directory for PNG, SVG, PDF, CSV and numerical summary")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records, values, provenance = load_records(system)
    targets = np.array([int(row["target"]) for row in records])
    baseline, flexible = values.T
    chosen = int(np.flatnonzero(targets == selected_target)[0])

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "legend.frameon": False,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    # Single quantitative panel, 183 mm wide; individual targets are paired.
    fig, ax = plt.subplots(figsize=(7.204724409, 4.133858268))
    fig.subplots_adjust(left=0.095, right=0.98, bottom=0.20, top=0.73)
    ax.vlines(targets, np.minimum(baseline, flexible), np.maximum(baseline, flexible),
              color="#C6CDD2", linewidth=0.8, zorder=1)
    ax.scatter(targets - 0.12, baseline, color=GRAY, s=17,
               label="Template matching", zorder=2)
    ax.scatter(targets + 0.12, flexible, color=BLUE, s=20,
               label="Flexible fitting", zorder=3)
    for offset, value, color in [(-0.12, baseline[chosen], GRAY),
                                 (0.12, flexible[chosen], BLUE)]:
        ax.scatter([selected_target + offset], [value], marker="D", s=43,
                   facecolor=color, edgecolor="white", linewidth=0.5, zorder=4)
    ax.set(xlabel="Target structure ID", ylabel="Aligned C-alpha RMSD (Å)",
           xlim=(targets.min() - 1, targets.max() + 1),
           ylim=(0, max(baseline.max(), flexible.max()) * 1.12))
    ax.set_xticks([2, 10, 19, 30, 40, 49] if system == "1AKE" else targets)
    ax.tick_params(length=3, width=0.7)
    ax.grid(axis="y", color="#E8ECEF", linewidth=0.5, zorder=0)
    fig.suptitle(f"{system}: template matching and flexible fitting", y=0.965,
                 fontsize=11, fontweight="bold")
    fig.text(0.5, 0.875,
             f"Case {selected_target}: {baseline[chosen]:.3f} → {flexible[chosen]:.3f} Å",
             ha="center", fontsize=9, color=BLUE)
    fig.legend(handles=ax.get_legend_handles_labels()[0],
               labels=ax.get_legend_handles_labels()[1],
               loc="center", bbox_to_anchor=(0.5, 0.805), ncol=2, fontsize=8)
    unit = "held-out target structures" if system == "1AKE" else "target structures"
    improved = int(np.count_nonzero(flexible < baseline - 1e-5))
    fig.text(0.095, 0.075,
             f"n = {len(records)} {unit}; RMSD improved in {improved}/{len(records)} targets.",
             fontsize=7)
    fig.text(0.095, 0.025,
             "Paired points: one target, one fixed seed (731). Source: archived 2026-09-30 run.",
             fontsize=7, color=GRAY)
    stem = args.out / f"{system.lower()}_rmsd"
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"),
                metadata={"Creator": "SPONGE image_cc_bias RMSD reproduction"})
    plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "system": system, "n_targets": len(records), "rmsd_improved": improved,
        "baseline_mean_rmsd_A": float(baseline.mean()),
        "flexible_mean_rmsd_A": float(flexible.mean()),
        "selected_target": selected_target,
        "selected_baseline_rmsd_A": float(baseline[chosen]),
        "selected_flexible_rmsd_A": float(flexible[chosen]),
        "source_rmsd_records_sha256": provenance["rmsd_records_sha256"],
        "source_plugin_sha256": provenance["source_plugin_sha256"],
        "scope": "Redraw of archived static fitting measurements", "seed": 731,
    }
    stem.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
