"""Update model comparison CSV and plot for training experiments."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = PROJECT_ROOT / "outputs" / "reports" / "model_comparison_summary.csv"
PLOT = PROJECT_ROOT / "outputs" / "plots" / "model_comparison_dataset_vs_accuracy.png"

ROWS = [
    {
        "model": "Round 1",
        "images": "1500",
        "imgsz": "416",
        "epochs": "25",
        "precision": "0.81500",
        "recall": "0.54762",
        "map50": "0.62414",
        "map50_95": "0.31180",
        "real_detections_conf25": "",
    },
    {
        "model": "Round 2",
        "images": "2500",
        "imgsz": "416",
        "epochs": "35",
        "precision": "0.88002",
        "recall": "0.75000",
        "map50": "0.83213",
        "map50_95": "0.48940",
        "real_detections_conf25": "48",
    },
    {
        "model": "Round 3",
        "images": "2500",
        "imgsz": "640",
        "epochs": "30",
        "precision": "0.93627",
        "recall": "0.84503",
        "map50": "0.92202",
        "map50_95": "0.62086",
        "real_detections_conf25": "21",
    },
    {
        "model": "Round 4",
        "images": "3266",
        "imgsz": "512",
        "epochs": "35",
        "precision": "0.95213",
        "recall": "0.89279",
        "map50": "0.95917",
        "map50_95": "0.70861",
        "real_detections_conf25": "3",
    },
]

# Writing the experiment metrics into the comparison CSV file.
def write_summary() -> None:
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROWS[0]))
        writer.writeheader()
        writer.writerows(ROWS)

# Plotting dataset size against validation metrics for each model round.
def plot() -> None:
    PLOT.parent.mkdir(parents=True, exist_ok=True)
    models = [row["model"] for row in ROWS]
    images = [int(row["images"]) for row in ROWS]
    map50 = [float(row["map50"]) for row in ROWS]
    map50_95 = [float(row["map50_95"]) for row in ROWS]
    recall = [float(row["recall"]) for row in ROWS]

    fig, ax1 = plt.subplots(figsize=(10, 5.8))
    ax2 = ax1.twinx()

    ax1.bar(models, images, color="#c8d5df", label="Dataset images")
    ax2.plot(models, map50, color="#15803d", marker="o", linewidth=2.5, label="mAP50")
    ax2.plot(models, map50_95, color="#1d4ed8", marker="o", linewidth=2.5, label="mAP50-95")
    ax2.plot(models, recall, color="#be123c", marker="o", linewidth=2.5, label="Recall")

    ax1.set_ylabel("Training dataset images")
    ax2.set_ylabel("Validation score")
    ax2.set_ylim(0, 1.0)
    ax1.set_title("Cockroach Detector: Dataset Size vs Accuracy")
    ax1.grid(axis="y", alpha=0.25)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    for idx, value in enumerate(map50_95):
        ax2.annotate(f"{value:.3f}", (idx, value), textcoords="offset points", xytext=(0, 8), ha="center")

    fig.tight_layout()
    fig.savefig(PLOT, dpi=160)
    plt.close(fig)

# Updating both the comparison table and graph together.
def main() -> None:
    write_summary()
    plot()
    print(f"Wrote {SUMMARY}")
    print(f"Wrote {PLOT}")


if __name__ == "__main__":
    main()
