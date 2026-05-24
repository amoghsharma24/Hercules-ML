"""Promote reviewed false positives into a YOLO hard-negative set."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_ROOT = PROJECT_ROOT / "data" / "real_review"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "real_hard_negatives"
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs" / "reports" / "real_hard_negatives_manifest.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Reading paths for the reviewed images and hard-negative output dataset.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy reviewed false positives into a YOLO negative-only dataset."
    )
    parser.add_argument("--review_root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()

# Finding image files inside one review folder.
def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

# Copying one reviewed negative image and creating its empty YOLO label file.
def copy_with_empty_label(source: Path, out_root: Path, group: str) -> tuple[Path, Path]:
    image_dir = out_root / "images"
    label_dir = out_root / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    target_name = f"{group}_{source.name}"
    target_image = image_dir / target_name
    target_label = label_dir / f"{Path(target_name).stem}.txt"
    shutil.copy2(source, target_image)
    target_label.write_text("", encoding="utf-8")
    return target_image, target_label

# Promoting reviewed false positives and sampled negatives into a YOLO dataset.
def main() -> None:
    args = parse_args()
    review_root = args.review_root.resolve()
    out_root = args.out_root.resolve()
    if not review_root.exists():
        raise SystemExit(f"Review root does not exist: {review_root}")

    groups = {
        "false_positive": review_root / "candidates" / "images",
        "sampled_negative": review_root / "negative_candidates" / "images",
    }
    rows: list[list[str]] = []
    for group, folder in groups.items():
        for source in list_images(folder):
            target_image, target_label = copy_with_empty_label(source, out_root, group)
            rows.append([group, str(source), str(target_image), str(target_label)])

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.resolve().open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "source_image", "target_image", "target_empty_label"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} hard-negative image(s) to {out_root}")
    print(f"Wrote manifest: {args.manifest.resolve()}")


if __name__ == "__main__":
    main()
