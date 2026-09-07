"""Build a review set from model predictions on real frames.

Author:
    Amogh Sharma <amoghsharma02@gmail.com>
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw_onedrive_training"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs" / "detect"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "real_review"
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs" / "reports" / "real_review_manifest.csv"
DEFAULT_SHEET = PROJECT_ROOT / "outputs" / "reviews" / "real_review_candidates_sheet.jpg"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_RUN_PATTERNS = {
    "round2": "round2_predict_tset_{set_id}",
    "round3_640": "round3_640_predict_tset_{set_id}",
}
COLORS = {
    "round2": (255, 175, 0),
    "round3_640": (35, 190, 255),
}


@dataclass(frozen=True)
class ReviewItem:
    set_name: str
    source_image: Path
    review_name: str
    models: tuple[str, ...]
    box_count: int
    kind: str

# Builds the CLI for the review-set builder.
#
# The negative_per_set count and the seed are exposed so the sampled
# negative queue can be adjusted and reproduced.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a review queue from saved YOLO predictions on real frames."
    )
    parser.add_argument("--raw_root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--runs_root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--contact_sheet", type=Path, default=DEFAULT_SHEET)
    parser.add_argument("--negative_per_set", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_sheet_images", type=int, default=120)
    return parser.parse_args()

# Finds supported image files inside a folder tree.
def list_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

# Loads YOLO label lines while skipping blank rows.
def read_labels(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

# Collects prediction labels for one image across the saved model runs.
#
# Each model has its own label folder, so this returns a map of
# model name -> label lines, omitting models that found nothing.
def prediction_labels_for(
    runs_root: Path, set_id: int, image_stem: str
) -> dict[str, list[str]]:
    predictions: dict[str, list[str]] = {}
    for model_name, pattern in MODEL_RUN_PATTERNS.items():
        label_path = runs_root / pattern.format(set_id=set_id) / "labels" / f"{image_stem}.txt"
        labels = read_labels(label_path)
        if labels:
            predictions[model_name] = labels
    return predictions

# Copies a source frame into the review image folder.
def copy_image(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

# Writes YOLO labels, including empty files for negative images.
#
# An empty file is still written for negatives so every review image has
# a matching label file, which keeps the folder structure predictable.
def write_labels(path: Path, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")

# Builds the positive review queue and sampled negative queue from real frames.
#
# Each raw set is walked: images any model flagged become review
# candidates (with that model's pseudo-labels copied alongside), and a
# random sample of unflagged images becomes negative candidates so the
# reviewer checks both "did we find it" and "did we over-fire".
def build_review_items(args: argparse.Namespace) -> list[ReviewItem]:
    rng = random.Random(args.seed)
    raw_root = args.raw_root.resolve()
    runs_root = args.runs_root.resolve()
    out_root = args.out_root.resolve()

    if not raw_root.exists():
        raise SystemExit(f"Raw image root does not exist: {raw_root}")
    if not runs_root.exists():
        raise SystemExit(f"Runs root does not exist: {runs_root}")

    candidate_items: list[ReviewItem] = []
    negative_items: list[ReviewItem] = []

    for set_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        try:
            set_id = int(set_dir.name.rsplit("_", 1)[-1])
        except ValueError:
            continue

        set_images = list_images(set_dir)
        unflagged: list[Path] = []

        for image_path in set_images:
            predictions = prediction_labels_for(runs_root, set_id, image_path.stem)
            review_name = f"{set_dir.name}_{image_path.name}"
            if predictions:
                models = tuple(sorted(predictions))
                box_count = sum(len(labels) for labels in predictions.values())
                target_image = out_root / "candidates" / "images" / review_name
                copy_image(image_path, target_image)
                for model_name, labels in predictions.items():
                    label_path = (
                        out_root
                        / "candidates"
                        / f"pseudo_labels_{model_name}"
                        / f"{Path(review_name).stem}.txt"
                    )
                    write_labels(label_path, labels)
                candidate_items.append(
                    ReviewItem(set_dir.name, image_path, review_name, models, box_count, "candidate")
                )
            else:
                unflagged.append(image_path)

        rng.shuffle(unflagged)
        for image_path in sorted(unflagged[: args.negative_per_set]):
            review_name = f"{set_dir.name}_{image_path.name}"
            target_image = out_root / "negative_candidates" / "images" / review_name
            copy_image(image_path, target_image)
            write_labels(
                out_root / "negative_candidates" / "empty_labels" / f"{Path(review_name).stem}.txt",
                [],
            )
            negative_items.append(
                ReviewItem(set_dir.name, image_path, review_name, tuple(), 0, "negative_candidate")
            )

    return sorted(candidate_items, key=lambda item: item.review_name) + sorted(
        negative_items, key=lambda item: item.review_name
    )

# Records every review image into a CSV manifest.
#
# Each row includes the review kind, source, which models flagged it, and
# the action a reviewer should take (verify vs confirm empty).
def write_manifest(path: Path, items: list[ReviewItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "kind",
                "set",
                "review_image",
                "source_image",
                "models_flagged",
                "prediction_box_count",
                "action",
            ]
        )
        for item in items:
            action = (
                "verify boxes, delete false positives, add missed roaches"
                if item.kind == "candidate"
                else "confirm empty unless a missed roach is visible"
            )
            writer.writerow(
                [
                    item.kind,
                    item.set_name,
                    item.review_name,
                    item.source_image,
                    "+".join(item.models) if item.models else "",
                    item.box_count,
                    action,
                ]
            )

# Draws normalized YOLO boxes onto a resized preview tile.
#
# The boxes are denormalized against the source image size, then scaled
# and shifted to match the tile padding so they line up on the preview.
def draw_yolo_boxes(
    draw: ImageDraw.ImageDraw,
    labels: list[str],
    source_size: tuple[int, int],
    scale: float,
    pad_x: int,
    pad_y: int,
    color: tuple[int, int, int],
) -> None:
    source_w, source_h = source_size
    for line in labels:
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = parts[:5]
        cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
        x1 = (cx - bw / 2) * source_w * scale + pad_x
        y1 = (cy - bh / 2) * source_h * scale + pad_y
        x2 = (cx + bw / 2) * source_w * scale + pad_x
        y2 = (cy + bh / 2) * source_h * scale + pad_y
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

# Creates a contact sheet that shows model suggestions for quick manual review.
#
# Only candidate (positive) images are drawn, capped at max_sheet_images.
# Each model's boxes are drawn in its own colour so disagreements between
# the two models are visible at a glance.
def make_contact_sheet(args: argparse.Namespace, items: list[ReviewItem]) -> None:
    candidate_items = [item for item in items if item.kind == "candidate"][: args.max_sheet_images]
    if not candidate_items:
        return

    thumb_w = 260
    thumb_h = 146
    label_h = 34
    cols = 4
    rows = (len(candidate_items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), (242, 242, 242))
    sheet_draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except OSError:
        font = ImageFont.load_default()

    for index, item in enumerate(candidate_items):
        image = Image.open(item.source_image).convert("RGB")
        source_size = image.size
        scale = min(thumb_w / source_size[0], thumb_h / source_size[1])
        new_size = (max(1, int(source_size[0] * scale)), max(1, int(source_size[1] * scale)))
        resized = image.resize(new_size, Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (thumb_w, thumb_h), (28, 28, 28))
        pad_x = (thumb_w - new_size[0]) // 2
        pad_y = (thumb_h - new_size[1]) // 2
        tile.paste(resized, (pad_x, pad_y))

        draw = ImageDraw.Draw(tile)
        try:
            set_id = int(item.set_name.rsplit("_", 1)[-1])
        except ValueError:
            set_id = 0
        predictions = prediction_labels_for(args.runs_root.resolve(), set_id, item.source_image.stem)
        for model_name, labels in predictions.items():
            draw_yolo_boxes(
                draw,
                labels,
                source_size,
                scale,
                pad_x,
                pad_y,
                COLORS.get(model_name, (255, 35, 35)),
            )

        col = index % cols
        row = index // cols
        x = col * thumb_w
        y = row * (thumb_h + label_h)
        sheet.paste(tile, (x, y))
        sheet_draw.rectangle((x, y + thumb_h, x + thumb_w, y + thumb_h + label_h), fill=(255, 255, 255))
        caption = f"{item.review_name} [{'+'.join(item.models)}]"
        sheet_draw.text((x + 5, y + thumb_h + 6), caption[:48], fill=(15, 15, 15), font=font)

    args.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.contact_sheet, quality=92)

# Runs the review-set builder and reports the generated files.
def main() -> None:
    args = parse_args()
    if args.negative_per_set < 0:
        raise SystemExit("--negative_per_set must be 0 or greater.")
    if args.max_sheet_images <= 0:
        raise SystemExit("--max_sheet_images must be greater than 0.")

    items = build_review_items(args)
    write_manifest(args.manifest.resolve(), items)
    make_contact_sheet(args, items)

    candidate_count = sum(1 for item in items if item.kind == "candidate")
    negative_count = sum(1 for item in items if item.kind == "negative_candidate")
    print(f"Wrote {candidate_count} candidate image(s).")
    print(f"Wrote {negative_count} negative candidate image(s).")
    print(f"Wrote manifest: {args.manifest.resolve()}")
    print(f"Wrote contact sheet: {args.contact_sheet.resolve()}")
    print(f"Review folder: {args.out_root.resolve()}")


if __name__ == "__main__":
    main()
