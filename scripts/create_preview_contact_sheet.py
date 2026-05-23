"""Create a labelled contact sheet from a generated YOLO dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "synthetic_dataset"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "previews" / "preview_contact_sheet.jpg"

# Reading contact-sheet options such as dataset path, output path, and tile layout.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw YOLO boxes onto a preview sheet.")
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max_images", type=int, default=100)
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--thumb_w", type=int, default=240)
    parser.add_argument("--thumb_h", type=int, default=180)
    return parser.parse_args()

# Loading YOLO label lines for one preview image.
def read_labels(label_path: Path) -> list[str]:
    if not label_path.exists():
        return []
    return [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

# Collecting train and validation images with their matching label files.
def collect_items(dataset_dir: Path) -> list[tuple[str, Path, Path]]:
    items = []
    for split in ("train", "val"):
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        for image_path in sorted(image_dir.glob("*.jpg")):
            items.append((split, image_path, label_dir / f"{image_path.stem}.txt"))
    return items

# Drawing normalized YOLO boxes onto a resized preview tile.
def draw_boxes(
    tile: Image.Image,
    labels: list[str],
    source_size: tuple[int, int],
    scale: float,
    pad_x: int,
    pad_y: int,
) -> None:
    draw = ImageDraw.Draw(tile)
    source_w, source_h = source_size
    for line in labels:
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = parts
        cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
        x1 = (cx - bw / 2) * source_w * scale + pad_x
        y1 = (cy - bh / 2) * source_h * scale + pad_y
        x2 = (cx + bw / 2) * source_w * scale + pad_x
        y2 = (cy + bh / 2) * source_h * scale + pad_y
        draw.rectangle((x1, y1, x2, y2), outline=(255, 35, 35), width=3)

# Building the preview contact sheet for a generated YOLO dataset.
def main() -> None:
    args = parse_args()
    if args.max_images <= 0:
        raise SystemExit("--max_images must be greater than 0.")
    if args.cols <= 0:
        raise SystemExit("--cols must be greater than 0.")

    items = collect_items(args.dataset_dir.resolve())[: args.max_images]
    if not items:
        raise SystemExit("No generated images found.")

    label_h = 26
    rows = (len(items) + args.cols - 1) // args.cols
    sheet = Image.new(
        "RGB",
        (args.cols * args.thumb_w, rows * (args.thumb_h + label_h)),
        (245, 245, 245),
    )
    draw_sheet = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    for idx, (split, image_path, label_path) in enumerate(items):
        image = Image.open(image_path).convert("RGB")
        source_w, source_h = image.size
        scale = min(args.thumb_w / source_w, args.thumb_h / source_h)
        new_w = max(1, int(source_w * scale))
        new_h = max(1, int(source_h * scale))
        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (args.thumb_w, args.thumb_h), (32, 32, 32))
        pad_x = (args.thumb_w - new_w) // 2
        pad_y = (args.thumb_h - new_h) // 2
        tile.paste(resized, (pad_x, pad_y))

        labels = read_labels(label_path)
        draw_boxes(tile, labels, image.size, scale, pad_x, pad_y)

        col = idx % args.cols
        row = idx // args.cols
        x = col * args.thumb_w
        y = row * (args.thumb_h + label_h)
        sheet.paste(tile, (x, y))
        caption = f"{split}/{image_path.name} boxes={len(labels)}"
        draw_sheet.rectangle(
            (x, y + args.thumb_h, x + args.thumb_w, y + args.thumb_h + label_h),
            fill=(255, 255, 255),
        )
        draw_sheet.text((x + 5, y + args.thumb_h + 6), caption, fill=(20, 20, 20), font=font)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=92)
    print(f"Wrote {args.out} with {len(items)} samples.")


if __name__ == "__main__":
    main()
