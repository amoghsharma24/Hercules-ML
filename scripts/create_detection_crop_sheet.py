"""Create enlarged crops around model detections for manual review.

Author:
    Amogh Sharma <amoghsharma02@gmail.com>
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw_onedrive_training"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs" / "detect"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "reviews" / "real_review_detection_crops.jpg"
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
class CropItem:
    image_path: Path
    label: str
    model_name: str
    box_index: int
    box: tuple[float, float, float, float]

# Builds the CLI for the crop-sheet generator.
#
# Padding controls how far the zoom extends around each box, and thumb/
# cols control the tile size and grid layout of the output sheet.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make a zoomed review sheet from YOLO labels.")
    parser.add_argument("--raw_root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--runs_root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--padding", type=float, default=2.4)
    parser.add_argument("--max_crops", type=int, default=160)
    parser.add_argument("--thumb", type=int, default=220)
    parser.add_argument("--cols", type=int, default=5)
    return parser.parse_args()

# Gathers supported image files from a folder tree.
def list_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

# Reads YOLO boxes as normalized center and size values.
#
# Malformed lines (fewer than 5 tokens, or non-numeric) are skipped so a
# stray bad label file does not abort the whole sheet.
def read_labels(path: Path) -> list[tuple[float, float, float, float]]:
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            _, cx, cy, bw, bh = parts[:5]
            boxes.append(tuple(map(float, (cx, cy, bw, bh))))
        except ValueError:
            continue
    return boxes

# Collects every saved detection box that should become a zoomed crop.
#
# Each raw set folder is matched to its prediction label files across the
# configured model runs, and every box becomes one CropItem for the sheet.
def collect_crops(raw_root: Path, runs_root: Path) -> list[CropItem]:
    crops: list[CropItem] = []
    for set_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        try:
            set_id = int(set_dir.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        for image_path in list_images(set_dir):
            for model_name, pattern in MODEL_RUN_PATTERNS.items():
                label_path = runs_root / pattern.format(set_id=set_id) / "labels" / f"{image_path.stem}.txt"
                for box_index, box in enumerate(read_labels(label_path), start=1):
                    crops.append(
                        CropItem(
                            image_path=image_path,
                            label=f"{set_dir.name}_{image_path.name}",
                            model_name=model_name,
                            box_index=box_index,
                            box=box,
                        )
                    )
    return crops

# Enlarges the region around one detection so small objects are easier to inspect.
#
# The crop is padded by the padding factor (with a 64px floor) and kept
# inside the image bounds, centred on the box.
def crop_around_box(image: Image.Image, box: tuple[float, float, float, float], padding: float) -> Image.Image:
    width, height = image.size
    cx, cy, bw, bh = box
    box_w = max(1.0, bw * width)
    box_h = max(1.0, bh * height)
    crop_w = max(box_w * padding, 64)
    crop_h = max(box_h * padding, 64)
    crop_w = min(crop_w, width)
    crop_h = min(crop_h, height)
    center_x = cx * width
    center_y = cy * height
    left = int(max(0, min(width - crop_w, center_x - crop_w / 2)))
    top = int(max(0, min(height - crop_h, center_y - crop_h / 2)))
    right = int(min(width, left + crop_w))
    bottom = int(min(height, top + crop_h))
    return image.crop((left, top, right, bottom))

# Marks the center of a crop where the model box should be inspected.
#
# The box is drawn inset from the tile edges so it does not clip against
# the tile border where the crop has been padded with black.
def draw_center_box(
    draw: ImageDraw.ImageDraw,
    tile_size: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    width, height = tile_size
    margin = int(min(width, height) * 0.34)
    draw.rectangle((margin, margin, width - margin, height - margin), outline=color, width=3)

# Arranges zoomed detection crops into a labelled review sheet.
#
# Crops are capped at max_crops, each is centred in a fixed tile with a
# caption naming its model and box index, then all tiles are laid out on
# a grid and saved.
def make_sheet(args: argparse.Namespace, crops: list[CropItem]) -> None:
    crops = crops[: args.max_crops]
    if not crops:
        raise SystemExit("No detections found to crop.")

    label_h = 42
    rows = (len(crops) + args.cols - 1) // args.cols
    sheet = Image.new("RGB", (args.cols * args.thumb, rows * (args.thumb + label_h)), (242, 242, 242))
    sheet_draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except OSError:
        font = ImageFont.load_default()

    for index, item in enumerate(crops):
        image = Image.open(item.image_path).convert("RGB")
        crop = crop_around_box(image, item.box, args.padding)
        crop.thumbnail((args.thumb, args.thumb), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (args.thumb, args.thumb), (24, 24, 24))
        pad_x = (args.thumb - crop.width) // 2
        pad_y = (args.thumb - crop.height) // 2
        tile.paste(crop, (pad_x, pad_y))
        draw = ImageDraw.Draw(tile)
        draw_center_box(draw, tile.size, COLORS.get(item.model_name, (255, 35, 35)))

        col = index % args.cols
        row = index // args.cols
        x = col * args.thumb
        y = row * (args.thumb + label_h)
        sheet.paste(tile, (x, y))
        sheet_draw.rectangle((x, y + args.thumb, x + args.thumb, y + args.thumb + label_h), fill=(255, 255, 255))
        caption = f"{index:03d} {item.model_name} b{item.box_index}"
        name = item.label
        sheet_draw.text((x + 5, y + args.thumb + 5), caption, fill=(15, 15, 15), font=font)
        sheet_draw.text((x + 5, y + args.thumb + 22), name[:36], fill=(15, 15, 15), font=font)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=94)

# Creates the crop review sheet from all available prediction labels.
def main() -> None:
    args = parse_args()
    if args.padding <= 0:
        raise SystemExit("--padding must be greater than 0.")
    if args.max_crops <= 0:
        raise SystemExit("--max_crops must be greater than 0.")
    if args.cols <= 0:
        raise SystemExit("--cols must be greater than 0.")
    crops = collect_crops(args.raw_root.resolve(), args.runs_root.resolve())
    make_sheet(args, crops)
    print(f"Wrote {min(len(crops), args.max_crops)} crop(s) to {args.out.resolve()}")


if __name__ == "__main__":
    main()
