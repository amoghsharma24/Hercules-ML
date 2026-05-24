"""Score and preview real frames before adding them as training backgrounds."""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "raw_onedrive_training"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "background_candidates"
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs" / "reports" / "real_frame_triage.csv"
DEFAULT_SHEET = PROJECT_ROOT / "outputs" / "previews" / "real_frame_candidates.jpg"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class FrameScore:
    source: Path
    width: int
    height: int
    bytes: int
    brightness: float
    contrast: float
    blur: float
    ahash: str
    decision: str
    reason: str
    score: float

# Reading triage thresholds, output paths, and copy options from the command line.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a first-pass candidate set from raw real camera frames."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--contact_sheet", type=Path, default=DEFAULT_SHEET)
    parser.add_argument("--min_blur", type=float, default=18.0)
    parser.add_argument("--min_brightness", type=float, default=20.0)
    parser.add_argument("--max_brightness", type=float, default=238.0)
    parser.add_argument("--max_per_folder", type=int, default=35)
    parser.add_argument("--max_sheet_images", type=int, default=120)
    parser.add_argument("--copy", action="store_true", help="Copy candidate frames to out_dir.")
    return parser.parse_args()

# Finding real image frames under the source folder.
def list_images(source: Path) -> list[Path]:
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

# Creating a small perceptual hash for spotting near-duplicate frames.
def average_hash(gray: np.ndarray) -> str:
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    bits = small > small.mean()
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:016x}"

# Scoring one frame for blur, brightness, contrast, and usefulness.
def score_frame(path: Path, args: argparse.Namespace) -> FrameScore:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return FrameScore(path, 0, 0, path.stat().st_size, 0, 0, 0, "", "reject", "unreadable", 0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    brightness = float(gray.mean())
    contrast = float(gray.std())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    ahash = average_hash(gray)

    reasons = []
    if blur < args.min_blur:
        reasons.append("blurry")
    if brightness < args.min_brightness:
        reasons.append("too_dark")
    if brightness > args.max_brightness:
        reasons.append("too_bright")

    decision = "reject" if reasons else "candidate"
    score = blur + contrast * 2.0 - abs(brightness - 128.0) * 0.25
    return FrameScore(
        source=path,
        width=width,
        height=height,
        bytes=path.stat().st_size,
        brightness=brightness,
        contrast=contrast,
        blur=blur,
        ahash=ahash,
        decision=decision,
        reason=";".join(reasons) if reasons else "ok",
        score=score,
    )

# Selecting diverse high-quality background candidates from each source folder.
def select_candidates(scores: list[FrameScore], max_per_folder: int) -> list[FrameScore]:
    selected: list[FrameScore] = []
    seen_hashes: set[str] = set()
    by_folder: dict[Path, list[FrameScore]] = {}

    for item in scores:
        if item.decision != "candidate":
            continue
        by_folder.setdefault(item.source.parent, []).append(item)

    for folder_items in by_folder.values():
        kept = 0
        for item in sorted(folder_items, key=lambda frame: frame.score, reverse=True):
            if item.ahash in seen_hashes:
                continue
            selected.append(item)
            seen_hashes.add(item.ahash)
            kept += 1
            if kept >= max_per_folder:
                break

    return sorted(selected, key=lambda frame: str(frame.source))

# Writing frame scores and selection decisions into a CSV manifest.
def write_manifest(path: Path, scores: list[FrameScore], selected: set[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "selected",
                "decision",
                "reason",
                "score",
                "blur",
                "brightness",
                "contrast",
                "width",
                "height",
                "bytes",
                "path",
            ]
        )
        for item in scores:
            writer.writerow(
                [
                    item.source in selected,
                    item.decision,
                    item.reason,
                    f"{item.score:.3f}",
                    f"{item.blur:.3f}",
                    f"{item.brightness:.3f}",
                    f"{item.contrast:.3f}",
                    item.width,
                    item.height,
                    item.bytes,
                    item.source,
                ]
            )

# Copying selected background candidates into the training background folder.
def copy_candidates(candidates: list[FrameScore], out_dir: Path, source_root: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(candidates):
        try:
            folder = item.source.parent.relative_to(source_root).as_posix().replace("/", "_")
        except ValueError:
            folder = item.source.parent.name
        prefix = folder if folder and folder != "." else "raw"
        target = out_dir / f"{prefix}_{index:04d}_{item.source.name}"
        shutil.copy2(item.source, target)

# Making a visual sheet of selected real frames for quick review.
def make_contact_sheet(candidates: list[FrameScore], out_path: Path, max_images: int) -> None:
    items = candidates[:max_images]
    if not items:
        return

    thumb_w = 240
    thumb_h = 135
    label_h = 28
    cols = 5
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except OSError:
        font = ImageFont.load_default()

    for index, item in enumerate(items):
        image = Image.open(item.source).convert("RGB")
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (thumb_w, thumb_h), (25, 25, 25))
        x_pad = (thumb_w - image.width) // 2
        y_pad = (thumb_h - image.height) // 2
        tile.paste(image, (x_pad, y_pad))

        col = index % cols
        row = index // cols
        x = col * thumb_w
        y = row * (thumb_h + label_h)
        sheet.paste(tile, (x, y))
        caption = f"{index:03d} blur={item.blur:.0f} {item.source.parent.name}/{item.source.name}"
        draw.rectangle((x, y + thumb_h, x + thumb_w, y + thumb_h + label_h), fill=(255, 255, 255))
        draw.text((x + 4, y + thumb_h + 7), caption[:42], fill=(20, 20, 20), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)

# Running the full real-frame triage workflow.
def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    if not source.exists():
        raise SystemExit(f"Source directory does not exist: {source}")
    if args.max_per_folder <= 0:
        raise SystemExit("--max_per_folder must be greater than 0.")

    image_paths = list_images(source)
    if not image_paths:
        raise SystemExit(f"No images found in {source}")

    scores = [score_frame(path, args) for path in image_paths]
    candidates = select_candidates(scores, args.max_per_folder)
    selected_paths = {item.source for item in candidates}
    write_manifest(args.manifest.resolve(), scores, selected_paths)
    make_contact_sheet(candidates, args.contact_sheet.resolve(), args.max_sheet_images)
    if args.copy:
        copy_candidates(candidates, args.out_dir.resolve(), source)

    print(f"Scored {len(scores)} image(s).")
    print(f"Selected {len(candidates)} candidate background frame(s).")
    print(f"Wrote manifest: {args.manifest.resolve()}")
    print(f"Wrote contact sheet: {args.contact_sheet.resolve()}")
    if args.copy:
        print(f"Copied candidates to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
