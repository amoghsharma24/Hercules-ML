"""Extract still frames from Raspberry Pi camera videos.

The saved frames become real Raspberry Pi camera backgrounds for the synthetic
dataset generator.

Author:
    Amogh Sharma <amoghsharma02@gmail.com>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "data" / "videos"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "backgrounds"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

# Builds the CLI for frame extraction.
#
# Defaults point at data/videos/ and data/backgrounds/ so the simplest
# call only needs --every_sec.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract background frames from Raspberry Pi videos."
    )
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help="Directory containing input videos. Default: data/videos/",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for extracted frames. Default: data/backgrounds/",
    )
    parser.add_argument(
        "--every_sec",
        type=float,
        default=2.0,
        help="Save one frame every N seconds.",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=92,
        help="JPEG quality for extracted frames.",
    )
    return parser.parse_args()

# Finds video files that can be sampled for background frames.
def list_videos(video_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )

# Samples frames from one video and saves them as numbered images.
#
# Frames are taken every N seconds (converted to a frame step from the
# video's FPS). If the video reports no FPS, 30 is assumed so the loop
# still runs. Files are named video_stem_000000.jpg so they stay sorted.
def extract_frames(
    video_path: Path, out_dir: Path, every_sec: float, jpeg_quality: int
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Skipping {video_path.name}: video source could not be opened.")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        print(f"Using fallback FPS of {fps:.1f} for {video_path.name}.")

    step = max(1, int(round(fps * every_sec)))
    frame_index = 0
    saved_count = 0

    print(f"Reading {video_path.name} at {fps:.2f} FPS, saving every {step} frames.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % step == 0:
            output_name = f"{video_path.stem}_{saved_count:06d}.jpg"
            output_path = out_dir / output_name
            cv2.imwrite(
                str(output_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
            saved_count += 1

        frame_index += 1

    cap.release()
    print(f"Saved {saved_count} frames from {video_path.name}.")
    return saved_count

# Runs frame extraction across every video in the input folder.
#
# Warns and exits cleanly when the folder is missing or has no videos,
# then loops the extractor over each file and prints a final total.
def main() -> None:
    args = parse_args()
    video_dir = args.video_dir.resolve()
    out_dir = args.out_dir.resolve()

    if args.every_sec <= 0:
        raise SystemExit("--every_sec must be greater than 0.")

    if not video_dir.exists():
        raise SystemExit(f"Video directory does not exist: {video_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    videos = list_videos(video_dir)
    if not videos:
        raise SystemExit(
            f"No videos found in {video_dir}. Add .mp4, .avi, .mov, or .mkv files."
        )

    print(f"Found {len(videos)} video file(s).")
    total = 0
    for video_path in videos:
        total += extract_frames(video_path, out_dir, args.every_sec, args.jpeg_quality)

    print(f"Finished extracting {total} total frame(s) into {out_dir}.")
    print("Review the frames and keep the useful Raspberry Pi backgrounds.")


if __name__ == "__main__":
    main()
