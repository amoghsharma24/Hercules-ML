"""Run YOLO cockroach detection on a video file or Raspberry Pi camera.

Author:
    Amogh Sharma <amoghsharma02@gmail.com>
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "videos" / "detections.mp4"

# Builds the command-line interface for the detection script.
#
# Only --model is required. The rest have defaults that make the common
# case work with a single flag: the default camera (index 0), a 0.35
# confidence threshold, the motion gate off, and no saved video.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cockroach detection on video.")
    parser.add_argument("--model", required=True, type=Path, help="Path to YOLO model weights.")
    parser.add_argument(
        "--source",
        default="0",
        help="Video path or camera index. Use 0 for the default camera.",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument(
        "--motion_gate",
        action="store_true",
        help="Only run YOLO when frame-difference motion is detected.",
    )
    parser.add_argument(
        "--save",
        nargs="?",
        const=str(DEFAULT_OUTPUT),
        default=None,
        help="Optional output video path. Defaults to outputs/videos/detections.mp4 when used.",
    )
    return parser.parse_args()

# Opens the input stream, handling both camera indexes and file paths.
#
# The same --source flag takes either form, so a value made only of
# digits is treated as a camera index and anything else as a video file
# path. Raises immediately if the stream cannot be opened.
def open_source(source: str) -> cv2.VideoCapture:
    video_source: int | str = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise SystemExit(f"Video source cannot be opened: {source}")
    return cap

# Creates the output video writer to match the input stream.
#
# The writer must use the same frame size and FPS as the input or OpenCV
# produces a corrupt file, so both are read straight from the capture.
# Falls back to 640x480 and 20 FPS if the capture cannot report them.
def create_writer(cap: cv2.VideoCapture, save_path: Path) -> cv2.VideoWriter:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    save_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(save_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise SystemExit(f"Output video cannot be opened for writing: {save_path}")
    return writer

# Decides whether YOLO should run on the current frame.
#
# Without the motion gate every frame is processed. With it, the frame
# is diffed against a running background model and YOLO only runs when
# enough pixels changed. This saves Raspberry Pi compute on still
# footage, where most frames are empty. The first frame only seeds the
# background model and is never processed.
def should_run_yolo(
    frame, background_model, motion_gate: bool
) -> tuple[bool, float, object]:
    if not motion_gate:
        return True, 1.0, background_model

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if background_model is None:
        return False, 0.0, gray.astype("float")

    cv2.accumulateWeighted(gray, background_model, 0.02)
    diff = cv2.absdiff(gray, cv2.convertScaleAbs(background_model))
    _, threshold = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    motion_score = float((threshold > 0).mean())
    # Skipping still frames is what makes the motion gate worth it on the Pi:
    # a scene with no movement costs one cheap diff instead of a full YOLO pass.
    return motion_score > 0.003, motion_score, background_model

# Draws each detection box and its confidence score onto the frame.
# The label is nudged up so it does not sit on the top edge of the frame.
def draw_detections(frame, detections: list[tuple[int, int, int, int, float]]) -> None:
    for x1, y1, x2, y2, confidence in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"cockroach {confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

# Runs the full detection loop: load model, open source, process frames,
# draw boxes, and report FPS and how many frames YOLO actually ran on.
def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    if not model_path.exists():
        raise SystemExit(f"Model file does not exist: {model_path}")

    model = YOLO(str(model_path))
    cap = open_source(str(args.source))
    writer = create_writer(cap, Path(args.save).resolve()) if args.save else None

    frame_count = 0
    model_runs = 0
    background_model = None
    start_time = time.time()

    print(f"Running detection on source {args.source}.")
    if args.motion_gate:
        print("Motion gate enabled.")
    if writer:
        print(f"Saving annotated video to {Path(args.save).resolve()}.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        run_model, motion_score, background_model = should_run_yolo(
            frame, background_model, args.motion_gate
        )

        detections: list[tuple[int, int, int, int, float]] = []
        if run_model:
            model_runs += 1
            result = model(frame, conf=args.conf, verbose=False)[0]
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                detections.append((x1, y1, x2, y2, confidence))

        draw_detections(frame, detections)
        elapsed = max(time.time() - start_time, 1e-6)
        fps = frame_count / elapsed
        cv2.putText(
            frame,
            f"fps={fps:.1f} motion={motion_score:.4f} detections={len(detections)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        if writer:
            writer.write(frame)

        frame_count += 1
        if frame_count % 100 == 0:
            print(
                f"Processed {frame_count} frames, YOLO ran {model_runs} times, avg FPS={fps:.2f}."
            )

    cap.release()
    if writer:
        writer.release()

    elapsed = max(time.time() - start_time, 1e-6)
    print(f"Finished {frame_count} frames in {elapsed:.1f}s.")
    print(f"Average FPS: {frame_count / elapsed:.2f}")
    print(f"YOLO ran on {model_runs} frame(s).")
    if writer:
        print(f"Saved output video to {Path(args.save).resolve()}.")


if __name__ == "__main__":
    main()
