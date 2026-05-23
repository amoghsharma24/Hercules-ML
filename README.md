# HERCULES Cockroach Detection

This project is a computer vision pipeline for finding cockroaches in video from a Raspberry Pi camera.

The main focus is the training data. The model only works well if it learns from scenes that look like the real camera setup. To do this, the project uses real background frames, transparent cockroach cutouts, automatic YOLO labels, and a small YOLO model that can run on video or a camera stream.

## Why Synthetic Data Is Used

The earlier model did not work well because the training images looked too fake. They did not match the real camera angle, lighting, floor, shadows, or clutter.

This pipeline makes training images automatically. It places cockroach cutouts onto real background frames from the camera setup. This gives us many labelled examples without manually editing every image.

Synthetic data is useful because we can control:

- cockroach size
- position
- rotation
- blur
- lighting
- shadows
- noise
- image quality

Synthetic data is not final proof that the model works. The model still needs to be tested on real Raspberry Pi footage.

## Why Real Camera Backgrounds Matter

Real Raspberry Pi frames include the same floor, room layout, camera angle, lighting, shadows, reflections, and video quality that the model will see later.

Training on these backgrounds helps the model learn the real environment instead of only learning clean or artificial images.

## Why YOLO Is Used

YOLO is a good choice because it finds object boxes quickly. A small model like `yolov8n.pt` is a practical starting point for Raspberry Pi testing.

## Why Motion Detection Is Included

The Raspberry Pi has limited processing power. Many video frames may have no movement at all.

The runtime can use simple motion detection before running YOLO. This means YOLO only runs when something changes in the frame. This can make the system faster on the Raspberry Pi.

## Project Structure

```text
HERCULES-cockroach-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── extract_frames.py
│   ├── generate_synthetic_dataset.py
│   ├── triage_real_frames.py
│   ├── build_real_review_set.py
│   ├── create_detection_crop_sheet.py
│   ├── create_preview_contact_sheet.py
│   ├── promote_real_hard_negatives.py
│   └── update_model_comparison_plot.py
├── runtime/
│   └── detect_video.py
├── training/
│   ├── roach_dataset.yaml
│   └── TRAINING.md
├── docs/
│   └── team_explanation.md
├── data/
│   ├── videos/
│   ├── backgrounds/
│   ├── roach_cutouts/
│   └── synthetic_dataset/
└── outputs/
    ├── logs/
    ├── plots/
    ├── previews/
    ├── reports/
    ├── reviews/
    └── videos/
```

The `data/`, `runs/`, and most `outputs/` folders are not meant to be committed to Git. They can become large and may contain local training data.

## Installing Requirements

Using a virtual environment is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Raspberry Pi, OpenCV can depend on the operating system image. If `opencv-python` causes problems, install the system OpenCV package for the Pi and keep the other requirements the same.

## Extracting Background Frames

Place Raspberry Pi videos in `data/videos/`, then run:

```bash
python scripts/extract_frames.py --every_sec 2
```

The extracted frames are written to `data/backgrounds/`.

After extraction, review the frames. Remove frames that are blurry, blocked, or not useful for the real test setup.

You can also set the input and output folders:

```bash
python scripts/extract_frames.py --video_dir data/videos --out_dir data/backgrounds --every_sec 1
```

## Sorting Real Image Frames

If you have exported image frames, place them in `data/raw_onedrive_training/`, then run:

```bash
python scripts/triage_real_frames.py --source data/raw_onedrive_training --copy
```

This creates:

- `outputs/reports/real_frame_triage.csv`
- `data/background_candidates/`
- `outputs/previews/real_frame_candidates.jpg`

The contact sheet helps with quick checking. Remove frames that clearly contain cockroaches or are not useful backgrounds before using them for synthetic generation.

## Building A Real Review Set

After a model has made predictions on real frames, build a review set:

```bash
python scripts/build_real_review_set.py
```

This creates:

- candidate images where the model found something
- sampled negative images where the model found nothing
- a CSV manifest
- a review contact sheet

The pseudo-labels are only hints. They should not be treated as correct labels. A person still needs to check them.

## Generating A Synthetic Dataset

Place transparent cockroach PNG cutouts in `data/roach_cutouts/`, then run:

```bash
python scripts/generate_synthetic_dataset.py --num 5000 --seed 42 --val_ratio 0.15
```

To make a smaller preview first, run:

```bash
python scripts/generate_synthetic_dataset.py --num 300 --seed 77 --val_ratio 0.15 --clean --negative_ratio 0.30 --motion_blur_ratio 0.20 --hard_negative_ratio 0.30 --shadow_strength 1.10 --noise_strength 1.00 --min_roach_scale 0.22 --max_roach_scale 0.60 --cutout_brightness_min 0.30 --cutout_brightness_max 0.85 --background_dir data/background_candidates
python scripts/create_preview_contact_sheet.py --max_images 100 --out outputs/previews/preview_contact_sheet.jpg
```

The generator reads:

- backgrounds from `data/backgrounds/`
- cockroach cutouts from `data/roach_cutouts/`

It writes:

- training images to `data/synthetic_dataset/images/train/`
- validation images to `data/synthetic_dataset/images/val/`
- YOLO labels to `data/synthetic_dataset/labels/train/` and `data/synthetic_dataset/labels/val/`

Each label uses YOLO format:

```text
class_id center_x center_y width height
```

The box values are normalised from 0 to 1. Empty label files are used for images with no cockroach. These negative examples teach the model that normal background clutter should not always cause a detection.

Useful realism settings:

- `--motion_blur_ratio` controls how often blur is added.
- `--hard_negative_ratio` controls how often distractors are added.
- `--shadow_strength` controls contact shadow strength.
- `--noise_strength` controls camera-style noise.
- `--min_roach_scale` and `--max_roach_scale` control cockroach size.
- `--cutout_brightness_min` and `--cutout_brightness_max` make pasted cockroaches darker so they look less like bright cutouts.

## Training YOLO

After generating the dataset, train a small YOLO model:

```bash
yolo detect train model=yolov8n.pt data=training/roach_dataset.yaml imgsz=416 epochs=50 batch=16
```

The best model is usually saved at:

```text
runs/detect/train/weights/best.pt
```

For Raspberry Pi deployment, export a smaller runtime format:

```bash
yolo export model=runs/detect/train/weights/best.pt format=ncnn imgsz=416
```

## Running Detection

Run detection on a video:

```bash
python runtime/detect_video.py --model runs/detect/train/weights/best.pt --source data/videos/test.mp4 --conf 0.35 --motion_gate --save
```

Run detection on the default camera:

```bash
python runtime/detect_video.py --model runs/detect/train/weights/best.pt --source 0 --conf 0.35 --motion_gate
```

If `--save` is used without a path, the output video is written to:

```text
outputs/videos/detections.mp4
```

## Limitations

- Synthetic cockroaches may not fully match real cockroaches.
- The model can overfit if the cutout set is too small.
- Backgrounds should come from the same camera angle as the real system.
- Motion detection may miss very slow movement.
- Real performance must be tested on real Raspberry Pi footage, not only synthetic validation images.

## Next Steps

- Collect more Raspberry Pi footage under different lighting.
- Add more real cockroach cutouts with different poses and sizes.
- Label clear real cockroach frames.
- Keep adding false positives as hard negatives.
- Build a small real validation set for honest testing.
