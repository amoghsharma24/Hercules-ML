# HERCULES Training Experiments

This file records the main training rounds and what was learnt from each one.

## Round 1: `roach_gpu_1500`

This was the first GPU training run using the improved synthetic data process.

Dataset:

- 1500 synthetic images
- 1280 training images
- 220 validation images
- 1051 positive images with cockroaches
- 449 negative images with no cockroach
- Strong augmentation with more blur, distractors, shadow, and noise

Weights:

```text
runs/detect/roach_gpu_1500/weights/best.pt
```

Validation results:

```text
precision 0.815
recall    0.548
mAP50     0.624
mAP50-95  0.312
```

Main takeaway:

The model started learning, but recall was still low. It was missing too many cockroaches.

## Round 2: `roach_gpu_round2`

This round used a larger and cleaner synthetic dataset.

Dataset:

- 2500 synthetic images
- 2111 training images
- 389 validation images
- 1736 positive images
- 764 negative images
- Larger cockroaches
- Less blur
- Fewer hard distractors
- Softer shadows and noise

Weights:

```text
runs/detect/roach_gpu_round2/weights/best.pt
```

Validation results:

```text
precision about 0.900
recall    about 0.757
mAP50     about 0.840
mAP50-95  about 0.489
```

Real OneDrive detections at `conf=0.25`:

```text
Tset 1: 14
Tset 2: 13
Tset 3: 6
Tset 4: 6
Tset 5: 9
Total: 48
```

Main takeaway:

The model improved a lot on synthetic validation. It also made many detections on real frames, but most of these were not real cockroaches.

## Round 3: `roach_gpu_round3_640`

This round fine-tuned the Round 2 model at a larger image size.

Dataset:

- Same dataset as Round 2
- Trained from Round 2 weights
- Used `imgsz=640` and `batch=4`

Weights:

```text
runs/detect/roach_gpu_round3_640/weights/best.pt
```

Validation results:

```text
precision 0.936
recall    0.845
mAP50     0.922
mAP50-95  0.621
```

Real OneDrive detections at `conf=0.25`:

```text
Tset 1: 7
Tset 2: 4
Tset 3: 3
Tset 4: 3
Tset 5: 4
Total: 21
```

Main takeaway:

Round 3 had much better validation results and fewer real-frame detections than Round 2. It was more careful, but it still detected some clutter as cockroaches.

## Real Review Set

A review set was created with:

```powershell
python scripts\build_real_review_set.py
```

Output:

- 60 likely positive candidate frames
- 60 sampled negative frames
- a manifest at `outputs/reports/real_review_manifest.csv`
- a contact sheet at `outputs/reviews/real_review_candidates_sheet.jpg`

Candidate source split:

- Round 2 only: 39 frames
- Round 3 only: 12 frames
- Both models: 9 frames

review result:

None of the 60 candidate detections were clear real cockroaches. They looked like clutter, marks, shadows, straps, and other background objects.

This was still useful because those images became hard negatives.

## Hard Negative Batch

The reviewed false positives and sampled negatives were promoted with:

```powershell
python scripts\promote_real_hard_negatives.py
```

Output:

- 120 real negative images
- 120 empty YOLO label files
- a manifest at `outputs/reports/real_hard_negatives_manifest.csv`

Main takeaway:

Hard negatives are important because they teach the model what not to detect.

## Bigger And Darker Cutout Preview

The synthetic generator was changed so pasted cockroach cutouts could be bigger and darker.

Added controls:

- `--cutout_brightness_min`
- `--cutout_brightness_max`

Preview command:

```powershell
python scripts\generate_synthetic_dataset.py --num 120 --seed 909 --val_ratio 0.15 --clean --negative_ratio 0.25 --motion_blur_ratio 0.18 --hard_negative_ratio 0.25 --shadow_strength 1.10 --noise_strength 1.00 --min_roach_scale 0.22 --max_roach_scale 0.60 --cutout_brightness_min 0.30 --cutout_brightness_max 0.85 --output_width 960 --output_height 540 --background_dir data\background_candidates --out_dir data\synthetic_preview_big_dark
python scripts\create_preview_contact_sheet.py --dataset_dir data\synthetic_preview_big_dark --max_images 80 --out outputs\previews\preview_big_dark_cutouts.jpg
```

Preview sheet:

```text
outputs/previews/preview_big_dark_cutouts.jpg
```

Main takeaway:

The images looked better. The cockroach cutouts were easier for the model to learn from and looked less like bright pasted objects.

## Round 4: `roach_gpu_round4_big_dark_hardneg_512`

This was the strongest model so far.

Dataset:

- 3070 bigger and darker synthetic images were generated before the command timed out
- 120 reviewed real hard negatives were added to the training split
- 2794 training images
- 472 validation images
- training split had 1861 positive images and 933 negative images
- validation split had 322 positive images and 150 negative images

Training command:

```powershell
yolo detect train model=runs\detect\roach_gpu_round3_640\weights\best.pt data=training\roach_dataset.yaml imgsz=512 epochs=35 batch=2 device=0 name=roach_gpu_round4_big_dark_hardneg_512 exist_ok=True patience=8 workers=0
```

The first attempt used `imgsz=640` and `batch=4`, but Windows ran out of memory during training. The successful run used `imgsz=512` and `batch=2`.

Weights:

```text
runs/detect/roach_gpu_round4_big_dark_hardneg_512/weights/best.pt
```

Final validation results:

```text
precision 0.952
recall    0.893
mAP50     0.959
mAP50-95  0.709
```

Real OneDrive detections at `conf=0.25`:

```text
Tset 1: 0
Tset 2: 0
Tset 3: 0
Tset 4: 3
Tset 5: 0
Total: 3
```

Main takeaway:

Round 4 was much better at avoiding false positives on real clutter. Real-frame detections dropped from 48 in Round 2, to 21 in Round 3, to 3 in Round 4.

The 3 remaining detections were checked by eye and still looked like false positives:

```text
Output_Tset_4/cockroach_82.jpg
Output_Tset_4/cockroach_91.jpg
Output_Tset_4/cockroach_92.jpg
```

These should be added to the next hard-negative batch, but 3 images are not enough to justify another full retrain by themselves.

## Overall Result So Far

The model has improved a lot across the four rounds.

Best current model:

```text
runs/detect/roach_gpu_round4_big_dark_hardneg_512/weights/best.pt
```

Best current validation results:

```text
precision 0.952
recall    0.893
mAP50     0.959
mAP50-95  0.709
```

The main missing piece is a real positive dataset. The reviewed OneDrive frames did not contain clear real cockroaches, so the next step is to collect and label clear real cockroach footage.

## Next Step

Collecting 30 to 50 clear real cockroach frames from the same camera setup. Label only the obvious cockroaches. Keep unclear frames out of the positive dataset.

After that, train Round 5 using:

- synthetic data
- reviewed hard negatives
- real labelled cockroach positives
- a small held-out real validation set
