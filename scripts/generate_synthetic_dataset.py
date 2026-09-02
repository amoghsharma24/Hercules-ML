"""Generate a synthetic YOLO dataset for cockroach detection.

Author:
    Amogh Sharma <amoghsharma02@gmail.com>
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKGROUND_DIR = PROJECT_ROOT / "data" / "backgrounds"
DEFAULT_CUTOUT_DIR = PROJECT_ROOT / "data" / "roach_cutouts"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "synthetic_dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class RealismConfig:
    motion_blur_ratio: float
    hard_negative_ratio: float
    shadow_strength: float
    noise_strength: float
    min_roach_scale: float
    max_roach_scale: float
    cutout_brightness_min: float
    cutout_brightness_max: float
    output_width: int
    output_height: int

# Builds the command-line interface for the synthetic dataset generator.
#
# Every realism knob has a default so a bare run still produces a usable
# dataset. The ratio-style flags are validated in main() rather than here.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paste cockroach cutouts onto real backgrounds and write YOLO labels."
    )
    parser.add_argument("--num", type=int, default=5000, help="Number of images to create.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Fraction of generated images written to the validation split.",
    )
    parser.add_argument(
        "--negative_ratio",
        type=float,
        default=0.20,
        help="Fraction of images that contain no cockroach.",
    )
    parser.add_argument(
        "--motion_blur_ratio",
        type=float,
        default=0.20,
        help="Fraction of samples or roaches receiving directional motion blur.",
    )
    parser.add_argument(
        "--hard_negative_ratio",
        type=float,
        default=0.40,
        help="Fraction of samples receiving debris or insect-like non-roach distractors.",
    )
    parser.add_argument(
        "--shadow_strength",
        type=float,
        default=1.0,
        help="Multiplier for synthetic contact shadows.",
    )
    parser.add_argument(
        "--noise_strength",
        type=float,
        default=1.0,
        help="Multiplier for background and cutout noise transfer.",
    )
    parser.add_argument(
        "--min_roach_scale",
        type=float,
        default=0.07,
        help="Minimum scale applied to source cockroach cutouts.",
    )
    parser.add_argument(
        "--max_roach_scale",
        type=float,
        default=0.56,
        help="Maximum scale applied to source cockroach cutouts.",
    )
    parser.add_argument(
        "--cutout_brightness_min",
        type=float,
        default=0.30,
        help="Minimum brightness multiplier for pasted cockroach cutouts.",
    )
    parser.add_argument(
        "--cutout_brightness_max",
        type=float,
        default=0.95,
        help="Maximum brightness multiplier for pasted cockroach cutouts.",
    )
    parser.add_argument(
        "--output_width",
        type=int,
        default=0,
        help="Resize generated images to this width before augmentation. Use 0 to keep source size.",
    )
    parser.add_argument(
        "--output_height",
        type=int,
        default=0,
        help="Resize generated images to this height before augmentation. Use 0 to keep source size.",
    )
    parser.add_argument(
        "--background_dir",
        type=Path,
        default=DEFAULT_BACKGROUND_DIR,
        help="Directory containing Raspberry Pi background frames.",
    )
    parser.add_argument(
        "--cutout_dir",
        type=Path,
        default=DEFAULT_CUTOUT_DIR,
        help="Directory containing transparent cockroach PNG cutouts.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="YOLO dataset output directory.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previous generated images and labels before writing the new dataset.",
    )
    return parser.parse_args()

# Lists supported background image files from one directory, sorted by name
# so a run is deterministic given the same folder contents.
def list_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

# Lists transparent cockroach cutouts from one directory.
# Only PNGs are accepted because the cutouts need an alpha channel.
def list_cutouts(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )

# Finds the tight bounding box of the opaque pixels in a cutout.
#
# YOLO labels must wrap the visible roach, not the whole (possibly padded)
# PNG canvas, so the box is taken from the alpha channel directly. Returns
# None when the cutout is fully transparent.
def alpha_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    alpha = np.array(image.split()[-1])
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

# Converts a pixel box into one normalized YOLO label row.
#
# YOLO wants class_id plus center/size expressed as fractions of the
# image width and height, so the box corners are converted and rounded
# to 6 decimals to keep label files short.
def to_yolo_label(box: tuple[int, int, int, int], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    center_x = ((x1 + x2) / 2) / width
    center_y = ((y1 + y2) / 2) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    # YOLO label format: class_id center_x center_y width height.
    return f"0 {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"

# Clamps a value into the 0-255 range used by image channels.
def clamp(value: int) -> int:
    return max(0, min(255, value))

# Crops and flips a background to create camera-view variation.
#
# Most frames are randomly cropped then resized back, which shifts the
# framing slightly so the model does not memorize one fixed angle. A
# random horizontal (and occasionally vertical) flip adds more variety.
# Very small images are returned untouched to avoid upscaling artifacts.
def random_crop_or_resize(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    if width < 64 or height < 64:
        return image

    if rng.random() < 0.75:
        crop_scale = rng.uniform(0.72, 1.0)
        crop_w = max(64, int(width * crop_scale))
        crop_h = max(64, int(height * crop_scale))
        x = rng.randint(0, width - crop_w)
        y = rng.randint(0, height - crop_h)
        image = image.crop((x, y, x + crop_w, y + crop_h)).resize(
            (width, height), Image.Resampling.BICUBIC
        )

    if rng.random() < 0.50:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < 0.20:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    return image

# Adds simple floor, wall, corner, dark-area, or clutter structure.
#
# Synthetic backgrounds start as plain photos, so this overlays faint
# geometric detail (tiles, wall lines, corners) to make the scenes feel
# more like the real camera view the model will see.
def add_surface_structure(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    surface_type = rng.choice(
        ["kitchen_floor", "wall", "corner", "dark_area", "cluttered", "rough"]
    )

    if surface_type == "kitchen_floor":
        gap = rng.randint(28, 80)
        color = rng.choice([(55, 55, 55, 55), (230, 230, 220, 35)])
        for x in range(rng.randint(-gap, gap), width, gap):
            draw.line((x, 0, x, height), fill=color, width=rng.choice([1, 1, 2]))
        for y in range(rng.randint(-gap, gap), height, gap):
            draw.line((0, y, width, y), fill=color, width=rng.choice([1, 1, 2]))
    elif surface_type == "wall":
        for _ in range(rng.randint(3, 9)):
            y = rng.randint(0, height)
            draw.line((0, y, width, y + rng.randint(-12, 12)), fill=(0, 0, 0, 18), width=1)
    elif surface_type == "corner":
        x = rng.randint(width // 5, 4 * width // 5)
        y = rng.randint(height // 5, 4 * height // 5)
        draw.line((x, 0, x + rng.randint(-45, 45), height), fill=(0, 0, 0, 80), width=3)
        draw.line((0, y, width, y + rng.randint(-35, 35)), fill=(255, 255, 255, 22), width=2)
    elif surface_type == "dark_area":
        overlay = add_gradient_shadow(overlay, rng, strong=True)
    elif surface_type == "cluttered":
        overlay = add_debris(overlay, rng, count=rng.randint(12, 35), insect_like=False)
    elif surface_type == "rough":
        overlay = add_rough_texture(overlay, rng, strength=rng.uniform(0.25, 0.55))

    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

# Blends fine noise into a surface so it looks less flat.
def add_rough_texture(image: Image.Image, rng: random.Random, strength: float) -> Image.Image:
    width, height = image.size
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    noise = np.random.normal(128, 45, (height, width)).clip(0, 255).astype(np.uint8)
    arr[:, :, :3] = noise[:, :, None]
    arr[:, :, 3] = int(45 * strength)
    texture = Image.fromarray(arr, "RGBA").filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.5)))
    return Image.alpha_composite(image.convert("RGBA"), texture)

# Paints broad shadows and dark regions across the frame.
#
# Real rooms have uneven lighting, so a soft ellipse or edge gradient is
# composited on top. The strong variant makes darker patches used for
# the "dark area" surface type.
def add_gradient_shadow(image: Image.Image, rng: random.Random, strong: bool) -> Image.Image:
    width, height = image.size
    shadow = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(shadow)
    max_alpha = rng.randint(80, 170) if strong else rng.randint(25, 90)
    if rng.random() < 0.60:
        cx = rng.randint(-width // 4, width + width // 4)
        cy = rng.randint(-height // 4, height + height // 4)
        radius = rng.randint(max(width, height) // 3, max(width, height))
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=max_alpha)
    else:
        x = rng.randint(0, width)
        draw.rectangle((x, 0, width, height), fill=max_alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(rng.randint(35, 90)))
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    layer.putalpha(shadow)
    return Image.alpha_composite(image.convert("RGBA"), layer)

# Applies camera-like color, brightness, blur, noise, and compression.
#
# Real Raspberry Pi footage is not perfectly clean, so this chains the
# individual effects (crop, surface structure, tint, blur, noise) into
# one pass that makes a generated background look like a real frame.
def apply_camera_effects(
    image: Image.Image, rng: random.Random, config: RealismConfig
) -> Image.Image:
    image = random_crop_or_resize(image, rng)
    image = add_surface_structure(image, rng)
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.55, 1.30))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.45, 1.55))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.60, 1.60))

    tint = Image.new(
        "RGB",
        image.size,
        (
            clamp(rng.randint(215, 255)),
            clamp(rng.randint(210, 255)),
            clamp(rng.randint(195, 255)),
        ),
    )
    image = Image.blend(image, tint, rng.uniform(0.00, 0.18))

    if rng.random() < 0.75:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 1.8)))
    if rng.random() < 0.18:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(2.0, 3.8)))

    image = add_background_noise(image, rng, config.noise_strength)
    if rng.random() < 0.55:
        image = add_gradient_shadow(image, rng, strong=rng.random() < 0.35).convert("RGB")

    return image

# Adds sensor-style noise to the full background image.
def add_background_noise(image: Image.Image, rng: random.Random, strength: float) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    sigma = rng.uniform(4, 24) * strength
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

# Re-encodes the image as JPEG to imitate camera compression artifacts.
#
# Round-tripping through a low-quality JPEG adds the blocky noise real
# frames get, which the model should learn to ignore.
def apply_jpeg_compression(image: Image.Image, rng: random.Random) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=rng.randint(48, 92))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

# Applies directional blur to imitate fast movement or camera shake.
#
# A kernel line is built and rotated to a random shallow angle, then
# convolved over the image. Used on both whole frames and cutouts.
def motion_blur(image: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.array(image)
    kernel_size = rng.choice([5, 7, 9, 11, 13])
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    angle = rng.uniform(-35, 35)
    center = (kernel_size / 2 - 0.5, kernel_size / 2 - 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (kernel_size, kernel_size))
    kernel /= max(kernel.sum(), 1e-6)
    blurred = cv2.filter2D(arr, -1, kernel)
    return Image.fromarray(blurred)

# Measures local background color and contrast under a future cutout.
#
# The pasted roach is recolored to match whatever sits underneath it, so
# this reads the mean and spread of the background patch it will cover.
# An empty patch returns neutral values so the math never divides by zero.
def local_stats(background: Image.Image, box: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = box
    patch = np.asarray(background.crop((x1, y1, x2, y2)).convert("RGB")).astype(np.float32)
    if patch.size == 0:
        return np.array([128, 128, 128]), np.array([32, 32, 32])
    return patch.mean(axis=(0, 1)), patch.std(axis=(0, 1)) + 1.0

# Recolours a cutout to match the local background lighting and noise.
#
# The cutout's own colors are shifted toward the background patch's mean
# and stretched to its contrast, then noise and brightness are applied on
# top. This is what stops the roach looking like a bright sticker.
def match_cutout_to_background(
    cutout: Image.Image,
    background: Image.Image,
    x: int,
    y: int,
    rng: random.Random,
    config: RealismConfig,
) -> Image.Image:
    rgb = np.asarray(cutout.convert("RGB")).astype(np.float32)
    alpha = np.asarray(cutout.getchannel("A")).astype(np.float32) / 255.0
    bg_mean, bg_std = local_stats(background, (x, y, x + cutout.width, y + cutout.height))
    mask = alpha > 0.05
    if mask.any():
        fg = rgb[mask]
        fg_mean = fg.mean(axis=0)
        fg_std = fg.std(axis=0) + 1.0
        matched = (rgb - fg_mean) * (bg_std / fg_std) * rng.uniform(0.35, 0.70) + fg_mean
        matched += (bg_mean - fg_mean) * rng.uniform(0.10, 0.32)
        rgb[mask] = matched[mask]

        local_noise = np.random.normal(0, float(bg_std.mean()) * 0.06 * config.noise_strength, rgb.shape)
        rgb[mask] += local_noise[mask]
        rgb[mask] *= rng.uniform(config.cutout_brightness_min, config.cutout_brightness_max)

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    alpha_img = cutout.getchannel("A").filter(ImageFilter.GaussianBlur(rng.uniform(0.25, 0.85)))
    alpha_arr = np.asarray(alpha_img).astype(np.float32)
    alpha_arr *= rng.uniform(0.78, 0.98)
    alpha_img = Image.fromarray(np.clip(alpha_arr, 0, 255).astype(np.uint8), "L")
    return Image.merge("RGBA", (*Image.fromarray(rgb).split(), alpha_img))

# Resizes, rotates, darkens, and blurs a cockroach cutout.
#
# Each cutout is scaled and rotated randomly so the model sees many poses
# and sizes, then brightness/contrast/color are tweaked and blur or
# motion blur applied. The output stays RGBA so alpha is preserved.
def transform_cutout(
    cutout: Image.Image, rng: random.Random, config: RealismConfig
) -> Image.Image:
    scale = rng.uniform(config.min_roach_scale, config.max_roach_scale)
    new_size = (
        max(8, int(cutout.width * scale)),
        max(6, int(cutout.height * scale)),
    )
    cutout = cutout.resize(new_size, Image.Resampling.LANCZOS)
    cutout = cutout.rotate(
        rng.uniform(-180, 180), expand=True, resample=Image.Resampling.BICUBIC
    )

    rgb = cutout.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(
        rng.uniform(config.cutout_brightness_min, config.cutout_brightness_max)
    )
    rgb = ImageEnhance.Contrast(rgb).enhance(rng.uniform(0.62, 1.60))
    rgb = ImageEnhance.Color(rgb).enhance(rng.uniform(0.70, 1.25))
    cutout = Image.merge("RGBA", (*rgb.split(), cutout.split()[-1]))

    if rng.random() < 0.70:
        cutout = cutout.filter(ImageFilter.GaussianBlur(rng.uniform(0.15, 1.8)))
    if rng.random() < config.motion_blur_ratio:
        cutout = motion_blur(cutout, rng)

    return cutout

# Adds a soft contact shadow under a pasted cockroach.
#
# A faint offset copy of the cutout's alpha is blurred and dropped just
# below-right of the roach, anchoring it to the floor so it does not
# float above the scene.
def add_shadow(
    background: Image.Image,
    cutout: Image.Image,
    x: int,
    y: int,
    rng: random.Random,
    config: RealismConfig,
) -> Image.Image:
    shadow_layer = Image.new("RGBA", background.size, (0, 0, 0, 0))
    alpha = cutout.split()[-1]
    opacity = rng.uniform(0.18, 0.58) * config.shadow_strength
    shadow = Image.new("RGBA", cutout.size, (0, 0, 0, clamp(int(115 * config.shadow_strength))))
    shadow.putalpha(alpha.point(lambda pixel: clamp(int(pixel * opacity))))
    shadow = shadow.filter(ImageFilter.GaussianBlur(rng.uniform(2, 10)))
    shadow_layer.alpha_composite(
        shadow,
        (x + rng.randint(1, 14), y + rng.randint(1, 14)),
    )
    return Image.alpha_composite(background, shadow_layer)

# Draws a cable-like line that can act as a hard negative.
#
# Cables look enough like a roach's long body that the model might false
# positive on them, so they are added as clutter the model must ignore.
def draw_cable(draw: ImageDraw.ImageDraw, rng: random.Random, width: int, height: int) -> None:
    points = []
    x = rng.randint(0, width)
    y = rng.randint(0, height)
    for _ in range(rng.randint(3, 6)):
        points.append((x, y))
        x += rng.randint(-width // 4, width // 4)
        y += rng.randint(-height // 4, height // 4)
    draw.line(points, fill=(rng.randint(5, 55), rng.randint(5, 55), rng.randint(5, 55), 180), width=rng.randint(2, 5))

# Draws an insect-like distractor that must not be labelled a cockroach.
#
# The point is to give the model hard negative examples: shapes close to
# a roach but with no label, so it learns to hold off rather than fire.
def draw_insect_like(draw: ImageDraw.ImageDraw, rng: random.Random, width: int, height: int) -> None:
    cx = rng.randint(8, max(8, width - 8))
    cy = rng.randint(8, max(8, height - 8))
    body_w = rng.randint(5, 18)
    body_h = rng.randint(10, 28)
    color = (rng.randint(25, 80), rng.randint(20, 70), rng.randint(10, 45), rng.randint(100, 170))
    draw.ellipse((cx - body_w, cy - body_h, cx + body_w, cy + body_h), fill=color)
    for _ in range(rng.randint(4, 8)):
        dx = rng.choice([-1, 1]) * rng.randint(body_w, body_w + 20)
        dy = rng.randint(-body_h, body_h)
        draw.line((cx, cy, cx + dx, cy + dy), fill=color, width=1)

# Draws leaf-like debris for extra visual clutter.
def draw_leaf(draw: ImageDraw.ImageDraw, rng: random.Random, width: int, height: int) -> None:
    cx = rng.randint(0, width)
    cy = rng.randint(0, height)
    leaf_w = rng.randint(8, 28)
    leaf_h = rng.randint(4, 16)
    color = (rng.randint(45, 105), rng.randint(55, 115), rng.randint(20, 65), rng.randint(90, 160))
    draw.ellipse((cx - leaf_w, cy - leaf_h, cx + leaf_w, cy + leaf_h), fill=color)
    draw.line((cx - leaf_w, cy, cx + leaf_w, cy), fill=(30, 60, 20, 120), width=1)

# Adds random debris and distractors to make negative examples harder.
#
# The kind is picked from a list (crumb, screw, leaf, dirt, stain, cable,
# insect) and scattered over the scene. When insect_like is set, insect
# distractors are included so negatives contain roach-like shapes the
# model has to reject.
def add_debris(
    image: Image.Image, rng: random.Random, count: int, insect_like: bool
) -> Image.Image:
    width, height = image.size
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    choices = ["crumb", "screw", "leaf", "dirt", "stain", "cable", "debris"]
    if insect_like:
        choices.append("insect")

    for _ in range(count):
        kind = rng.choice(choices)
        x = rng.randint(0, width)
        y = rng.randint(0, height)
        if kind == "crumb":
            r = rng.randint(1, 5)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(rng.randint(80, 170), rng.randint(55, 130), rng.randint(25, 75), rng.randint(80, 190)))
        elif kind == "screw":
            r = rng.randint(3, 9)
            draw.ellipse((x - r, y - r, x + r, y + r), outline=(45, 45, 45, 150), width=2)
            draw.line((x - r, y, x + r, y), fill=(35, 35, 35, 150), width=1)
        elif kind == "leaf":
            draw_leaf(draw, rng, width, height)
        elif kind == "dirt":
            r = rng.randint(2, 9)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(rng.randint(15, 80), rng.randint(12, 70), rng.randint(8, 50), rng.randint(45, 135)))
        elif kind == "stain":
            r = rng.randint(10, 45)
            draw.ellipse((x - r, y - r // 2, x + r, y + r // 2), fill=(10, 10, 8, rng.randint(25, 80)))
        elif kind == "cable":
            draw_cable(draw, rng, width, height)
        elif kind == "insect":
            draw_insect_like(draw, rng, width, height)
        else:
            draw.rectangle((x, y, x + rng.randint(2, 12), y + rng.randint(2, 12)), fill=(rng.randint(35, 130), rng.randint(35, 130), rng.randint(35, 130), rng.randint(60, 150)))

    layer = layer.filter(ImageFilter.GaussianBlur(rng.uniform(0.0, 0.6)))
    return Image.alpha_composite(image.convert("RGBA"), layer)

# Places a foreground object over part of the scene to simulate occlusion.
#
# Partially hiding a roach (or its label area) teaches the model to still
# find it when something is in the way.
def add_occluder(
    background: Image.Image,
    roach_box: tuple[int, int, int, int],
    rng: random.Random,
) -> Image.Image:
    x1, y1, x2, y2 = roach_box
    layer = Image.new("RGBA", background.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if rng.random() < 0.55:
        draw_cable(draw, rng, background.width, background.height)
    else:
        ox = rng.randint(max(0, x1 - 12), min(background.width, x2 + 12))
        oy = rng.randint(max(0, y1 - 12), min(background.height, y2 + 12))
        draw.rectangle(
            (ox, oy, ox + rng.randint(5, 28), oy + rng.randint(2, 14)),
            fill=(rng.randint(20, 95), rng.randint(20, 95), rng.randint(20, 95), rng.randint(90, 175)),
        )
    layer = layer.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.9)))
    return Image.alpha_composite(background.convert("RGBA"), layer)

# Chooses a valid paste position for one transformed cockroach.
#
# Roughly half the time the roach is biased toward the edges of the frame
# (where roaches tend to appear) and the rest of the time it is placed
# anywhere, so the model does not learn one corner.
def choose_position(
    background_size: tuple[int, int], cutout_size: tuple[int, int], rng: random.Random
) -> tuple[int, int]:
    width, height = background_size
    cutout_width, cutout_height = cutout_size

    if rng.random() < 0.55:
        x_choices = [
            rng.randint(0, max(0, width // 5)),
            rng.randint(max(0, 4 * width // 5 - cutout_width), max(0, width - cutout_width)),
        ]
        y_choices = [
            rng.randint(0, max(0, height // 5)),
            rng.randint(
                max(0, 4 * height // 5 - cutout_height), max(0, height - cutout_height)
            ),
        ]
        return rng.choice(x_choices), rng.choice(y_choices)

    return rng.randint(0, width - cutout_width), rng.randint(0, height - cutout_height)

# Creates the expected train and val image and label folders.
def prepare_output_dirs(out_dir: Path) -> None:
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

# Removes previously generated images and labels before a clean run.
#
# .gitkeep files are left in place so the empty directories still survive
# in Git when the dataset folder is otherwise ignored.
def clean_output_dirs(out_dir: Path) -> None:
    for folder in (
        out_dir / "images" / "train",
        out_dir / "images" / "val",
        out_dir / "labels" / "train",
        out_dir / "labels" / "val",
    ):
        folder.mkdir(parents=True, exist_ok=True)
        for path in folder.iterdir():
            if path.is_file() and path.name != ".gitkeep":
                path.unlink()

# Generates one synthetic image and its YOLO labels.
#
# One background is picked, passed through camera effects, then zero to
# three cutouts are pasted and labelled. Negative images (no roach) are
# deliberately kept so the model learns that empty scenes are normal.
# Returns True if the image ended up with at least one labelled roach.
def generate_one(
    index: int,
    split: str,
    background_paths: list[Path],
    cutout_paths: list[Path],
    out_dir: Path,
    negative_ratio: float,
    rng: random.Random,
    config: RealismConfig,
) -> bool:
    background = Image.open(rng.choice(background_paths)).convert("RGB")
    if config.output_width > 0 and config.output_height > 0:
        background = background.resize(
            (config.output_width, config.output_height), Image.Resampling.BICUBIC
        )
    background = apply_camera_effects(background, rng, config).convert("RGBA")
    width, height = background.size
    labels: list[str] = []

    if rng.random() < config.hard_negative_ratio:
        background = add_debris(
            background,
            rng,
            count=rng.randint(4, 22),
            insect_like=rng.random() < 0.55,
        )

    # Synthetic data turns limited real footage into many labelled examples.
    roach_count = 0 if rng.random() < negative_ratio else rng.choice([1, 1, 1, 2, 2, 3])

    if roach_count == 0 and rng.random() < 0.85:
        background = add_debris(
            background,
            rng,
            count=rng.randint(10, 35),
            insect_like=True,
        )

    for _ in range(roach_count):
        cutout = Image.open(rng.choice(cutout_paths)).convert("RGBA")
        cutout = transform_cutout(cutout, rng, config)
        if cutout.width >= width or cutout.height >= height:
            continue

        x, y = choose_position((width, height), cutout.size, rng)
        cutout = match_cutout_to_background(cutout, background, x, y, rng, config)
        background = add_shadow(background, cutout, x, y, rng, config)
        background.alpha_composite(cutout, (x, y))

        box = alpha_bbox(cutout)
        if box is None:
            continue

        x1, y1, x2, y2 = box
        label_box = (x + x1, y + y1, x + x2, y + y2)
        labels.append(to_yolo_label(label_box, width, height))

        if rng.random() < 0.28:
            background = add_occluder(background, label_box, rng)

    # Keeping negatives teaches the model that empty scenes are normal.
    final = background.convert("RGB")
    if rng.random() < config.motion_blur_ratio * 0.35:
        final = motion_blur(final, rng)
    final = apply_jpeg_compression(final, rng)

    image_path = out_dir / "images" / split / f"synth_{index:06d}.jpg"
    label_path = out_dir / "labels" / split / f"synth_{index:06d}.txt"
    final.save(image_path, quality=rng.randint(68, 94))
    label_path.write_text("\n".join(labels), encoding="utf-8")
    return bool(labels)

# Validates that a probability-style setting is between 0 and 1.
def validate_ratio(name: str, value: float) -> None:
    if not 0 <= value < 1:
        raise SystemExit(f"--{name} must be at least 0 and less than 1.")

# Runs the full synthetic dataset generation workflow.
#
# Validates every flag, seeds the RNG so runs are reproducible, resolves
# the three directories, then loops generating images while printing
# progress and a final positive/negative count.
def main() -> None:
    args = parse_args()
    if args.num <= 0:
        raise SystemExit("--num must be greater than 0.")
    validate_ratio("val_ratio", args.val_ratio)
    validate_ratio("negative_ratio", args.negative_ratio)
    validate_ratio("motion_blur_ratio", args.motion_blur_ratio)
    validate_ratio("hard_negative_ratio", args.hard_negative_ratio)
    if args.shadow_strength < 0:
        raise SystemExit("--shadow_strength must be 0 or greater.")
    if args.noise_strength < 0:
        raise SystemExit("--noise_strength must be 0 or greater.")
    if args.min_roach_scale <= 0:
        raise SystemExit("--min_roach_scale must be greater than 0.")
    if args.max_roach_scale < args.min_roach_scale:
        raise SystemExit("--max_roach_scale must be greater than or equal to --min_roach_scale.")
    if args.cutout_brightness_min <= 0:
        raise SystemExit("--cutout_brightness_min must be greater than 0.")
    if args.cutout_brightness_max < args.cutout_brightness_min:
        raise SystemExit(
            "--cutout_brightness_max must be greater than or equal to --cutout_brightness_min."
        )
    if (args.output_width == 0) != (args.output_height == 0):
        raise SystemExit("--output_width and --output_height must be used together.")
    if args.output_width < 0 or args.output_height < 0:
        raise SystemExit("--output_width and --output_height must be 0 or greater.")

    rng = random.Random(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    config = RealismConfig(
        motion_blur_ratio=args.motion_blur_ratio,
        hard_negative_ratio=args.hard_negative_ratio,
        shadow_strength=args.shadow_strength,
        noise_strength=args.noise_strength,
        min_roach_scale=args.min_roach_scale,
        max_roach_scale=args.max_roach_scale,
        cutout_brightness_min=args.cutout_brightness_min,
        cutout_brightness_max=args.cutout_brightness_max,
        output_width=args.output_width,
        output_height=args.output_height,
    )

    background_dir = args.background_dir.resolve()
    cutout_dir = args.cutout_dir.resolve()
    out_dir = args.out_dir.resolve()

    if not background_dir.exists():
        raise SystemExit(f"Background directory does not exist: {background_dir}")
    if not cutout_dir.exists():
        raise SystemExit(f"Cockroach cutout directory does not exist: {cutout_dir}")

    background_paths = list_images(background_dir)
    cutout_paths = list_cutouts(cutout_dir)
    if not background_paths:
        raise SystemExit(f"No backgrounds found in {background_dir}.")
    if not cutout_paths:
        raise SystemExit(f"No cockroach PNG cutouts found in {cutout_dir}.")

    prepare_output_dirs(out_dir)
    if args.clean:
        clean_output_dirs(out_dir)

    print(f"Generating {args.num} synthetic image(s).")
    print(f"Using {len(background_paths)} background(s) and {len(cutout_paths)} cutout(s).")
    print(
        "Realism controls: "
        f"motion_blur_ratio={config.motion_blur_ratio}, "
        f"hard_negative_ratio={config.hard_negative_ratio}, "
        f"shadow_strength={config.shadow_strength}, "
        f"noise_strength={config.noise_strength}, "
        f"roach_scale={config.min_roach_scale}-{config.max_roach_scale}, "
        f"cutout_brightness={config.cutout_brightness_min}-{config.cutout_brightness_max}, "
        f"output_size={config.output_width or 'source'}x{config.output_height or 'source'}"
    )

    positive_count = 0
    for index in range(args.num):
        split = "val" if rng.random() < args.val_ratio else "train"
        if generate_one(
            index,
            split,
            background_paths,
            cutout_paths,
            out_dir,
            args.negative_ratio,
            rng,
            config,
        ):
            positive_count += 1

        if (index + 1) % 100 == 0 or index + 1 == args.num:
            print(f"Created {index + 1}/{args.num} images.")

    print(f"Finished dataset at {out_dir}.")
    print(f"Images with at least one cockroach: {positive_count}")
    print(f"Negative images: {args.num - positive_count}")


if __name__ == "__main__":
    main()
