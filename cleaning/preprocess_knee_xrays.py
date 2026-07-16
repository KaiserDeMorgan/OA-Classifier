#!/usr/bin/env python3
"""
preprocess_knee_xrays.py

Preprocessing pipeline for the raw "Digital Knee X-ray Images" dataset
(Gornale & Patravali, Mendeley Data).

ASSUMPTIONS ABOUT FOLDER STRUCTURE (adjust if yours differs):
This dataset is commonly distributed as folder-per-class, sometimes with
two separate expert gradings, e.g.:

    root/
      MedicalExpert-I/
        0/  *.png or *.jpg
        1/
        2/
        3/
        4/
      MedicalExpert-II/
        0/
        1/
        2/
        3/
        4/

If your download instead has a flat folder of images + a CSV mapping
filename -> label, use --labels-csv instead of relying on folder names
(see --mode flag below).

WHAT THIS PIPELINE DOES, PER IMAGE:
  1. Load image (grayscale)
  2. Denoise (mild median filter - X-rays often have salt-and-pepper noise
     from scanning/digitization)
  3. CLAHE contrast enhancement (Contrast Limited Adaptive Histogram
     Equalization) - standard for X-ray preprocessing, brings out
     joint space / osteophyte edges without blowing out contrast
  4. Crop to the region of interest (ROI) - simple intensity-based
     approach: finds the largest non-background bounding box. This is a
     heuristic, not a trained detector - inspect results and adjust
     threshold if it's cropping too aggressively or not enough.
  5. Resize to a consistent size (default 224x224)
  6. Normalize pixel values to [0, 1]
  7. Save as .png into an organized output folder, plus write a manifest
     CSV: filepath, label, source_expert (if applicable), orig_size

USAGE:
    python preprocess_knee_xrays.py --input /path/to/raw/dataset --output /path/to/processed --mode folders

    # if you have a flat folder + csv instead:
    python preprocess_knee_xrays.py --input /path/to/raw/images --output /path/to/processed --mode csv --labels-csv labels.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    print("Missing dependency. Install it with:")
    print("    pip install opencv-python")
    sys.exit(1)

from PIL import Image

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TARGET_SIZE = 224  # default output resolution (change with --size)


def load_grayscale(path: Path):
    """Load an image as grayscale numpy array. Falls back to PIL if OpenCV can't read it."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        # some files may be malformed for OpenCV; try PIL as a fallback
        try:
            pil_img = Image.open(path).convert("L")
            img = np.array(pil_img)
        except Exception:
            return None
    return img


def denoise(img):
    """Mild median filter to reduce salt-and-pepper / scan noise without blurring edges too much."""
    return cv2.medianBlur(img, 3)


def apply_clahe(img):
    """Contrast Limited Adaptive Histogram Equalization - standard X-ray contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def crop_to_roi(img, bg_threshold=10, min_area_fraction=0.1):
    """
    Heuristic crop: finds the bounding box of non-background pixels
    (pixels brighter than bg_threshold) and crops to it, with a small margin.

    This assumes the knee X-ray is the bright subject against a dark
    background - typical for radiographs, but INSPECT YOUR RESULTS,
    since some images may have borders, text/annotations burned in, or
    scanner artifacts that confuse this heuristic.
    """
    h, w = img.shape
    mask = img > bg_threshold

    if mask.sum() < (h * w * min_area_fraction):
        # too little "foreground" found - probably a very dark or unusual
        # image; skip cropping and return original rather than guess wrong
        return img

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # small margin so we don't crop right up against the joint edge
    margin_y = int(0.02 * h)
    margin_x = int(0.02 * w)
    y0 = max(0, y0 - margin_y)
    x0 = max(0, x0 - margin_x)
    y1 = min(h, y1 + margin_y)
    x1 = min(w, x1 + margin_x)

    return img[y0:y1, x0:x1]


def detect_bilateral(img, valley_threshold=0.15, min_aspect_ratio=1.15):
    """
    Heuristically detects whether an image contains TWO knees side by side
    (bilateral) vs just ONE knee.

    Approach:
      1. Quick aspect-ratio filter - bilateral images are typically
         noticeably wider than tall. Single-knee crops are usually closer
         to square or taller-than-wide. Images that don't clear this bar
         are assumed single-knee without further checks.
      2. For images that pass the aspect check, look at the column-wise
         "foreground density" profile (how much bright/tissue content is
         in each vertical column). A bilateral image typically has two
         humps (one per knee) separated by a valley (background gap)
         somewhere in the middle third of the image. A single knee image
         usually has one broad hump with no such central valley.

    This is a heuristic, not a trained classifier - it WILL be wrong on
    some images (unusual framing, cropped joints, text/markers in the
    frame, etc). Always spot check the auto-detected split vs. keep
    decisions in the manifest before trusting it fully.

    Returns True if the image looks bilateral (should be split), False otherwise.
    """
    h, w = img.shape

    if (w / h) < min_aspect_ratio:
        return False

    # foreground = pixels brighter than a simple background threshold
    fg_mask = img > 10
    col_density = fg_mask.sum(axis=0).astype(np.float32)
    if col_density.max() == 0:
        return False
    col_density /= col_density.max()

    # look for a valley (low-density region) within the middle third
    mid_start = w // 3
    mid_end = 2 * w // 3
    middle_region = col_density[mid_start:mid_end]

    if len(middle_region) == 0:
        return False

    min_in_middle = middle_region.min()
    # peaks on either side of that valley should be meaningfully higher
    left_peak = col_density[:mid_start].max() if mid_start > 0 else 0
    right_peak = col_density[mid_end:].max() if mid_end < w else 0

    has_valley = min_in_middle < valley_threshold
    has_two_peaks = left_peak > 0.5 and right_peak > 0.5

    return bool(has_valley and has_two_peaks)


def split_left_right(img):
    """
    Splits a bilateral knee X-ray (both knees in one frame) down the
    vertical midline into two single-knee halves.

    NOTE: this labels halves by their POSITION in the image
    ("left_half" = left side of the frame, "right_half" = right side),
    NOT by confirmed anatomical laterality - that would require reading
    DICOM Laterality metadata or radiologist-convention knowledge, which
    this heuristic does not have. If you need true anatomical L/R, verify
    against source metadata before trusting these labels.

    Returns (left_half, right_half) as two separate arrays.
    """
    h, w = img.shape
    mid = w // 2
    left_half = img[:, :mid]
    right_half = img[:, mid:]
    return left_half, right_half


def resize(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def normalize(img):
    """Scale to [0, 1] float32."""
    return img.astype(np.float32) / 255.0


def process_one_image(path: Path, size: int, do_crop: bool, split_mode: str):
    """
    Runs the full pipeline on one image.
    split_mode: "off" (never split), "force" (always split), or
                "auto" (detect per-image whether it's bilateral)

    Returns a list of (tag, array) tuples: one entry normally, two if
    split happened (tagged 'left_half' / 'right_half'). Returns None on failure.
    Also returns whether a split was performed, for logging purposes.
    """
    img = load_grayscale(path)
    if img is None:
        return None, False

    img = denoise(img)
    img = apply_clahe(img)

    do_split = False
    if split_mode == "force":
        do_split = True
    elif split_mode == "auto":
        do_split = detect_bilateral(img)
    # split_mode == "off" -> do_split stays False

    if do_split:
        left_half, right_half = split_left_right(img)
        results = []
        for tag, half_img in [("left_half", left_half), ("right_half", right_half)]:
            h = crop_to_roi(half_img) if do_crop else half_img
            h = resize(h, size)
            h_norm = normalize(h)
            results.append((tag, (h_norm * 255).astype(np.uint8)))
        return results, True

    if do_crop:
        img = crop_to_roi(img)

    img = resize(img, size)
    img_norm = normalize(img)
    img_uint8 = (img_norm * 255).astype(np.uint8)
    return [(None, img_uint8)], False


def collect_folder_mode(input_dir: Path):
    """
    Walks a folder-per-class structure. Handles folder names that are
    either exactly "0".."4", OR start with a digit 0-4 followed by a
    label word, e.g. "0Normal", "1Doubtful", "2Mild", "3Moderate",
    "4Severe" - the common naming in the Gornale/Patravali Digital Knee
    X-ray Images dataset.

    Also handles an optional expert-name folder above the label folder
    (e.g. MedicalExpert-I/0Normal/...).

    Returns list of (filepath, label, source_expert_or_None)
    """
    import re
    label_pattern = re.compile(r"^([0-4])")

    items = []

    for path in input_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        folder_name = path.parent.name
        match = label_pattern.match(folder_name)
        if not match:
            continue
        label = int(match.group(1))

        # if there's an expert-name folder above the label folder, capture it
        expert = path.parent.parent.name if path.parent.parent != input_dir else None
        items.append((path, label, expert))

    return items


def collect_csv_mode(input_dir: Path, labels_csv: Path):
    """
    Reads a CSV with columns: filename,label
    Matches filenames against files found under input_dir (recursively).
    """
    label_map = {}
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("filename") or row.get("file") or row.get("image")
            label = row.get("label") or row.get("klg") or row.get("kl_grade")
            if fname is None or label is None:
                continue
            label_map[fname.strip()] = int(label)

    items = []
    for path in input_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        if path.name in label_map:
            items.append((path, label_map[path.name], None))

    missing = set(label_map.keys()) - {p.name for p, _, _ in items}
    if missing:
        print(f"Warning: {len(missing)} filenames in the CSV were not found under {input_dir}")

    return items


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw knee X-ray images: denoise, contrast-enhance, crop, resize, normalize.")
    parser.add_argument("--input", required=True, help="Root folder of the raw downloaded dataset")
    parser.add_argument("--output", required=True, help="Folder to write processed images + manifest.csv into")
    parser.add_argument("--mode", choices=["folders", "csv"], default="folders",
                         help="'folders' = label inferred from parent folder name (0-4). 'csv' = label from a filename->label CSV")
    parser.add_argument("--labels-csv", help="Required if --mode csv. CSV with columns: filename,label")
    parser.add_argument("--size", type=int, default=TARGET_SIZE, help="Output image size (square), default 224")
    parser.add_argument("--no-crop", action="store_true", help="Skip the ROI-cropping heuristic and just resize the full image")
    parser.add_argument("--split-lr", choices=["off", "force", "auto"], default="off",
                         help="'off' = never split (default). 'force' = split every image in half. "
                              "'auto' = detect per-image whether it looks bilateral (two knees) and "
                              "only split those - use this if your dataset is a MIX of single and "
                              "bilateral images. Labels halves by frame position, not confirmed "
                              "anatomical laterality.")
    parser.add_argument("--dry-run", action="store_true", help="Process and log without writing output images")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Input folder does not exist: {input_dir}")
        sys.exit(1)

    if args.mode == "csv" and not args.labels_csv:
        print("--labels-csv is required when --mode csv is used.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "folders":
        items = collect_folder_mode(input_dir)
    else:
        items = collect_csv_mode(input_dir, Path(args.labels_csv))

    if not items:
        print("No labeled images found. Check --mode and your folder/CSV structure, "
              "then re-run - nothing was processed.")
        sys.exit(1)

    print(f"Found {len(items)} labeled images to process.\n")

    manifest_path = output_dir / "manifest.csv"
    manifest_file = open(manifest_path, "w", newline="")
    manifest_writer = csv.writer(manifest_file)
    manifest_writer.writerow(["filepath", "label", "source_expert", "orig_height", "orig_width", "was_split"])

    per_class_counts = {i: 0 for i in range(5)}
    failed = 0
    split_count = 0

    for i, (path, label, expert) in enumerate(items, 1):
        orig_img = load_grayscale(path)
        if orig_img is None:
            failed += 1
            continue
        orig_h, orig_w = orig_img.shape

        results, was_split = process_one_image(path, args.size, do_crop=not args.no_crop, split_mode=args.split_lr)
        if results is None:
            failed += 1
            continue

        class_dir = output_dir / str(label)
        class_dir.mkdir(exist_ok=True)

        for tag, processed in results:
            suffix = f"_{tag}" if tag else ""
            out_name = f"{expert + '_' if expert else ''}{path.stem}{suffix}.png"
            out_path = class_dir / out_name

            if not args.dry_run:
                cv2.imwrite(str(out_path), processed)

            manifest_writer.writerow([str(out_path), label, expert or "", orig_h, orig_w, was_split])
            per_class_counts[label] += 1

        if was_split:
            split_count += 1

        if i % 100 == 0 or i == len(items):
            print(f"[{i}/{len(items)}] processed", end="\r")

    manifest_file.close()

    print("\n\n--- Done ---")
    print(f"Processed: {len(items) - failed}")
    print(f"Failed to read: {failed}")
    if args.split_lr == "auto":
        print(f"Auto-detected as bilateral and split: {split_count} / {len(items)}")
    print("Class distribution:")
    for cls, count in per_class_counts.items():
        print(f"  Grade {cls}: {count}")
    print(f"\nManifest written to: {manifest_path}")
    if args.dry_run:
        print("This was a DRY RUN - no output images were written.")


if __name__ == "__main__":
    main()