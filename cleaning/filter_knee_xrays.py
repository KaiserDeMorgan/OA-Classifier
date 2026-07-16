#!/usr/bin/env python3
"""
filter_knee_xrays_batch.py

Use this version now that your OAIBaselineImages download is COMPLETE.
Unlike the earlier "live watch" version, this does a single pass over
every file already on disk - no waiting/polling loop, since nothing new
is being downloaded anymore.

WHAT IT DOES TO YOUR FOLDERS:
  - Source folder (--source): files get READ, then either MOVED out (if
    knee X-ray) or DELETED (if not). By the end, your source folder should
    be empty (or contain only unreadable/skipped files) - it is NOT left
    untouched. This is what reclaims your disk space.
  - Destination folder (--keep): knee X-ray files get moved HERE. This
    folder grows as the script runs.

USAGE:
    python filter_knee_xrays_batch.py --source /path/to/oai_download --keep /path/to/knee_xrays_only --dry-run
    (check filter_log.csv, confirm it looks right, then run again without --dry-run)

    python filter_knee_xrays_batch.py --source /path/to/oai_download --keep /path/to/knee_xrays_only
"""

import argparse
import os
import shutil
import time
import csv
import sys
from pathlib import Path

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
except ImportError:
    print("Missing dependency. Install it with:")
    print("    pip install pydicom")
    sys.exit(1)

# Modalities that correspond to X-ray (radiograph) images.
# CR = Computed Radiography, DX = Digital Radiography
XRAY_MODALITIES = {"CR", "DX"}

# Keywords we look for in BodyPartExamined / SeriesDescription to confirm "knee"
KNEE_KEYWORDS = ["KNEE"]

# Files to skip outright (not DICOMs, no point trying to parse them)
SKIP_EXT = {".txt", ".csv", ".json", ".xml", ".md5", ".log"}


def find_candidate_files(source_dir: Path):
    """Walk the whole source tree and yield every file that isn't an obvious sidecar file."""
    for root, _, files in os.walk(source_dir):
        for fname in files:
            if Path(fname).suffix.lower() in SKIP_EXT:
                continue
            yield Path(root) / fname


def classify_dicom(path: Path):
    """
    Returns (decision, modality, body_part, series_desc)
    decision is one of: "knee_xray", "not_knee", "unreadable"
    """
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except (InvalidDicomError, Exception):
        return "unreadable", "", "", ""

    modality = str(getattr(ds, "Modality", "")).upper().strip()
    body_part = str(getattr(ds, "BodyPartExamined", "")).upper()
    series_desc = str(getattr(ds, "SeriesDescription", "")).upper()

    if modality not in XRAY_MODALITIES:
        return "not_knee", modality, body_part, series_desc

    text_fields = f"{body_part} {series_desc}"
    if any(kw in text_fields for kw in KNEE_KEYWORDS):
        return "knee_xray", modality, body_part, series_desc

    return "not_knee", modality, body_part, series_desc


def human_size(num_bytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PB"


def main():
    parser = argparse.ArgumentParser(description="One-pass filter of a completed OAI DICOM download to knee radiographs only.")
    parser.add_argument("--source", required=True, help="Folder containing the completed download (files will be moved/deleted from here)")
    parser.add_argument("--keep", required=True, help="Folder to move kept knee radiograph files into")
    parser.add_argument("--dry-run", action="store_true", help="Log decisions without actually moving or deleting anything")
    parser.add_argument("--log-csv", default="filter_log.csv", help="Path to write the decision log")
    args = parser.parse_args()

    source_dir = Path(args.source)
    keep_dir = Path(args.keep)

    if not source_dir.exists():
        print(f"Source folder does not exist: {source_dir}")
        sys.exit(1)

    keep_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_csv)
    log_exists = log_path.exists()
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(["timestamp", "file", "decision", "modality", "body_part", "series_description"])

    print(f"Scanning: {source_dir}")
    print(f"Keeping knee radiographs in: {keep_dir}")
    if args.dry_run:
        print("*** DRY RUN - nothing will actually be moved or deleted ***")
    print()

    kept_count = 0
    deleted_count = 0
    unreadable_count = 0
    bytes_freed = 0
    bytes_kept = 0
    total_scanned = 0

    all_files = list(find_candidate_files(source_dir))
    total_files = len(all_files)
    print(f"Found {total_files} candidate files to inspect.\n")

    start_time = time.time()

    for i, fpath in enumerate(all_files, 1):
        if not fpath.exists():
            continue

        total_scanned += 1
        decision, modality, body_part, series_desc = classify_dicom(fpath)

        if decision == "unreadable":
            unreadable_count += 1
            log_writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), str(fpath), "unreadable_skipped", "", "", ""])
        elif decision == "knee_xray":
            size = fpath.stat().st_size
            dest = keep_dir / fpath.name
            # avoid silent overwrite if a filename collision happens across subfolders
            if dest.exists():
                dest = keep_dir / f"{fpath.parent.name}_{fpath.name}"
            if not args.dry_run:
                shutil.move(str(fpath), str(dest))
            kept_count += 1
            bytes_kept += size
            log_writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), str(fpath), "knee_xray", modality, body_part, series_desc])
        else:  # not_knee
            size = fpath.stat().st_size
            if not args.dry_run:
                fpath.unlink()
            deleted_count += 1
            bytes_freed += size
            log_writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), str(fpath), "not_knee", modality, body_part, series_desc])

        if i % 200 == 0 or i == total_files:
            elapsed = time.time() - start_time
            print(f"[{i}/{total_files}] kept={kept_count}  deleted={deleted_count}  "
                  f"unreadable={unreadable_count}  freed={human_size(bytes_freed)}  "
                  f"elapsed={elapsed:.0f}s", end="\r")

        log_file.flush()

    log_file.close()

    print("\n\n--- Done ---")
    print(f"Scanned:      {total_scanned}")
    print(f"Kept (knee):  {kept_count}  ({human_size(bytes_kept)}) -> {keep_dir}")
    print(f"Deleted:      {deleted_count}  ({human_size(bytes_freed)} freed)")
    print(f"Unreadable:   {unreadable_count}  (left untouched in source folder)")
    print(f"Log written:  {log_path}")

    if args.dry_run:
        print("\nThis was a DRY RUN. Review filter_log.csv, then re-run without --dry-run to actually apply changes.")


if __name__ == "__main__":
    main()