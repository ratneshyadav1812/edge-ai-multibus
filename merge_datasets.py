from pathlib import Path
from collections import Counter
import random
import shutil
import hashlib
import csv

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT = Path(r"D:\PotholeProject")
SOURCE_ROOT = PROJECT / "extracted"
OUTPUT = PROJECT / "merged_pothole_dataset"

# Dataset 1 has 3 classes; only class ID 2 is pothole.
# Datasets 2 and 3 have one class, ID 0.
DATASETS = [
    ("dataset1", 2),
    ("dataset2", 0),
    ("dataset3", 0),
]

SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REBUILD_OUTPUT = True


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    """Calculate exact file hash for duplicate detection."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_split_folder(dataset_dir, split):
    """Find train, valid/val, or test folder."""
    alternatives = {
        "train": ["train"],
        "val": ["valid", "val"],
        "test": ["test"],
    }

    for name in alternatives[split]:
        path = dataset_dir / name
        if path.is_dir():
            return path

    return None


def load_dataset(dataset_name, target_class):
    """Read images and keep only pothole boxes."""
    dataset_dir = SOURCE_ROOT / dataset_name
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_dir}"
        )

    records = []
    stats = Counter()

    for split in ["train", "val", "test"]:
        split_dir = find_split_folder(dataset_dir, split)

        if split_dir is None:
            print(f"WARNING: {dataset_name} has no {split} folder")
            continue

        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(
                f"Expected images and labels inside {split_dir}"
            )

        for image_path in images_dir.rglob("*"):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = labels_dir / (image_path.stem + ".txt")

            if not label_path.is_file():
                stats["missing_label_files"] += 1
                print(f"WARNING: No label file for {image_path}")
                continue

            kept_lines = []
            invalid_lines = 0

            for line in label_path.read_text(
                encoding="utf-8-sig"
            ).splitlines():
                line = line.strip()
                if not line:
                    continue

                parts = line.split()

                # YOLO detection labels have class + 4 coordinates.
                if len(parts) != 5:
                    invalid_lines += 1
                    continue

                try:
                    class_id = int(parts[0])
                    coords = [float(value) for value in parts[1:]]
                except ValueError:
                    invalid_lines += 1
                    continue

                x_center, y_center, width, height = coords

                # Reject malformed or out-of-range bounding boxes.
                if not all(0 <= value <= 1 for value in coords):
                    invalid_lines += 1
                    continue
                if width <= 0 or height <= 0:
                    invalid_lines += 1
                    continue
                if (x_center - width / 2 < -1e-6 or
                    x_center + width / 2 > 1.000001 or
                    y_center - height / 2 < -1e-6 or
                    y_center + height / 2 > 1.000001):
                    invalid_lines += 1
                    continue

                # Keep only pothole annotations and remap to class 0.
                if class_id == target_class:
                    kept_lines.append(
                        "0 " + " ".join(
                            f"{value:.8f}" for value in coords
                        )
                    )

            stats["invalid_label_lines"] += invalid_lines
            stats["images_read"] += 1
            stats["pothole_boxes"] += len(kept_lines)

            records.append({
                "dataset": dataset_name,
                "split": split,
                "image": image_path,
                "labels": kept_lines,
            })

    return records, stats
