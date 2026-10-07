
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

# Set True to remove the old merged output and rebuild it.
# This deletes ONLY merged_pothole_dataset, not the source datasets.
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

            # Keep images even if no pothole box remains:
            # these can serve as negative examples.
            records.append({
                "dataset": dataset_name,
                "split": split,
                "image": image_path,
                "labels": kept_lines,
            })

    print(
        f"{dataset_name}: {len(records)} images read, "
        f"{stats['pothole_boxes']} pothole boxes, "
        f"{stats['missing_label_files']} missing label files, "
        f"{stats['invalid_label_lines']} invalid label lines"
    )

    return records, stats


# ============================================================
# MAIN
# ============================================================

def main():
    random.seed(SEED)

    if REBUILD_OUTPUT and OUTPUT.exists():
        print(f"Removing previous merged output: {OUTPUT}")
        shutil.rmtree(OUTPUT)

    # Read the three datasets and filter their labels.
    all_records = []
    source_counts = Counter()

    for dataset_name, target_class in DATASETS:
        records, _ = load_dataset(dataset_name, target_class)
        all_records.extend(records)
        source_counts[dataset_name] += len(records)

    if not all_records:
        raise RuntimeError(
            "No images found. Check your extracted folder structure."
        )

    # Remove exact duplicate images across all sources and splits.
    # The first copy found is retained.
    unique_records = []
    seen_hashes = set()
    duplicates = 0

    print("\nChecking exact duplicate images...")

    for record in all_records:
        digest = sha256_file(record["image"])

        if digest in seen_hashes:
            duplicates += 1
            continue

        seen_hashes.add(digest)
        record["hash"] = digest
        unique_records.append(record)

    print(f"Images before deduplication: {len(all_records)}")
    print(f"Exact duplicates skipped:    {duplicates}")
    print(f"Unique images retained:      {len(unique_records)}")

    # Randomize before assigning new common splits.
    random.shuffle(unique_records)

    total = len(unique_records)
    n_train = int(total * 0.80)
    n_val = int(total * 0.10)

    for index, record in enumerate(unique_records):
        if index < n_train:
            record["new_split"] = "train"
        elif index < n_train + n_val:
            record["new_split"] = "val"
        else:
            record["new_split"] = "test"

    # Create output directories.
    for split in ["train", "val", "test"]:
        (OUTPUT / "images" / split).mkdir(
            parents=True, exist_ok=True
        )
        (OUTPUT / "labels" / split).mkdir(
            parents=True, exist_ok=True
        )

    # Copy files with unique names.
    report_rows = []
    split_counts = Counter()
    box_counts = Counter()
    negative_counts = Counter()

    for index, record in enumerate(unique_records, start=1):
        split = record["new_split"]
        dataset = record["dataset"]
        source_image = record["image"]

        # Index guarantees unique filenames even when source names clash.
        new_stem = f"{dataset}_{index:07d}"

        destination_image = (
            OUTPUT / "images" / split /
            (new_stem + source_image.suffix.lower())
        )
        destination_label = (
            OUTPUT / "labels" / split / (new_stem + ".txt")
        )

        shutil.copy2(source_image, destination_image)

        # An empty label file represents an image with no pothole boxes.
        destination_label.write_text(
            "\n".join(record["labels"]) +
            ("\n" if record["labels"] else ""),
            encoding="utf-8"
        )

        split_counts[split] += 1
        box_counts[split] += len(record["labels"])

        if not record["labels"]:
            negative_counts[split] += 1

        report_rows.append({
            "new_split": split,
            "source_dataset": dataset,
            "original_split": record["split"],
            "original_image": str(source_image),
            "new_image": str(destination_image),
            "new_label": str(destination_label),
            "pothole_boxes": len(record["labels"]),
            "image_sha256": record["hash"],
        })

    # Create YOLO dataset configuration.
    yaml_path = OUTPUT / "data.yaml"
    yaml_path.write_text(
        f"path: {OUTPUT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: pothole\n",
        encoding="utf-8"
    )

    # Save a CSV audit report.
    report_path = OUTPUT / "merge_report.csv"
    with report_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "new_split",
                "source_dataset",
                "original_split",
                "original_image",
                "new_image",
                "new_label",
                "pothole_boxes",
                "image_sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print("\n========== MERGE COMPLETE ==========")
    print(f"Output folder: {OUTPUT}")
    print(f"Total unique images: {len(unique_records)}")
    print(f"Exact duplicates skipped: {duplicates}")

    for split in ["train", "val", "test"]:
        print(
            f"{split}: {split_counts[split]} images, "
            f"{box_counts[split]} pothole boxes, "
            f"{negative_counts[split]} images without pothole boxes"
        )

    print(f"\nYOLO config: {yaml_path}")
    print(f"Merge report: {report_path}")
    print("\nRDD2022 was not modified.")


if __name__ == "__main__":
    main()
