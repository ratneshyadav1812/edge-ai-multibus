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
