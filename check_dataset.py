
from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt

ROOT = Path(r"D:\PotholeProject\merged_pothole_dataset")
SAMPLES = 12
random.seed(42)

# Collect images from all three splits
image_files = []
for split in ("train", "val", "test"):
    image_files.extend(
        p for p in (ROOT / "images" / split).iterdir()
        if p.is_file() and p.suffix.lower()
        in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

if not image_files:
    raise RuntimeError("No images found in the merged dataset.")

samples = random.sample(image_files, min(SAMPLES, len(image_files)))

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for ax, image_path in zip(axes, samples):
    image = cv2.imread(str(image_path))
    if image is None:
        ax.set_title(f"Cannot read: {image_path.name}")
        ax.axis("off")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    label_path = (
        ROOT / "labels" / image_path.parent.name
        / f"{image_path.stem}.txt"
    )

    if label_path.exists():
        for line in label_path.read_text(
            encoding="utf-8-sig"
        ).splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue

            _, xc, yc, bw, bh = map(float, parts)
            x1 = int((xc - bw / 2) * width)
            y1 = int((yc - bh / 2) * height)
            x2 = int((xc + bw / 2) * width)
            y2 = int((yc + bh / 2) * height)

            cv2.rectangle(
                image, (x1, y1), (x2, y2),
                (255, 0, 0), 2
            )

    ax.imshow(image)
    ax.set_title(image_path.name, fontsize=8)
    ax.axis("off")

for ax in axes[len(samples):]:
    ax.axis("off")

plt.tight_layout()
plt.show()

