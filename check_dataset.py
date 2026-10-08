from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt

ROOT = Path(r"D:\PotholeProject\merged_pothole_dataset")
SAMPLES = 12
random.seed(42)

image_files = []
for split in ("train", "val", "test"):
    split_dir = ROOT / "images" / split
    if split_dir.exists():
        image_files.extend(
            p for p in split_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
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
    ax.imshow(image)
    ax.set_title(image_path.name, fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.show()
