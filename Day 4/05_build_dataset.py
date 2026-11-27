"""Step 5 (Day 2): split the labeled frames into train / val folders.

Every 5th frame goes to validation, so validation covers the whole video
(start, middle and end) instead of only the last seconds.
"""
from pathlib import Path
import shutil

PROJECT = Path(__file__).resolve().parent
FRAMES = PROJECT / "video_frames"
LABELS = PROJECT / "labelimg_labels"
DATASET = PROJECT / "dataset"
VAL_EVERY = 5

images = sorted(FRAMES.glob("*.jpg"))
if not images:
    raise FileNotFoundError(f"Run 01_extract_frames.py first: {FRAMES}")

counts = {"train": 0, "val": 0}
for index, image in enumerate(images, start=1):
    label = LABELS / f"{image.stem}.txt"
    if not label.is_file():
        raise FileNotFoundError(f"Missing label for {image.name}. Label it in 04_label_img.py.")
    split = "val" if index % VAL_EVERY == 0 else "train"
    for kind, source in (("images", image), ("labels", label)):
        target = DATASET / split / kind
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target / source.name)
    counts[split] += 1

(DATASET / "classes.txt").write_text((PROJECT / "labelimg_classes.txt").read_text())
for cache in DATASET.glob("*/labels.cache"):
    cache.unlink()
print(f"train: {counts['train']} images | val: {counts['val']} images")
