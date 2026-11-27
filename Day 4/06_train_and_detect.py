"""Step 6 (Day 2): fine-tune YOLO12n on the labeled frames (class: person)."""
from pathlib import Path

import yaml
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent
DATASET = PROJECT / "dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def check_split(split, class_count):
    split_dir = DATASET / split
    nested = (split_dir / "images").is_dir()
    image_dir = split_dir / "images" if nested else split_dir
    label_dir = split_dir / "labels" if nested else split_dir

    images = sorted(
        path for path in image_dir.glob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No images found in {image_dir}")

    box_count = 0
    for image in images:
        label = label_dir / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Missing label for {image.name}: {label}")

        for line_number, line in enumerate(
            label.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5:
                raise ValueError(f"{label.name}, line {line_number}: expected 5 values.")
            class_id = int(fields[0])
            if not 0 <= class_id < class_count:
                raise ValueError(f"{label.name}: class ID {class_id} is missing from classes.txt.")
            box_count += 1

    if box_count == 0:
        raise ValueError(f"No labeled objects found in {split}.")

    print(f"{split}: {len(images)} images, {box_count} boxes")
    return image_dir


def main():
    classes_file = DATASET / "classes.txt"
    names = [
        line.strip()
        for line in classes_file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not names:
        raise ValueError("classes.txt is empty.")

    print("Classes:", dict(enumerate(names)))
    train_dir = check_split("train", len(names))
    val_dir = check_split("val", len(names))

    config = DATASET / "data.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "path": DATASET.as_posix(),
                "train": train_dir.relative_to(DATASET).as_posix(),
                "val": val_dir.relative_to(DATASET).as_posix(),
                "names": names,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    model = YOLO(str(PROJECT / "yolo12n.pt"))
    model.train(
        data=str(config),
        epochs=10,
        imgsz=640,
        batch=2,
        device="cpu",
        workers=0,
        project=str(PROJECT / "runs"),
        name="yolo12_training",
        exist_ok=True,
    )

    best_weights = model.trainer.save_dir / "weights" / "best.pt"
    print(f"\nBest weights: {best_weights}")

    detector = YOLO(str(best_weights))
    for result in detector.predict(
        source=str(val_dir),
        device="cpu",
        imgsz=640,
        conf=0.25,
        save=True,
        stream=True,
        project=str(PROJECT / "runs"),
        name="validation_detections",
        exist_ok=True,
    ):
        print(f"{Path(result.path).name}: {len(result.boxes)} detections")

    print(f"\nOpen the annotated images in: {detector.predictor.save_dir}")


if __name__ == "__main__":
    main()