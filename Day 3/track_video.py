import sys
from pathlib import Path

from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent
WEIGHTS = PROJECT / "best.pt"
VIDEO = PROJECT / "traffic_video.mp4"
CONF = 0.10
RUN_NAME = "confidence_010"

# Optional: pass the confidence on the command line, e.g. python track_video.py 0.50
if len(sys.argv) > 1:
    CONF = float(sys.argv[1])
    RUN_NAME = f"confidence_{round(CONF * 100):03d}"


def main():
    for path in (WEIGHTS, VIDEO):
        if not path.is_file():
            raise FileNotFoundError(f"Missing file: {path}")

    model = YOLO(str(WEIGHTS))
    results = model.track(
        source=str(VIDEO),
        tracker="bytetrack.yaml",
        device="cpu",
        conf=CONF,
        stream=True,
        save=True,
        project=str(PROJECT / "runs" / "tracking"),
        name=RUN_NAME,
    )

    # Remember the last side of the line for each car track.
    previous_side = {}
    counted_ids = set()

    for frame_number, result in enumerate(results, start=1):
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue

        line_x = result.orig_shape[1] / 2
        ids = boxes.id.int().cpu().tolist()
        classes = boxes.cls.int().cpu().tolist()
        centers_x = boxes.xywh[:, 0].cpu().tolist()

        for track_id, class_id, center_x in zip(ids, classes, centers_x):
            if result.names[class_id] != "car" or center_x == line_x:
                continue

            side = "left" if center_x < line_x else "right"
            if (
                previous_side.get(track_id) == "right"
                and side == "left"
                and track_id not in counted_ids
            ):
                counted_ids.add(track_id)
                print(f"Crossing at frame {frame_number}, car ID {track_id}")

            previous_side[track_id] = side

    print(f"Right-to-left car crossings: {len(counted_ids)}")
    print(f"Tracked video folder: {model.predictor.save_dir}")


if __name__ == "__main__":
    main()
