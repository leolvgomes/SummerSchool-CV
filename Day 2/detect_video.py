from pathlib import Path
from ultralytics import YOLO

project = Path(__file__).resolve().parent
weights = project / "runs/yolo12_training/weights/best.pt"
video = project / "traffic_video.mp4"

for path in (weights, video):
    if not path.is_file():
        raise FileNotFoundError(path)

model = YOLO(str(weights))
results = model.predict(
    source=str(video), device="cpu", imgsz=320, conf=0.25,
    save=True, stream=True, project=str(project / "runs"),
    name="video_detections",
)
for _ in results:
    pass

print(f"Saved video folder: {model.predictor.save_dir}")
