"""Step 3 (Day 2): detect objects in one frame with the pretrained YOLO12n."""
from pathlib import Path
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent
IMAGE = PROJECT / "video_frames" / "frame_004.jpg"
OUTPUT = PROJECT / "results" / "detect_one.jpg"

if not IMAGE.is_file():
    raise FileNotFoundError(f"Run 01_extract_frames.py first. Missing: {IMAGE}")

model = YOLO(str(PROJECT / "yolo12n.pt"))
result = model.predict(str(IMAGE), device="cpu", conf=0.25)[0]
OUTPUT.parent.mkdir(exist_ok=True)
result.save(filename=str(OUTPUT))

for box in result.boxes:
    class_id = int(box.cls.item())
    xyxy = [round(v) for v in box.xyxy[0].tolist()]
    print(f"{result.names[class_id]} | {box.conf.item():.2f} | {xyxy}")
print(f"Saved: {OUTPUT}")
