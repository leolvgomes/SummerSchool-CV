from pathlib import Path
from ultralytics import YOLO

project = Path(__file__).resolve().parent
image_path = project / "street.png"
if not image_path.is_file():
    raise FileNotFoundError(f"Put street.png in this folder: {project}")

weights = project / "yolo12n.pt"
model = YOLO(str(weights) if weights.is_file() else "yolo12n.pt")
result = model.predict(str(image_path), device="cpu", conf=0.25)[0]

output_path = project / "street_detected.png"
result.save(filename=str(output_path))

for box in result.boxes:
    class_id = int(box.cls.item())
    confidence = box.conf.item()
    xyxy = [round(value) for value in box.xyxy[0].tolist()]
    print(f"{result.names[class_id]} | {confidence:.2f} | {xyxy}")

print(f"Saved: {output_path}")
