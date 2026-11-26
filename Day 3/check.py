from pathlib import Path
import ultralytics

project = Path(__file__).resolve().parent
weights = project / "best.pt"

print(f"Checkpoint: {weights}")
print(f"Checkpoint exists: {weights.is_file()}")
print(f"YOLO code: {ultralytics.__file__}")
