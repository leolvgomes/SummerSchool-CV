from pathlib import Path
from shutil import copy2
import subprocess
import sys

project = Path(__file__).resolve().parent
frames = project / "video_frames"
classes = project / "labelimg_classes.txt"
labels = project / "labelimg_labels"

python = Path(sys.executable)
labelimg = python.with_name("labelImg.exe")
canvas = Path(sys.prefix) / "Lib" / "site-packages" / "libs" / "canvas.py"

if not labelimg.is_file() or not canvas.is_file():
    raise FileNotFoundError(
        f"Run this script with the Day-2 interpreter that has LabelImg installed.\n"
        f"Current interpreter: {python}"
    )

if not frames.is_dir():
    raise FileNotFoundError(f"Frame folder missing: {frames}")

replacements = {
    "p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), self.pixmap.height())":
        "p.drawLine(int(self.prev_point.x()), 0, int(self.prev_point.x()), int(self.pixmap.height()))",
    "p.drawLine(0, self.prev_point.y(), self.pixmap.width(), self.prev_point.y())":
        "p.drawLine(0, int(self.prev_point.y()), int(self.pixmap.width()), int(self.prev_point.y()))",
    "p.drawRect(left_top.x(), left_top.y(), rect_width, rect_height)":
        "p.drawRect(int(left_top.x()), int(left_top.y()), int(rect_width), int(rect_height))",
}

original = canvas.read_text(encoding="utf-8")
updated = original

for old_line, new_line in replacements.items():
    if old_line in updated:
        updated = updated.replace(old_line, new_line)
    elif new_line not in updated:
        raise RuntimeError(f"Could not find the expected drawing line:\n{old_line}")

if updated != original:
    compile(updated, str(canvas), "exec")
    backup = canvas.with_name("canvas.py.bak")
    if not backup.exists():
        copy2(canvas, backup)

    temporary = canvas.with_name("canvas.py.tmp")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(canvas)
    print(f"LabelImg fixed. Backup: {backup}", flush=True)
else:
    print("The LabelImg drawing fix is already applied.", flush=True)

if not classes.exists():
    classes.write_text("car\n", encoding="ascii")

labels.mkdir(exist_ok=True)
subprocess.run(
    [str(labelimg), str(frames), str(classes), str(labels)],
    cwd=project,
    check=True,
)