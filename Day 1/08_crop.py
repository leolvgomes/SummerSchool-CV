import cv2
import matplotlib.pyplot as plt

image_bgr = cv2.imread("car.png")
if image_bgr is None:
    raise ValueError("Could not read car.png")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

height, width = image_rgb.shape[:2]
x1, x2 = int(0.30 * width), int(0.88 * width)
y1, y2 = int(0.35 * height), int(0.83 * height)
crop = image_rgb[y1:y2, x1:x2].copy()
print("Crop shape:", crop.shape)

plt.imshow(crop)
plt.title("Car region")
plt.axis("off")
plt.show()
