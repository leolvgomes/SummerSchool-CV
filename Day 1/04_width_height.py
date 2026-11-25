import cv2
import matplotlib.pyplot as plt

image_bgr = cv2.imread("car.png")
if image_bgr is None:
    raise ValueError("Could not read car.png")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

height, width, channels = image_rgb.shape
print("Height:", height)
print("Width:", width)
print("Channels:", channels)
print("Data type:", image_rgb.dtype)

x, y = 100, 50
pixel = image_rgb[y, x]
print("RGB pixel:", pixel)
