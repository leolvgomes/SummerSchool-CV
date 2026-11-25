import cv2
import matplotlib.pyplot as plt

image_bgr = cv2.imread("car.png")
if image_bgr is None:
    raise ValueError("Could not read car.png")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

height, width = image_rgb.shape[:2]
new_width = 640
new_height = round(height * new_width / width)
resized = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)

print("original:", image_rgb.shape)
print("Resized:", resized.shape)

plt.imshow(resized)
plt.title("Resized image")
plt.axis("off")
plt.show()
