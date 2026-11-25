import cv2
import matplotlib.pyplot as plt

image_bgr = cv2.imread("car.png")
if image_bgr is None:
    raise ValueError("Could not read car.png")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

blur5 = cv2.GaussianBlur(image_rgb, (5, 5), 0)
blur21 = cv2.GaussianBlur(image_rgb, (21, 21), 0)

images = [image_rgb, blur5, blur21]
titles = ["Original", "5 x 5", "21 x 21"]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, img, title in zip(axes, images, titles):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
