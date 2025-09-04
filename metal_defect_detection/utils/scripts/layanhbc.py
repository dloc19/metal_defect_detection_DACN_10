import cv2
import matplotlib.pyplot as plt

# Đọc ảnh gốc
image = cv2.imread(r"C:\Users\dloc\Desktop\metal_defect_detection\metal_defect_detection\data_train\test\crease\img_06_430103100_01149.jpg")  # thay "sample.jpg" bằng đường dẫn ảnh của bạn
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # chuyển sang RGB để hiển thị

# Resize ảnh về 224x224
resized = cv2.resize(image, (224, 224))
resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

# Chuyển sang ảnh xám
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# Khử nhiễu bằng Gaussian Blur
denoised = cv2.GaussianBlur(gray, (5, 5), 0)

# -------------------------
# Biểu đồ 1: Ảnh gốc vs Resize
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Ảnh gốc")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(resized_rgb)
plt.title("Ảnh resize (224x224)")
plt.axis("off")
plt.tight_layout()
plt.show()

# -------------------------
# Biểu đồ 2: Ảnh gốc vs Ảnh xám
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Ảnh gốc")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(gray, cmap='gray')
plt.title("Ảnh xám")
plt.axis("off")
plt.tight_layout()
plt.show()

# -------------------------
# Biểu đồ 3: Ảnh gốc vs Ảnh khử nhiễu
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Ảnh gốc")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(denoised, cmap='gray')
plt.title("Ảnh khử nhiễu (Gaussian Blur)")
plt.axis("off")
plt.tight_layout()
plt.show()
