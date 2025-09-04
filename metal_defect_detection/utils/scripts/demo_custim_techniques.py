import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------- utils -----------
def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def show_pair(title_left, img_left, title_right, img_right, is_right_gray=False):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_left)
    plt.title(title_left)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    if is_right_gray:
        plt.imshow(img_right, cmap="gray")
    else:
        plt.imshow(img_right)
    plt.title(title_right)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# ----------- techniques -----------
def rotate_image(img, angle=10):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return rot

def adjust_contrast_brightness(img, alpha=1.2, beta=15):
    # alpha: contrast (1.0 giữ nguyên), beta: brightness (0 giữ nguyên)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def random_zoom_or_crop(img, out_size=None, rng=None, zoom_range=(0.8, 1.2)):
    """
    zoom<1.0 -> crop & resize; zoom>1.0 -> scale-up then center-crop
    out_size: (w,h) đầu ra; mặc định giữ nguyên kích thước ban đầu
    """
    if rng is None:
        rng = np.random.default_rng(123)
    h, w = img.shape[:2]
    if out_size is None:
        out_w, out_h = w, h
    else:
        out_w, out_h = out_size

    z = rng.uniform(*zoom_range)

    if z < 1.0:
        # crop ngẫu nhiên theo tỉ lệ z, rồi resize về out_size
        crop_w, crop_h = int(w*z), int(h*z)
        x0 = rng.integers(0, max(1, w - crop_w + 1))
        y0 = rng.integers(0, max(1, h - crop_h + 1))
        crop = img[y0:y0+crop_h, x0:x0+crop_w]
        return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    else:
        # phóng to rồi center-crop về kích thước gốc
        new_w, new_h = int(w*z), int(h*z)
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # center-crop
        x0 = (new_w - out_w) // 2
        y0 = (new_h - out_h) // 2
        return scaled[y0:y0+out_h, x0:x0+out_w]

def add_gaussian_noise(img, sigma=12.0, seed=123):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy

def grabcut_foreground(img, rect_ratio=0.8, iter_count=5):
    """
    Tách nền (foreground extraction) bằng GrabCut.
    Khởi tạo hình chữ nhật trung tâm chiếm rect_ratio của ảnh.
    Trả về ảnh nền trắng + foreground giữ nguyên (để dễ quan sát).
    """
    h, w = img.shape[:2]
    rw, rh = int(w*rect_ratio), int(h*rect_ratio)
    x = (w - rw) // 2
    y = (h - rh) // 2
    rect = (x, y, rw, rh)

    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT)

    # Xác định vùng foreground
    mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype("uint8")
    # Áp mask lên ảnh
    fg = img * mask2[:, :, np.newaxis]

    # Nền trắng để tương phản
    white_bg = np.ones_like(img, dtype=np.uint8) * 255
    out = white_bg.copy()
    out[mask2 == 1] = fg[mask2 == 1]
    return out

def canny_edges(img, blur_ksize=3, t1=100, t2=200):
    """
    Trích biên bằng Canny; chuyển sang grayscale + blur nhẹ để ổn định
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if blur_ksize and blur_ksize > 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(gray, t1, t2)
    return edges

# ----------- main -----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Đường dẫn ảnh đầu vào")
    parser.add_argument("--angle", type=float, default=10.0, help="Góc xoay (độ)")
    parser.add_argument("--alpha", type=float, default=1.2, help="Hệ số tương phản (alpha)")
    parser.add_argument("--beta", type=float, default=15.0, help="Độ sáng (beta)")
    parser.add_argument("--sigma_noise", type=float, default=12.0, help="Độ lệch chuẩn nhiễu Gaussian")
    parser.add_argument("--zoom_min", type=float, default=0.85, help="Zoom min cho random zoom/crop")
    parser.add_argument("--zoom_max", type=float, default=1.15, help="Zoom max cho random zoom/crop")
    args = parser.parse_args()

    # Đọc ảnh và resize về 224x224
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Không đọc được ảnh: {args.image}")
    img = cv2.resize(img, (224, 224))   # resize trước khi augment

    rgb = bgr2rgb(img)

    # 1) Xoay ảnh
    rot = rotate_image(img, angle=args.angle)
    show_pair("Original", rgb, f"Rotate ({args.angle}°)", bgr2rgb(rot))

    # 2) Thay đổi độ tương phản/độ sáng
    cb = adjust_contrast_brightness(img, alpha=args.alpha, beta=int(args.beta))
    show_pair("Original", rgb, f"Contrast/Brightness (α={args.alpha}, β={int(args.beta)})", bgr2rgb(cb))

    # 3) Phóng to hoặc cắt xén ngẫu nhiên (giữ kích thước 224x224)
    zc = random_zoom_or_crop(img, out_size=(224, 224),
                             rng=np.random.default_rng(2024),
                             zoom_range=(args.zoom_min, args.zoom_max))
    show_pair("Original", rgb, f"Random Zoom/Crop [{args.zoom_min}, {args.zoom_max}]", bgr2rgb(zc))

    # 4) Thêm nhiễu Gaussian
    noisy = add_gaussian_noise(img, sigma=args.sigma_noise, seed=999)
    show_pair("Original", rgb, f"Gaussian Noise (σ={args.sigma_noise})", bgr2rgb(noisy))

    # 5) Tách nền (GrabCut)
    fg = grabcut_foreground(img, rect_ratio=0.8, iter_count=5)
    show_pair("Original", rgb, "Foreground (GrabCut, bg=white)", bgr2rgb(fg))

    # 6) Trích biên (Canny)
    edges = canny_edges(img, blur_ksize=3, t1=100, t2=200)
    show_pair("Original", rgb, "Canny Edges", edges, is_right_gray=True)

if __name__ == "__main__":
    main()
