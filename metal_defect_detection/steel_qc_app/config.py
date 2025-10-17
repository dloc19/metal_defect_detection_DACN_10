# ===================== CONFIG =====================
# lưu các tham số cấu hình (IMG_SIZE, SAVE_DIR, ALPHA_THR,...)
CKPT_PATH = "metal_defect_detection/weights/mobilenetv2_best.pth"  # đường dẫn model đã huấn luyện
MAP_PATH  = "metal_defect_detection/weights/idx_to_class.json" # đường dẫn file mapping class
IMG_SIZE  = 224
SAVE_DIR  = "./qc_outputs"

# ONNX Runtime Configuration
ONNX_PATH = "metal_defect_detection/weights/mobilenetv2_best.onnx"  # đường dẫn ONNX model
USE_ONNX = True  # True để dùng ONNX Runtime (nhanh hơn), False để dùng PyTorch
ONNX_USE_GPU = True  # True để dùng GPU với ONNX Runtime

SCENE_DELTA_THR = 6.0

# CAM/maskcd
PCT_START = 85
PCT_MIN   = 74
PCT_STEP  = 3
MIN_AREA  = 80

# Ngưỡng quyết định lỗi theo xác suất
ALPHA_THR = 0.6

# Grad-CAM++ target layers cho MobileNetV2
CAM_TOP_LAYER = "features.18"
CAM_AUX_LAYER = "features.17"

# Device
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Chuẩn hóa ảnh
from torchvision import transforms
MEAN, STD = [0.485,0.456,0.406], [0.229,0.224,0.225]
TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
