import os

folders = [
    "metal_defect_detection/data/train",
    "metal_defect_detection/data/val",
    "metal_defect_detection/data/test",
    "metal_defect_detection/dataset",
    "metal_defect_detection/models",
    "metal_defect_detection/train",
    "metal_defect_detection/evaluate",
    "metal_defect_detection/gui_app",
    "metal_defect_detection/weights",
    "metal_defect_detection/utils",
]

files = {
    "metal_defect_detection/README.md": "# Metal Surface Defect Detection\n",
    "metal_defect_detection/requirements.txt": "\n".join([
        "tensorflow", "opencv-python", "numpy", "pandas", 
        "matplotlib", "scikit-learn", "albumentations", "pyqt5"
    ]),
    "metal_defect_detection/config.yaml": "batch_size: 32\nepochs: 20\nimage_size: [224, 224]\n"
}

# Tạo thư mục
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Tạo file
for path, content in files.items():
    with open(path, "w") as f:
        f.write(content)

print("✅ Đã tạo xong cấu trúc dự án!")
