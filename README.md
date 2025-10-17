# Metal Surface Defect Detection

Ứng dụng kiểm tra lỗi bề mặt thép bằng MobileNetV2 + Grad-CAM++ (GUI PyQt5) và một module C++ demo xử lý ảnh với OpenCV.

## Dữ liệu / Trọng số
- Dataset tham khảo: `https://drive.google.com/file/d/1AXKNkXFU2Fc92qyEcyIX87raCWzcJSp-/view?usp=drive_link`
- Trọng số model và mapping lớp có sẵn:
  - `metal_defect_detection/weights/mobilenetv2_best.pth`
  - `metal_defect_detection/weights/idx_to_class.json`

## Chạy GUI suy luận (Windows)
1) Cài đặt Python 3.10/3.11 (khuyến nghị) và pip.

2) Cài đặt phụ thuộc:
```
pip install -r requirements.txt
```

3) Kiểm tra cấu hình tại `metal_defect_detection/steel_qc_app/config.py`:
- `CKPT_PATH = "metal_defect_detection/weights/mobilenetv2_best.pth"`
- `MAP_PATH  = "metal_defect_detection/weights/idx_to_class.json"`
- Điều chỉnh `ALPHA_THR`, `SAVE_DIR` nếu cần.

4) Chạy ứng dụng:
```
python metal_defect_detection/steel_qc_app/main_gui.py
```

5) Sử dụng:
- Chọn Camera/Video/Ảnh, nhấn Start để stream, hoặc mở Ảnh để suy luận 1 lần.
- Kết quả hiển thị khung phải, file lưu trong `SAVE_DIR`.

### Khắc phục sự cố
- Nếu không mở được camera: đổi backend trong `utils.list_cameras` hoặc dùng file video để kiểm tra.
- Nếu GPU không khả dụng, app tự chạy CPU (`DEVICE = "cpu"`).
- Lỗi cài đặt: đảm bảo đã gỡ `tensorflow` nếu không dùng, vì dự án inference dùng PyTorch.

## Build & chạy module C++ (OpenCV)
Yêu cầu: CMake ≥ 3.16, OpenCV C++ (đặt biến `OpenCV_DIR` tới thư mục build có `OpenCVConfig.cmake`).

```
cd cpp
mkdir build && cd build
cmake .. -DOpenCV_DIR="C:/opencv/build/x64/vc16/lib"
cmake --build . --config Release
./Release/mdproc.exe
```

Mặc định đọc ảnh từ `../data/images` (sửa trong `cpp/src/main.cpp`).

## Augment dữ liệu cân bằng theo lớp
Script: `metal_defect_detection/utils/scripts/augment_class_equalize.py`

Ví dụ:
```
python metal_defect_detection/utils/scripts/augment_class_equalize.py \
  --data metal_defect_detection/dataset \
  --target 1200 --ext jpg
```

## Gợi ý cải thiện
- Độ chính xác:
  - Tái huấn luyện với augment cân bằng theo lớp, mixup/cutmix (PyTorch), và label smoothing.
  - Fine-tune MobileNetV2 với scheduler cosine + warmup, early stopping.
  - Thử backbone khác (EfficientNet-B0/B2, ConvNeXt-T) nếu latency cho phép.
- Tốc độ:
  - Bật `torch.backends.cudnn.benchmark = True` khi dùng GPU, batch inference 1 khung nhưng tái sử dụng tensor/stream.
  - Chuyển model sang TorchScript hoặc ONNX + TensorRT trên Windows để giảm latency.
- Ổn định suy luận video:
  - Dùng cửa sổ trượt trên xác suất để làm mượt (ví dụ trung bình 5 khung) trước khi so `ALPHA_THR`.
  - Ghép heuristic phát hiện cảnh mới bằng `SCENE_DELTA_THR` với debounce thời gian tối thiểu giữa 2 lần lưu.
- UX:
  - Cho phép chọn `CKPT_PATH`, `MAP_PATH`, `ALPHA_THR` trực tiếp từ GUI và lưu cấu hình gần nhất.
  - Thêm nút toggle hiển thị heatmap/bbox.

