#(MainWindow + main)
# main_gui.py
import sys, os, json, time, cv2
import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
import torch

# Import module tách riêng
import config
from utils import list_cameras, qimg_from_bgr, is_no_signal
from model import build_model, CamPlusPlus, heatmap_to_mask, mask_to_bboxes, overlay_heatmap, class_name_from_map
from worker import Worker

# ONNX Runtime support
try:
    from onnx_inference import create_onnx_inference
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️  ONNX Runtime not available. Using PyTorch inference.")

# ===================== THEME =====================
DARK_QSS = """
* { font-family: Segoe UI, Roboto, Arial; }
QMainWindow { background: #0f1115; }
QGroupBox {
  border: 1px solid #2a2f3a; border-radius: 10px; margin-top: 16px;
  background: #12151b; color: #d8dee9; font-weight: 600; padding-top: 10px;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #9aa5b1; }
QLabel { color: #d8dee9; }
QPushButton {
  background: #1f2430; color: #e6edf3; padding: 8px 14px; border-radius: 8px;
  border: 1px solid #2a2f3a;
}
QPushButton:hover { background: #283040; }
QPushButton:pressed { background: #1a1f29; }
QComboBox, QLineEdit {
  background: #151923; color: #d8dee9; padding: 6px 10px; border-radius: 8px;
  border: 1px solid #2a2f3a;
}
QStatusBar { background: #0f1115; color: #8a93a2; }
"""

# ===================== MainWindow =====================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Steel Surface QC")
        self.resize(1320, 860)

        QtWidgets.QApplication.instance().setStyleSheet(DARK_QSS)

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central); outer.setContentsMargins(14, 12, 14, 12); outer.setSpacing(12)

        # ==== Controls ====
        top_box = QtWidgets.QGroupBox("Input & Save")
        top = QtWidgets.QHBoxLayout(top_box); top.setSpacing(8)

        cams = list_cameras(10)
        self.cmb_cam = QtWidgets.QComboBox(); self.cmb_cam.addItems([str(i) for i in cams]); self.cmb_cam.setMinimumWidth(70)

        self.btn_cam = QtWidgets.QPushButton("Use Camera")
        self.btn_vid = QtWidgets.QPushButton("Open Video…")
        self.btn_img = QtWidgets.QPushButton("Open Image…")
        self.btn_dir = QtWidgets.QPushButton("Select Save Folder…")
        self.lbl_dir = QtWidgets.QLabel(os.path.abspath(config.SAVE_DIR))
        self.lbl_dir.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.lbl_dir.setStyleSheet("color:#9aa5b1;")

        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop  = QtWidgets.QPushButton("Stop"); self.btn_stop.setEnabled(False)

        top.addWidget(QtWidgets.QLabel("Camera:")); top.addWidget(self.cmb_cam); top.addWidget(self.btn_cam)
        top.addSpacing(10); top.addWidget(self.btn_vid); top.addWidget(self.btn_img)
        top.addSpacing(10); top.addWidget(self.btn_dir); top.addWidget(self.lbl_dir, 1)
        top.addSpacing(10); top.addWidget(self.btn_start); top.addWidget(self.btn_stop)
        outer.addWidget(top_box)

        # ==== Views ====
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.setStyleSheet("QSplitter::handle { background:#1a1f29; }")

        left_frame  = QtWidgets.QFrame();  left_frame.setStyleSheet("background:#0b0d12; border-radius:12px;")
        right_frame = QtWidgets.QFrame(); right_frame.setStyleSheet("background:#0b0d12; border-radius:12px;")
        for fr in (left_frame, right_frame): fr.setFrameShape(QtWidgets.QFrame.StyledPanel)

        layL = QtWidgets.QVBoxLayout(left_frame);  layL.setContentsMargins(10,10,10,10)
        layR = QtWidgets.QVBoxLayout(right_frame); layR.setContentsMargins(10,10,10,10)

        capL = QtWidgets.QLabel("Source"); capL.setStyleSheet("color:#9aa5b1; font-size:13px;")
        capR = QtWidgets.QLabel("Result View"); capR.setStyleSheet("color:#9aa5b1; font-size:13px;")
        self.lbl_left  = QtWidgets.QLabel(); self.lbl_left.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_right = QtWidgets.QLabel(); self.lbl_right.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_left.setMinimumSize(640, 420); self.lbl_right.setMinimumSize(640, 420)
        self.lbl_left.setStyleSheet("background:#111418; border:1px solid #242a33; border-radius:10px;")
        self.lbl_right.setStyleSheet("background:#111418; border:1px solid #242a33; border-radius:10px;")

        layL.addWidget(capL); layL.addWidget(self.lbl_left, 1)
        layR.addWidget(capR); layR.addWidget(self.lbl_right, 1)

        split.addWidget(left_frame); split.addWidget(right_frame); split.setSizes([660, 660])
        outer.addWidget(split, 1)

        # ==== Status ====
        status_box = QtWidgets.QFrame(); status_box.setStyleSheet("background:#0b0d12; border-radius:14px;")
        sb = QtWidgets.QVBoxLayout(status_box); sb.setContentsMargins(14,14,14,14)

        self.lbl_status = QtWidgets.QLabel("Waiting…")
        self.lbl_status.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_status.setMinimumHeight(120)
        self.lbl_status.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #132a13, stop:1 #0f2210);"
            "border:1px solid #1f4022; border-radius:16px; color:#eaffea; font-size:34px; font-weight:700;")
        sb.addWidget(self.lbl_status)
        outer.addWidget(status_box)

        # status bar
        self.status = QtWidgets.QStatusBar(); self.setStatusBar(self.status)
        dev = (torch.cuda.get_device_name(0) if config.DEVICE=='cuda' else 'CPU')
        self.status.showMessage(f"Device: {dev}   |   Model: MobileNetV2 + Grad-CAM++   |   α={config.ALPHA_THR}")

        # events
        self.btn_cam.clicked.connect(self.use_camera)
        self.btn_vid.clicked.connect(self.open_video)
        self.btn_img.clicked.connect(self.open_image)
        self.btn_dir.clicked.connect(self.pick_save_dir)
        self.btn_start.clicked.connect(self.start_stream)
        self.btn_stop.clicked.connect(self.stop_stream)

        # state
        self.source = (int(cams[0]) if cams else 0)
        self.save_dir = os.path.abspath(config.SAVE_DIR)
        self.idx_to_class, self.model, self.onnx_inference = self.load_model()
        self.cam_top = CamPlusPlus(self.model, config.CAM_TOP_LAYER) if self.model else None
        self.cam_aux = CamPlusPlus(self.model, config.CAM_AUX_LAYER) if self.model else None
        self.worker = None

    # ---------- helpers ----------
    def load_model(self):
        with open(config.MAP_PATH, "r", encoding="utf-8") as f:
            idx_to_class = json.load(f)
        
        # Try ONNX first if enabled and available
        if config.USE_ONNX and ONNX_AVAILABLE and os.path.exists(config.ONNX_PATH):
            print("🚀 Loading ONNX model for faster inference...")
            onnx_inference = create_onnx_inference(
                config.ONNX_PATH, 
                config.MAP_PATH,
                img_size=config.IMG_SIZE,
                use_gpu=config.ONNX_USE_GPU,
                alpha_thr=config.ALPHA_THR
            )
            if onnx_inference:
                print("✅ ONNX model loaded successfully")
                return idx_to_class, None, onnx_inference
            else:
                print("⚠️  ONNX loading failed, falling back to PyTorch")
        
        # Fallback to PyTorch
        print("🐍 Loading PyTorch model...")
        num_classes = len(idx_to_class)
        model = build_model(num_classes).to(config.DEVICE).eval()

        ckpt = torch.load(config.CKPT_PATH, map_location=config.DEVICE)
        state = ckpt.get("model", ckpt)
        new_state = {}
        for k,v in state.items():
            if k.startswith("module."): k = k[len("module."):]
            new_state[k] = v
        model.load_state_dict(new_state, strict=False)
        print("✅ PyTorch model loaded successfully")
        return idx_to_class, model, None

    def _show_on_label(self, lbl, img_bgr):
        h,w = img_bgr.shape[:2]
        maxw, maxh = lbl.width(), lbl.height()
        s = min(maxw/max(1,w), maxh/max(1,h))
        disp = cv2.resize(img_bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
        lbl.setPixmap(QtGui.QPixmap.fromImage(qimg_from_bgr(disp)))

    def _update_status(self, status_text, is_ok):
        if status_text == "No signal":
            self.lbl_status.setStyleSheet(
                "background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #2b2f3a, stop:1 #1b1f28);"
                "border:1px solid #3b4252; border-radius:16px; color:#e5e9f0; font-size:34px; font-weight:700;")
            self.lbl_status.setText("No signal")
        elif is_ok:
            self.lbl_status.setStyleSheet(
                "background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #164b2f, stop:1 #0f3a23);"
                "border:1px solid #246c45; border-radius:16px; color:#eaffea; font-size:34px; font-weight:800;")
            self.lbl_status.setText("✔ OK")
        else:
            self.lbl_status.setStyleSheet(
                "background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #5a1b1b, stop:1 #3d1111);"
                "border:1px solid #8b1e1e; border-radius:16px; color:#ffecec; font-size:34px; font-weight:800;")
            self.lbl_status.setText(status_text)

    # ---------- actions ----------
    def use_camera(self):
        self.source = int(self.cmb_cam.currentText())
        self.status.showMessage(f"Camera {self.source}")

    def open_video(self):
        fp, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open video", "", "Video files (*.mp4 *.avi *.mov *.mkv)")
        if fp:
            self.source = fp
            self.status.showMessage(f"Video: {fp}")

    def open_image(self):
        fp, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open image", "", "Image files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not fp: return
        img = cv2.imread(fp)
        if img is None:
            QtWidgets.QMessageBox.warning(self, "Error", "Không đọc được ảnh.")
            return

        self._show_on_label(self.lbl_left, img)

        if is_no_signal(img):
            right = np.zeros_like(img)
            self._show_on_label(self.lbl_right, right)
            self._update_status("No signal", True)
            return

        right, status_text, is_ok = self.infer_on_frame(img)

        self._show_on_label(self.lbl_right, right)
        self._update_status(status_text, is_ok)

        os.makedirs(self.save_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        if is_ok:
            cv2.imwrite(os.path.join(self.save_dir, f"{stamp}_IMG_OK_raw.jpg"), img)
            cv2.imwrite(os.path.join(self.save_dir, f"{stamp}_IMG_OK_view.jpg"), right)
        else:
            cls_name = status_text.replace("✖", "").strip().split()[0]
            cv2.imwrite(os.path.join(self.save_dir, f"{stamp}_IMG_{cls_name}_raw.jpg"), img)
            cv2.imwrite(os.path.join(self.save_dir, f"{stamp}_IMG_{cls_name}_view.jpg"), right)

        self.status.showMessage(f"Predicted image: {fp}")

    def pick_save_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select save folder", self.save_dir)
        if d:
            self.save_dir = d
            self.lbl_dir.setText(d)

    def start_stream(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self.worker = Worker(self.source, self.model, self.idx_to_class, self.save_dir, self.onnx_inference)
        self.worker.frame_ready.connect(self.update_frames)
        self.worker.start()
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
        self.status.showMessage(f"Streaming: {self.source}")

    def stop_stream(self):
        if self.worker:
            self.worker.stop(); self.worker = None
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        self.status.showMessage("Stopped.")

    @QtCore.pyqtSlot(np.ndarray, np.ndarray, str, bool)
    def update_frames(self, left_bgr, right_bgr, status_text, is_ok):
        self._show_on_label(self.lbl_left, left_bgr)
        self._show_on_label(self.lbl_right, right_bgr)
        self._update_status(status_text, is_ok)

    def infer_on_frame(self, frame_bgr):
        """Ảnh đơn: trả (view, status_text, is_ok) theo ngưỡng α + CAM."""
        view = frame_bgr.copy()
        
        # Use ONNX inference if available
        if self.onnx_inference:
            cls_name, max_conf, probs = self.onnx_inference.predict(frame_bgr)
            
            if max_conf < config.ALPHA_THR:
                return view, "OK", True
            
            # For ONNX, we can't do Grad-CAM easily, so just return the prediction
            return view, f"✖ {cls_name} ({max_conf:.2f})", False
        
        # Fallback to PyTorch inference with Grad-CAM
        if not self.model:
            return view, "Model not loaded", False
            
        x = config.TF(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        cls_idx  = int(np.argmax(probs))
        cls_name = class_name_from_map(self.idx_to_class, cls_idx)
        max_conf = float(probs[cls_idx])

        if max_conf < config.ALPHA_THR:
            return view, "OK", True

        H, W = frame_bgr.shape[:2]
        def try_cam(cam_obj):
            cam = cam_obj(x)
            hm  = cv2.resize(cam, (W, H), interpolation=cv2.INTER_CUBIC)
            pct = config.PCT_START
            while pct >= config.PCT_MIN:
                mask = heatmap_to_mask(hm, percentile=pct, min_area=config.MIN_AREA)
                bxs  = mask_to_bboxes(mask)
                if len(bxs) > 0:
                    return bxs, hm
                pct -= config.PCT_STEP
            return [], hm

        if self.cam_top and self.cam_aux:
            bboxes, hm = try_cam(self.cam_top)
            if len(bboxes) == 0:
                bboxes, hm = try_cam(self.cam_aux)

            if len(bboxes) > 0:
                for (x1,y1,w,h) in bboxes:
                    cv2.rectangle(view, (x1,y1), (x1+w,y1+h), (0,255,0), 2)
            else:
                view = overlay_heatmap(view, hm)

        return view, f"✖ {cls_name} ({max_conf:.2f})", False

    def closeEvent(self, e):
        self.stop_stream()
        try:
            if self.cam_top: self.cam_top.remove()
            if self.cam_aux: self.cam_aux.remove()
        except Exception:
            pass
        e.accept()
#main
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec_())
