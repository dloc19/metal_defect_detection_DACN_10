import sys, cv2
import numpy as np
from PyQt5 import QtGui
 # các hàm tiện ích: list_cameras, qimg_from_bgr, is_no_signal,...
def list_cameras(max_index=10):
    """Tìm các index camera khả dụng."""
    found = []
    backends_win = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    backends_mac = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    backends_linux = [cv2.CAP_V4L2, cv2.CAP_ANY]
    if sys.platform.startswith("win"): backends = backends_win
    elif sys.platform == "darwin":     backends = backends_mac
    else:                              backends = backends_linux
    for i in range(max_index):
        ok = False; cap=None
        for api in backends:
            cap = cv2.VideoCapture(i, api)
            if cap.isOpened():
                ok = True; break
            if cap: cap.release()
        if ok:
            found.append(i); cap.release()
    return found or [0]

def open_video_capture(source, width=1280, height=720):
    """Mở VideoCapture robust cho camera/video file."""
    if isinstance(source, int):
        if sys.platform.startswith("win"):
            candidates = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        elif sys.platform == "darwin":
            candidates = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        else:
            candidates = [cv2.CAP_V4L2, cv2.CAP_ANY]
        for api in candidates:
            cap = cv2.VideoCapture(source, api)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                for _ in range(5): cap.read()
                return cap
            else:
                cap.release()
        return cv2.VideoCapture(source)  # fallback
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def qimg_from_bgr(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h,w,ch = rgb.shape
    return QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)

def is_no_signal(frame_bgr) -> bool:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    m   = float(gray.mean())
    s   = float(gray.std())
    mn  = int(gray.min())
    mx  = int(gray.max())
    dyn = mx - mn
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = float((edges > 0).mean())
    if ((m < 6.0 or m > 249.0) and s < 2.0 and dyn < 8 and edge_ratio < 0.001):
        return True
    if mx < 10 or mn > 245:
        return True
    return False
