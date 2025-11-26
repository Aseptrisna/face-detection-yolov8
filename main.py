import os
import urllib.request
import cv2
import tkinter as tk
from ultralytics import YOLO

# ============================================
# 1. LINK MODEL WAJAH
# ============================================
FACE_MODEL = "yolov8n-face.pt"
FACE_MODEL_URL = (
    "https://github.com/akanametov/"
    "yolov8-face-detection/releases/download/v1.0/yolov8n-face.pt"
)

# ============================================
# 2. Download Model
# ============================================
def download_face_model():
    if os.path.exists(FACE_MODEL):
        print(f"[INFO] Model sudah ada: {FACE_MODEL}")
        return

    print("[INFO] Mengunduh model YOLO-Face...")
    try:
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL)
        print("[INFO] Download berhasil!")
    except Exception as e:
        print("[ERROR] Gagal mengunduh model:", e)
        exit()

# ============================================
# 3. Load Model
# ============================================
def load_face_detector():
    download_face_model()
    print("[INFO] Loading model wajah...")
    return YOLO(FACE_MODEL)

# ============================================
# Utility: Centering Window
# ============================================
def center_window(window_name, width, height):
    # Ambil ukuran layar
    root = tk.Tk()
    root.withdraw()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()

    # Hitung posisi center
    pos_x = (screen_w - width) // 2
    pos_y = (screen_h - height) // 2

    # Pindahkan window OpenCV
    cv2.moveWindow(window_name, pos_x, pos_y)

# ============================================
# 4. Deteksi Wajah
# ============================================
def detect_face():
    model = load_face_detector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Kamera tidak bisa dibuka.")
        return

    window_name = "Deteksi Wajah Akurat (YOLO-Face)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Resize jendela (opsional)
    default_w, default_h = 900, 600
    cv2.resizeWindow(window_name, default_w, default_h)

    # Center-kan window sekali (di awal)
    center_window(window_name, default_w, default_h)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.5)[0]

        # Bounding Box UI asli (tanpa modifikasi)
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = f"Wajah {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 200, 0), 2)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

# ============================================
# 5. Main
# ============================================
if __name__ == "__main__":
    detect_face()
