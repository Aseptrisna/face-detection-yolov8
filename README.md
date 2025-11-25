# YOLOv8 Face Detection

Project ini menyediakan script Python untuk melakukan deteksi wajah menggunakan YOLOv8.
Jika model khusus wajah tidak ada, sistem akan otomatis mengunduh model tersebut dari Ultralytics.

------------------------------------------------------------
Requirements
------------------------------------------------------------

Package yang dibutuhkan:

1. ultralytics  
   - Package resmi YOLOv8.  
   - Digunakan untuk load model YOLO, melakukan inference, dan mengelola file model.  

2. opencv-python  
   - Digunakan untuk akses kamera, membaca frame video, menggambar bounding box, dan menampilkan hasil deteksi.  

Install dependencies:

pip install -r requirements.txt

Isi file requirements.txt:

ultralytics
opencv-python

------------------------------------------------------------
File Utama: yolo_face_auto.py
------------------------------------------------------------

Kode lengkap deteksi wajah YOLOv8 dengan auto-download model:

------------------------------------------------------------

import os
import cv2
from ultralytics import YOLO

FACE_MODEL = "yolov8n-face.pt"
DEFAULT_MODEL = "yolov8n.pt"

def load_model():
    """
    Load model YOLO.
    Jika model wajah tidak ada, Ultralytics akan otomatis mendownload.
    Jika gagal download → fallback ke yolov8n.pt.
    """
    if os.path.exists(FACE_MODEL):
        print(f"[INFO] Menggunakan model lokal: {FACE_MODEL}")
        return YOLO(FACE_MODEL)

    print("[INFO] Model wajah tidak ditemukan, mencoba download otomatis...")
    try:
        model = YOLO(FACE_MODEL)
        print("[INFO] Model wajah berhasil diunduh.")
        return model
    except:
        print("[WARN] Gagal download model wajah! Menggunakan model default YOLOv8.")
        return YOLO(DEFAULT_MODEL)

def detect_face(source=0):
    model = load_model()
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Gagal membuka kamera/video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.4)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                label = f"{model.names.get(cls, 'face')} {conf:.2f}"

                cv2.rectangle(frame, (int(x1), int(y1)),
                              (int(x2), int(y2)), (0,255,0), 2)
                cv2.putText(frame, label,
                            (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0,255,0), 2)

        cv2.imshow("YOLOv8 Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_face(0)

------------------------------------------------------------
Cara Menjalankan
------------------------------------------------------------

1. Install dependencies:
   pip install -r requirements.txt

2. Jalankan program:
   python yolo_face_auto.py

------------------------------------------------------------
Struktur Folder
------------------------------------------------------------

project/
│
├─ main.py  
├─ requirements.txt  
└─ README.md  

------------------------------------------------------------

