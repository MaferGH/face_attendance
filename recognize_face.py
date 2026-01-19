import cv2
import numpy as np
import sqlite3
import mediapipe as mp
from datetime import datetime

MODEL_PATH = "models/lbph_model.yml"
LABELS_PATH = "models/labels.npy"
DB_PATH = "attendance.db"
THRESHOLD = 80 # Un poco más alto ayuda si estás lejos

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# --- DB ---
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT,
    fecha TEXT,
    hora TEXT
)
""")
conn.commit()

# --- Modelo ---
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)
label_map = np.load(LABELS_PATH, allow_pickle=True).item()

cap = cv2.VideoCapture(0)
registered_today = set()

def register_attendance(name):
    ahora = datetime.now()
    fecha = ahora.strftime("%d/%m/%Y")
    hora = ahora.strftime("%H:%M:%S")
    
    c.execute("INSERT INTO attendance (nombre, fecha, hora) VALUES (?, ?, ?)", 
              (name, fecha, hora))
    conn.commit()
    print(f"Asistencia: {name} registrada a las {hora} el {fecha}")

while True:
    ret, frame = cap.read()
    if not ret: break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
            x, y = max(0, x), max(0, y)
            
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size > 0:
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
               
                # Filtro para suavizar y mejorar reconocimiento de lejos
                gray_face = cv2.GaussianBlur(gray_face, (3,3), 0)
                gray_face = cv2.resize(gray_face, (200, 200))
                gray_face = cv2.equalizeHist(gray_face)

                label, confidence = recognizer.predict(gray_face)

                if confidence < THRESHOLD:
                    name = label_map[label]
                    color = (0, 255, 0)
                    if name not in registered_today:
                        register_attendance(name)
                        registered_today.add(name)
                else:
                    name = "Desconocido"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{name}", (x, y - 10), 0, 0.7, color, 2)

    cv2.imshow("Sistema de Asistencia", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
conn.close()