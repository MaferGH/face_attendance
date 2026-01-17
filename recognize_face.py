import cv2
import numpy as np
import sqlite3
from datetime import datetime

MODEL_PATH = "models/lbph_model.yml"
LABELS_PATH = "models/labels.npy"
DB_PATH = "attendance.db"
THRESHOLD = 60

# --- DB ---
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT,
    fecha_hora TEXT
)
""")
conn.commit()

# --- Modelo ---
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)
label_map = np.load(LABELS_PATH, allow_pickle=True).item()

# --- Cámara ---
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# --- Control de duplicados ---
registered_today = set()

def register_attendance(name):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO attendance (nombre, fecha_hora) VALUES (?, ?)",
        (name, now)
    )
    conn.commit()
    print(f"Asistencia registrada: {name} {now}")

print("Presiona Q para salir")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("No frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        face = cv2.equalizeHist(face)

        label, confidence = recognizer.predict(face)

        if confidence < THRESHOLD:
            name = label_map[label]
            color = (0, 255, 0)

            if name not in registered_today:
                register_attendance(name)
                registered_today.add(name)
        else:
            name = "Desconocida"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame,
            f"{name} ({confidence:.1f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.imshow("Reconocimiento Facial", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()