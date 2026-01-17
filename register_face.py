import cv2
import os

NAME = "Fernanda"
DATASET_PATH = f"dataset/{NAME}"
TOTAL_IMAGES = 80

os.makedirs(DATASET_PATH, exist_ok=True)

# 0 normalmente es la cámara interna de la Mac
cap = cv2.VideoCapture(0)  
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

count = 0

print("Presiona S para capturar. Q para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se puede abrir la cámara")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        face = cv2.equalizeHist(face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Registro", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Captura al presionar S
        if len(faces) == 0:
            print("No se detecta rostro, intenta otra vez")
        else:
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (200, 200))
                face_img = cv2.equalizeHist(face_img)
                img_path = f"{DATASET_PATH}/{count:03}.jpg"
                cv2.imwrite(img_path, face_img)
                print(f"Imagen guardada: {img_path}")
                count += 1

    elif key == ord('q'):  # Salir al presionar Q
        break

    if count >= TOTAL_IMAGES:
        print("Se alcanzó el total de imágenes")
        break

cap.release()
cv2.destroyAllWindows()
print("Registro terminado")