import cv2
import os
import mediapipe as mp
import time

NAME = "name"
DATASET_PATH = f"dataset/{NAME}"
TOTAL_IMAGES = 100 
os.makedirs(DATASET_PATH, exist_ok=True)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
count = 0

print("El registro comenzará en 3 segundos...")
time.sleep(3)

while count < TOTAL_IMAGES:
    ret, frame = cap.read()
    if not ret: break

    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
            x, y = max(0, x), max(0, y)
            
            face = frame[y:y+h, x:x+w]
            if face.size > 0:
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_gray = cv2.resize(face_gray, (200, 200))
                face_gray = cv2.equalizeHist(face_gray)

                img_path = f"{DATASET_PATH}/{count:03}.jpg"
                cv2.imwrite(img_path, face_gray)
                count += 1
                
                # PAUSA para que te dé tiempo de moverte
                time.sleep(0.1) 

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"Captura {count}/{TOTAL_IMAGES}", (x, y-10), 1, 1.5, (255,0,0), 2)

    cv2.imshow("Registro (Muevete despacio)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()