import cv2
import numpy as np
import os

DATASET_PATH = "dataset"
MODEL_PATH = "models/lbph_model.xml"  # CAMBIADO A .XML PARA MAC M4
LABELS_PATH = "models/labels.npy"

faces = []
labels = []
label_map = {}
current_label = 0

def augment_image(image):
    """Crea variaciones de la foto para reconocer de lado y con luz distinta"""
    variations = [image]
    variations.append(cv2.flip(image, 1)) # Espejo
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    for angle in [-10, 10]: # Rotaciones
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        variations.append(cv2.warpAffine(image, M, (w, h)))
    return variations

print("Entrenando modelo...")

for person in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person)
    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # Multiplicamos las fotos para que el modelo sea más inteligente
        augmented = augment_image(img)
        for f in augmented:
            faces.append(f)
            labels.append(current_label)
    current_label += 1

if len(faces) < 2:
    print("Error: Necesitas más fotos en el dataset.")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=10, grid_x=8, grid_y=8)
recognizer.train(faces, np.array(labels))

os.makedirs("models", exist_ok=True)
# IMPORTANTE: .save en XML es más estable en Mac 
recognizer.save(MODEL_PATH) 
np.save(LABELS_PATH, label_map)

print(f"Éxito. Rostros procesados: {len(faces)}")