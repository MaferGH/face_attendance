import cv2
import numpy as np
import os

DATASET_PATH = "dataset"
MODEL_PATH = "models/lbph_model.yml"
LABELS_PATH = "models/labels.npy"

faces = []
labels = []
label_map = {}
current_label = 0

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

        faces.append(img)
        labels.append(current_label)

    current_label += 1

faces = np.array(faces)
labels = np.array(labels)

if len(faces) < 2:
    raise ValueError("No hay suficientes rostros para entrenar.")

recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

recognizer.train(faces, labels)

os.makedirs("models", exist_ok=True)
recognizer.save(MODEL_PATH)
np.save(LABELS_PATH, label_map)

print(f"Total de rostros cargados: {len(faces)}")
print("Modelo entrenado correctamente")
print("Mapa de etiquetas:", label_map)