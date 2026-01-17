# Face Attendance System (Local) 

Sistema de reconocimiento facial local para el registro de asistencia, desarrollado en Python. Este proyecto permite identificar personas a través de la cámara de la computadora y registrar automáticamente quién fue detectado y a qué hora.

## Características Principales
* **Reconocimiento en tiempo real:** Identifica usuarios al instante usando la webcam.
* **Funcionamiento 100% Local:** No requiere conexión a internet ni servicios en la nube.
* **Base de Datos Integrada:** Registro automático de nombre y hora en SQLite.
* **Prevención de Duplicados:** Evita múltiples registros de la misma persona durante una sesión.
* **Control por Teclado:** Interfaz sencilla para registro ('S') y salida ('Q').
 
## Tecnologías Utilizadas
* **Python 3:** Lenguaje principal del sistema.
* **OpenCV:** Captura de video y pre-procesamiento de imágenes.
* **LBPH (Local Binary Patterns Histograms):** Algoritmo para el reconocimiento facial.
* **NumPy:** Procesamiento de matrices de datos.
* **SQLite:** Almacenamiento local de asistencia.

## Arquitectura


El sistema sigue el siguiente flujo de datos:
1. **Cámara:** Captura de video mediante OpenCV.
2. **Detección:** Localización del rostro con Haar Cascades.
3. **Embeddings:** Extracción de rasgos faciales únicos.
4. **Identificación:** Comparación con la base de datos de rostros conocidos.
5. **Registro:** Almacenamiento en base de datos local.
   
## Estructura del Proyecto
* `register_face.py`: Captura de imágenes para nuevos usuarios.
* `train_faces.py`: Entrenamiento del modelo con las imágenes capturadas.
* `recognize_face.py`: Ejecución principal del reconocimiento y asistencia.
* `dataset/`: Carpeta que almacena las imágenes de entrenamiento.
* `models/`: Contiene el modelo entrenado (`lbph_model.yml`) y etiquetas.

## Guía de Uso

### 1. Instalación
```bash
# Clonar el repositorio
git clone [https://github.com/MaferGH/face_attendance.git](https://github.com/MaferGH/face_attendance.git)

# Crear entorno virtual
cd face_attendance
python -m venv venv
source venv/bin/activate 

# Instalar dependencias
pip uninstall opencv-python opencv-contrib-python -y
pip install -r requirements.txt

# Ejecutar el sistema
1. register_face.py
2. train_faces.py
3. recognize_face.py
```

### 2. Uso

Primero ejecuta `register_face.py` y toma las imagenes necesarias

<p align="center"><img width="786" height="655" alt="cap1" src="https://github.com/user-attachments/assets/3a47de21-ca74-4925-983f-8480add10e27" /></p>

Luego ejecuta `train_faces.py` para que reconozca al usuario y por último ejecuta `recognize_face.py`

<p align="center"><img width="610" height="706" alt="cap3" src="https://github.com/user-attachments/assets/fa80660f-902a-4e7a-80ce-ed19f52772e2" /></p>

###
> [!IMPORTANT]
> **Importante:** En este sistema (LBPH), **menor confidence = mejor coincidencia**. El umbral (THRESHOLD) está configurado en 60; si el valor obtenido es menor a esto, el sistema reconoce al usuario.
