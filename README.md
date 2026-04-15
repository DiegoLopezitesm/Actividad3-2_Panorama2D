# Panorama Pipeline – Robots Autónomos

Pipeline completo en Python para calibración de cámara + creación de panoramas usando OpenCV (SIFT + Homografía + RANSAC).

---

## 🚀 ¿Qué hace este proyecto?

Este sistema permite:

1. Calibrar la cámara con tablero de ajedrez
2. Corregir distorsión de lente
3. Construir panoramas automáticamente

---

## 🧠 Tecnologías usadas

- Python 3.x
- OpenCV
- NumPy
- Matplotlib

---

## 📁 Estructura del proyecto

proyecto/
├── calib/                 # imágenes de calibración
├── images/                # imágenes del panorama
├── output/                # resultados generados
├── panorama_pipeline.py   # script principal
└── README.md

---

## ⚙️ Instalación

pip install opencv-python opencv-contrib-python numpy matplotlib

---

## ▶️ Uso

1. Coloca imágenes en calib/
2. Coloca imágenes en images/
3. Ejecuta:

python panorama_pipeline.py

---

## 📌 Notas

- Requiere traslape del 30–50%
- Necesitas mínimo 4 imágenes válidas para calibración

---
