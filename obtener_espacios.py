import cv2
import pickle

# Leer una imagen o el primer frame de tu video
# Si tienes un video, puedes extraer el primer frame con cv2.VideoCapture
img = cv2.imread('pasaje.jpg') # Cambia esto por el nombre de tu imagen de fondo
if img is None:
    # Si no tienes imagen, extraemos el primer frame del video
    video = cv2.VideoCapture('video.mp4')
    ret, img = video.read()
    video.release()

print("Selecciona una franja delgada donde quieras poner la línea vertical...")
# Seleccionar la zona de la línea (una franja delgada)
roi = cv2.selectROI('Selecciona la zona de la linea', img, False)
cv2.destroyWindow('Selecciona la zona de la linea')

x, y, w, h = roi
# La línea vertical estará justo en el centro de la franja que seleccionaste
linea_x = x + w // 2

print(f"Línea guardada en la coordenada X: {linea_x}")

with open('linea.pkl', 'wb') as file:
    pickle.dump(linea_x, file)