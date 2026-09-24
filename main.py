import cv2
import pickle
import numpy as np

estacionamientos = []
with open('espacios.pkl', 'rb') as file:
    estacionamientos = pickle.load(file)

video = cv2.VideoCapture('hola3.mov')

# Obtiene las dimensiones del video
ancho = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

print('Ancho: ', ancho)
print('Alto: ', alto)

contador = 0

while True:
    check, img = video.read()
    imgBN = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgTH = cv2.adaptiveThreshold(imgBN, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgTH, 5)
    kernel = np.ones((5,5), np.int8)
    imgDil = cv2.dilate(imgMedian, kernel)

    cuadrados_verdes = 0  # Variable para contar los cuadrados verdes

    for x, y, w, h in estacionamientos:
        espacio = imgDil[y:y+h, x:x+w]
        count = cv2.countNonZero(espacio)
        cv2.putText(img, str(count), (x,y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        if count < 490:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            cuadrados_verdes += 1  # Incrementa el contador de cuadrados verdes

    # Muestra el cuadro con el texto
    cv2.putText(img, f'Espacios libres: {cuadrados_verdes}', (ancho - 200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('video', img)
    # cv2.imshow('video TH', imgTH)
    # cv2.imshow('video Median', imgMedian)
    # cv2.imshow('video Dilatada', imgDil)

    key = cv2.waitKey(10)
    if key == 27:  # Presionar la tecla Esc para salir del bucle
        break

video.release()
cv2.destroyAllWindows()
