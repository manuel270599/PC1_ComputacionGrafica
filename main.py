import cv2
import numpy as np
import pickle

# 1. Cargar la posición de la línea vertical
try:
    with open('linea.pkl', 'rb') as file:
        linea_x = pickle.load(file)
except FileNotFoundError:
    print("Error: Primero ejecuta configurar_linea.py para definir la línea.")
    exit()

# 2. Leer el video
video = cv2.VideoCapture('video_personas.mov')

# 3. Crear el sustractor de fondo (MOG2 es ideal para detectar movimiento)
# history: cuántos frames recuerda. varThreshold: sensibilidad al cambio.
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# Diccionario para rastrear a las personas: {id: (cx, cy)}
rastreador = {}
id_actual = 0

# Contadores
contador_izq_der = 0
contador_der_izq = 0

while True:
    check, img = video.read()
    if not check:
        break

    # 4. Aplicar sustracción de fondo (detecta el movimiento)
    imgMascara = fgbg.apply(img)

    # 5. Limpieza de ruido (Similar a tu código original: medianBlur y dilate)
    imgMedian = cv2.medianBlur(imgMascara, 5)
    kernel = np.ones((5, 5), np.uint8)
    imgDil = cv2.dilate(imgMedian, kernel, iterations=2)

    # 6. Encontrar contornos (las siluetas de las personas)
    contornos, _ = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 7. Dibujar la línea vertical
    h, w = img.shape[:2]
    cv2.line(img, (linea_x, 0), (linea_x, h), (0, 0, 255), 3)
    cv2.putText(img, "ZONA DE CONTEO", (linea_x - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 8. Procesar contornos y rastrear personas
    nuevos_centros = []
    for c in contornos:
        area = cv2.contourArea(c)
        # Filtrar por área para ignorar sombras o ruido pequeño
        if area > 1500: 
            x, y, w_rect, h_rect = cv2.boundingRect(c)
            cx = x + w_rect // 2
            cy = y + h_rect // 2
            nuevos_centros.append((cx, cy))
            
            # Dibujar el rectángulo de la persona detectada
            cv2.rectangle(img, (x, y), (x + w_rect, y + h_rect), (0, 255, 0), 2)

    # 9. Lógica de rastreo simple (evitar contar a la misma persona varias veces)
    # Emparejar centros nuevos con los del frame anterior
    ids_actualizados = {}
    for cx, cy in nuevos_centros:
        match = False
        for obj_id, (px, py) in rastreador.items():
            # Si el centro está cerca del centro anterior, es la misma persona
            if abs(cx - px) < 60 and abs(cy - py) < 60:
                ids_actualizados[obj_id] = (cx, cy)
                
                # 10. Detectar cruce de línea
                if px < linea_x and cx >= linea_x:
                    contador_izq_der += 1
                elif px > linea_x and cx <= linea_x:
                    contador_der_izq += 1
                
                match = True
                break
        
        # Si es una persona nueva, asignarle un ID
        if not match:
            ids_actualizados[id_actual] = (cx, cy)
            id_actual += 1

    rastreador = ids_actualizados

    # 11. Mostrar la información en pantalla
    cv2.putText(img, f"Izq a Der: {contador_izq_der}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(img, f"Der a Izq: {contador_der_izq}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(img, f"Total: {contador_izq_der + contador_der_izq}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow('Contador de Personas', img)
    # cv2.imshow('Mascara de Movimiento', imgDil) # Descomenta si quieres ver la detección en crudo

    # Presiona ESC para salir
    if cv2.waitKey(30) == 27:
        break

video.release()
cv2.destroyAllWindows()