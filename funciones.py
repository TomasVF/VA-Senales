import cv2
import numpy as np
import showImg as si


def erosion(imagen, kernelSize=5):
    # Definir un kernel para la erosión
    kernel = np.ones((kernelSize, kernelSize), np.uint8)

    # Aplicar erosión
    erosion_resultado = cv2.erode(imagen, kernel, iterations=1)
    return erosion_resultado


def dilatacion(imagen, kernelSize=5):
    # Definir un kernel para la erosión
    # Definir un kernel para la dilatación
    kernel = np.ones((kernelSize, kernelSize), np.uint8)

    # Aplicar dilatación
    dilatacion_resultado = cv2.dilate(imagen, kernel, iterations=1)

    return dilatacion_resultado



def apertura(imagen, kernelSize = 5):
    # Definir un kernel para la operación de apertura
    kernel = np.ones((kernelSize, kernelSize), np.uint8)

    # Aplicar dilatación seguida de erosión (operación de apertura)
    apertura_resultado = cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)
    return apertura_resultado


def cerradura(imagen, kernelSize = 5):
     # Definir un kernel para la operación de cerradura
    kernel = np.ones((kernelSize, kernelSize), np.uint8)

    # Aplicar erosión seguida de dilatación (operación de cerradura)
    cerradura_resultado = cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel)
    return cerradura_resultado


def laplace(imagen):
    v_channel = imagen[:, :, 2]
    # Aplicar el operador Laplaciano
    bordes_laplaciana = cv2.Laplacian(v_channel, cv2.CV_64F)

    # Convertir los valores negativos a positivos
    bordes_laplaciana = np.abs(bordes_laplaciana)

    # Convertir a 8 bits para mostrar
    bordes_laplaciana = np.uint8(bordes_laplaciana)
    return bordes_laplaciana

def detectCircles(imagen, edges):
    # Apply HoughCircles to detect circles in the Canny edges
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=150,
                            param1=1500, param2=16, minRadius=10, maxRadius=50)

    # If circles are found, draw them on the original image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the outer circle
            cv2.circle(imagen, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(imagen, (i[0], i[1]), 2, (0, 0, 255), 3)
    return imagen


def detectTriangles(imagen, edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterar sobre los contornos y filtrar los que parecen ser triángulos
    triangles = []
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 3:  # El contorno es un triángulo

            side_lengths = [
                cv2.norm(approx[1] - approx[0]),
                cv2.norm(approx[2] - approx[1]),
                cv2.norm(approx[0] - approx[2])
            ]

            # Calcular el área del triángulo
            area = cv2.contourArea(approx)

            # Verificar si el área es mayor que el valor mínimo
            if area >= 1000 and all(abs(side_lengths[i] - side_lengths[(i + 1) % 3]) < 0.1 * sum(side_lengths) for i in range(3)):
                triangles.append(approx)

    # Dibujar los triángulos encontrados en la imagen original
    image_with_triangles = cv2.drawContours(imagen, triangles, -1, (0, 255, 0), 2)

    return image_with_triangles


def detectSquares(imagen, edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterating over the contours and filtering those that appear to be squares
    squares = []
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:  # The contour is a quadrilateral (could be a square)
            # Calculate the area of the quadrilateral
            area = cv2.contourArea(approx)

            # Check if the area is greater than the minimum value
            if area >= 1000:
                # Check the aspect ratio to see if it's roughly a square
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.9 <= aspect_ratio <= 1.1:
                    squares.append(approx)

    # Draw the detected squares on the original image
    image_with_squares = cv2.drawContours(imagen, squares, -1, (0, 255, 0), 2)

    return image_with_squares



def cannyHSV(imagen):

    # Apply Canny edge detection 
    edges = cv2.Canny(imagen, 50, 150)
    return edges

# Eliminamos los colores que no aparecen en señales
def elimOtherColors(img, red, showAll):
    imagen_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if red==False :
        lower_blue = np.array([100, 100, 50])
        upper_blue = np.array([120, 255, 255])
        mascara_azul = cv2.inRange(imagen_hsv, lower_blue, upper_blue)
        mascara_combinada = mascara_azul
    else:
        # Definir el rango de color para el rojo en el espacio de color HSV
        lower_red1 = np.array([0, 100, 80])  # Rango bajo para el matiz, saturación y valor
        upper_red1 = np.array([10, 255, 255])  # Rango alto para el matiz, saturación y valor
        mascara_roja1 = cv2.inRange(imagen_hsv, lower_red1, upper_red1)

        # Definir otro rango para el rojo que se encuentra en la parte superior del espectro
        lower_red2 = np.array([160, 100, 80])  # Rango bajo para el matiz, saturación y valor
        upper_red2 = np.array([180, 255, 255])  # Rango alto para el matiz, saturación y valor
        mascara_roja2 = cv2.inRange(imagen_hsv, lower_red2, upper_red2)

        # Combinar ambas máscaras para cubrir todo el rango de colores rojos
        mascara_combinada = cv2.bitwise_or(mascara_roja1, mascara_roja2)

    if showAll: si.mostrar_imagen(mascara_combinada)
        


    # # Aplicar erode para eliminar lo que no sean lineas verticales
    # kernel = np.array([[1], [1], [1], [1], [1], [1], [1]])
    # mascara_erode = cv2.erode(mascara_combinada, kernel, iterations=1)
    # si.mostrar_imagen(mascara_erode)

    # # Aplicar erode para eliminar lo que no sean lineas horizontales
    # kernel = np.array([[1, 1, 1, 1, 1, 1, 1]])
    # mascara_erode2 = cv2.erode(mascara_combinada, kernel, iterations=1)
    # si.mostrar_imagen(mascara_erode2)


    # # mezclamos las imagenes resultantes de los dos erodes
    # mascara_erode_final = cv2.bitwise_or(mascara_erode, mascara_erode2)
    # si.mostrar_imagen(mascara_erode_final)


    # # Aplicar dilation para mejorar la máscara suavizada
    # kernel = np.ones((4, 4), np.uint8)
    # mascara_final = cv2.dilate(mascara_erode_final, kernel, iterations=1)

    # Aplicar desenfoque para eliminar zonas con tonos no uniformes
    mascara_suavizada = cv2.medianBlur(mascara_combinada, 5)
    if showAll: si.mostrar_imagen(mascara_suavizada)

    # Aplicar dilation para mejorar la máscara suavizada
    mascara_final = dilatacion(mascara_suavizada, 10)
    if showAll: si.mostrar_imagen(mascara_final)


    lower_white = np.array([0, 0, 200])  # Rango bajo para el matiz, saturación y valor
    upper_white = np.array([180, 30, 255])  # Rango alto para el matiz, saturación y valor
    mascara_blanco = cv2.inRange(imagen_hsv, lower_white, upper_white)

    mascara_prueba = cv2.bitwise_and((cv2.bitwise_or(mascara_combinada, mascara_blanco)), mascara_final)
    if showAll: si.mostrar_imagen(mascara_prueba)

    # Aplicar desenfoque para eliminar zonas con tonos no uniformes
    mascara_suavizada = cv2.medianBlur(mascara_prueba, 9)
    if showAll: si.mostrar_imagen(mascara_suavizada)

    # Aplicar la máscara a la imagen original
    resultado = cv2.bitwise_and(img, img, mask=mascara_suavizada)
    return mascara_suavizada





def elimOtherColors2(img,red, showAll):
    imagen_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if red==False :
        lower_blue = np.array([100, 100, 50])
        upper_blue = np.array([120, 255, 255])
        mascara_azul = cv2.inRange(imagen_hsv, lower_blue, upper_blue)
        mascara_combinada = mascara_azul
    else:
        # Definir el rango de color para el rojo en el espacio de color HSV
        lower_red1 = np.array([0, 100, 80])  # Rango bajo para el matiz, saturación y valor
        upper_red1 = np.array([10, 255, 255])  # Rango alto para el matiz, saturación y valor
        mascara_roja1 = cv2.inRange(imagen_hsv, lower_red1, upper_red1)

        # Definir otro rango para el rojo que se encuentra en la parte superior del espectro
        lower_red2 = np.array([160, 100, 80])  # Rango bajo para el matiz, saturación y valor
        upper_red2 = np.array([180, 255, 255])  # Rango alto para el matiz, saturación y valor
        mascara_roja2 = cv2.inRange(imagen_hsv, lower_red2, upper_red2)

        # Combinar ambas máscaras para cubrir todo el rango de colores rojos
        mascara_combinada = cv2.bitwise_or(mascara_roja1, mascara_roja2)
        
    if showAll: si.mostrar_imagen(mascara_combinada)

    # # Aplicar erode para eliminar lo que no sean lineas verticales
    # kernel = np.array([[1], [1], [1], [1], [1], [1], [1]])
    # mascara_erode = cv2.erode(mascara_combinada, kernel, iterations=1)
    # si.mostrar_imagen(mascara_erode)

    # # Aplicar erode para eliminar lo que no sean lineas horizontales
    # kernel = np.array([[1, 1, 1, 1, 1, 1, 1]])
    # mascara_erode2 = cv2.erode(mascara_combinada, kernel, iterations=1)
    # si.mostrar_imagen(mascara_erode2)


    # # mezclamos las imagenes resultantes de los dos erodes
    # mascara_erode_final = cv2.bitwise_or(mascara_erode, mascara_erode2)
    # si.mostrar_imagen(mascara_erode_final)


    # # Aplicar dilation para mejorar la máscara suavizada
    # kernel = np.ones((4, 4), np.uint8)
    # mascara_final = cv2.dilate(mascara_erode_final, kernel, iterations=1)

    # Aplicar desenfoque para eliminar zonas con tonos no uniformes
    mascara_suavizada = cv2.medianBlur(mascara_combinada, 5)
    if showAll: si.mostrar_imagen(mascara_suavizada)

    # Aplicar dilation para mejorar la máscara suavizada
    mascara_final = cerradura(mascara_suavizada, 5)
    if showAll: si.mostrar_imagen(mascara_final)

    # Aplicar la máscara a la imagen original
    resultado = cv2.bitwise_and(img, img, mask=mascara_final)
    return mascara_final