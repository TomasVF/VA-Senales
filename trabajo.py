import cv2
import numpy as np

images = [cv2.imread(f"materialSenales/{i}.ppm") for i in range(1, 13)]


imagen = images[1]

cv2.imshow('Detección de Señales de Tráfico', imagen)
cv2.waitKey(0)


# Eliminar ruido
imagen = cv2.GaussianBlur(imagen, (7,7), 0)

cv2.imshow('Detección de Señales de Tráfico', imagen)
cv2.waitKey(0)



# Eliminamos los colores que no aparecen en señales
def elimOtherColors(img):
    imagen_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Definir el rango de color para el rojo en el espacio de color HSV
    rango_bajo = np.array([0, 100, 100])  # Rango bajo para el matiz, saturación y valor
    rango_alto = np.array([10, 255, 255])  # Rango alto para el matiz, saturación y valor

    # Crear una máscara utilizando inRange para seleccionar píxeles dentro del rango de color
    mascara_roja1 = cv2.inRange(imagen_hsv, rango_bajo, rango_alto)

    lower_blue = np.array([90, 40, 50])
    upper_blue = np.array([120, 255, 210])

    mascara_azul = cv2.inRange(imagen_hsv, lower_blue, upper_blue)

    # Definir otro rango para el rojo que se encuentra en la parte superior del espectro
    rango_bajo = np.array([160, 100, 100])  # Rango bajo para el matiz, saturación y valor
    rango_alto = np.array([180, 255, 255])  # Rango alto para el matiz, saturación y valor

    # Crear otra máscara para el rango superior del rojo
    mascara_roja2 = cv2.inRange(imagen_hsv, rango_bajo, rango_alto)

    # Combinar ambas máscaras para cubrir todo el rango de colores rojos
    mascara_roja = cv2.bitwise_or(mascara_roja1, cv2.bitwise_or(mascara_roja2, mascara_azul))

    # Aplicar la máscara a la imagen original
    resultado = cv2.bitwise_and(img, img, mask=mascara_roja)
    return resultado


imagenEnchance = elimOtherColors(imagen)

cv2.imshow('Detección de Señales de Tráfico', imagenEnchance)
cv2.waitKey(0)


v_channel = imagenEnchance[:, :, 2]

# Apply Canny edge detection 
edges = cv2.Canny(v_channel, 50, 150)


cv2.imshow('Detección de Señales de Tráfico', edges)
cv2.waitKey(0)

# Apply HoughCircles to detect circles in the Canny edges
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=150,
                           param1=200, param2=30, minRadius=20, maxRadius=50)

# If circles are found, draw them on the original image
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw the outer circle
        cv2.circle(imagen, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw the center of the circle
        cv2.circle(imagen, (i[0], i[1]), 2, (0, 0, 255), 3)

# Display the original image with detected circles
cv2.imshow('Detected Circles', imagen)
cv2.waitKey(0)






contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterar sobre los contornos y filtrar los que parecen ser triángulos
triangles = []
for contour in contours:
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) == 3:  # El contorno es un triángulo
        triangles.append(approx)

# Dibujar los triángulos encontrados en la imagen original
image_with_triangles = cv2.drawContours(imagen.copy(), triangles, -1, (0, 255, 0), 2)

# Display the original image with detected circles
cv2.imshow('Detected Circles', image_with_triangles)
cv2.waitKey(0)