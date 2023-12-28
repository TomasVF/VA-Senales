import cv2
import numpy as np

def encontrar_triangulos_rojos(imagen_path):
    # Cargar la imagen en formato PPM
    imagen = cv2.imread(imagen_path)

    # Convertir la imagen de BGR a HSV
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Definir el rango de color para el rojo en HSV
    rango_bajo = np.array([0, 100, 100])
    rango_alto = np.array([10, 255, 255])

    # Crear una máscara para el rango de color rojo
    mascara = cv2.inRange(hsv, rango_bajo, rango_alto)

    # Encontrar contornos en la máscara
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar los contornos para encontrar triángulos
    triangulos = []
    for contorno in contornos:
        perimetro = cv2.arcLength(contorno, True)
        aprox_poligono = cv2.approxPolyDP(contorno, 0.04 * perimetro, True)

        if len(aprox_poligono) == 3:
            triangulos.append(aprox_poligono)

    # Dibujar los triángulos encontrados en la imagen original
    for triangulo in triangulos:
        # Dibujar el borde del triángulo en rojo
        cv2.drawContours(imagen, [triangulo], 0, (0, 0, 255), 2)
        # Rellenar el triángulo con color blanco
        cv2.fillPoly(imagen, [triangulo], (255, 255, 255))

    # Mostrar la imagen original con los triángulos rojos encontrados
    cv2.imshow("Triángulos Rojos", imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ruta de la imagen PPM
imagen_path = "materialSenales/00023.ppm"

# Llamar a la función para encontrar triángulos rojos
encontrar_triangulos_rojos(imagen_path)
