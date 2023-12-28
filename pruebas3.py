import cv2
import numpy as np

def encontrar_circulos_rojos(imagen_path):
    # Cargar la imagen en formato PPM
    imagen = cv2.imread(imagen_path)

    # Convertir la imagen de BGR a HSV
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Definir el rango de color para el rojo en HSV
    rango_bajo = np.array([0, 100, 100])
    rango_alto = np.array([10, 255, 255])

    # Crear una máscara para el rango de color rojo
    mascara = cv2.inRange(hsv, rango_bajo, rango_alto)

    # Aplicar la máscara a la imagen original
    resultado = cv2.bitwise_and(imagen, imagen, mask=mascara)

    # Convertir la imagen a escala de grises
    gris = cv2.cvtColor(resultado, cv2.COLOR_BGR2GRAY)

    # Aplicar un desenfoque para suavizar la imagen
    gris = cv2.GaussianBlur(gris, (5, 5), 0)

    # Utilizar la transformada de Hough para encontrar círculos
    circulos = cv2.HoughCircles(
        gris,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=50
    )

    # Dibujar los círculos encontrados en la imagen original
    if circulos is not None:
        circulos = np.uint16(np.around(circulos))
        for i in circulos[0, :]:
            # Dibujar el círculo
            cv2.circle(imagen, (i[0], i[1]), i[2], (0, 255, 0), 2)

    # Mostrar la imagen original con los círculos encontrados
    cv2.imshow("Circulos Rojos", imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ruta de la imagen PPM
imagen_path = "materialSenales/00023.ppm"

# Llamar a la función para encontrar círculos rojos
encontrar_circulos_rojos(imagen_path)
