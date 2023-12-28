import cv2
import numpy as np
import showImg as si
import funciones as fs

images = [cv2.imread(f"materialSenales/{i}.ppm") for i in range(1, 13)]


imagen = images[1]

si.mostrar_imagen(imagen)


# Eliminar ruido
imagen = cv2.GaussianBlur(imagen, (7,7), 0)
si.mostrar_imagen(imagen)


# Dejamos solo los colores relevantes
imagenEnchance = fs.elimOtherColors(imagen)
si.mostrar_imagen(imagenEnchance)


# Canny
edges = fs.cannyHSV(imagenEnchance)
si.mostrar_imagen(edges)



# Deteccion de circulos
imagenCirculos = fs.detectCircles(imagen, edges)
si.mostrar_imagen(imagenCirculos)



# Dteccion de triangulos
imagenTriangulos = fs.detectCircles(imagen, edges)
si.mostrar_imagen(imagenTriangulos)