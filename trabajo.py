import cv2
import numpy as np
import showImg as si
import funciones as fs

images = [cv2.imread(f"materialSenales/{i}.ppm") for i in range(1, 13)]

showAll = False


imagen = images[1]

si.mostrar_imagen(imagen)


# # Eliminar ruido
# imagen = cv2.GaussianBlur(imagen, (7,7), 0)
# si.mostrar_imagen(imagen)


# Dejamos solo los colores relevantes
imagenEnchanceR = fs.elimOtherColors2(imagen, True, showAll)
if showAll: si.mostrar_imagen(imagenEnchanceR)

imagenEnchanceA = fs.elimOtherColors2(imagen, False, showAll)
if showAll: si.mostrar_imagen(imagenEnchanceA)

# Canny
edgesR = fs.cannyHSV(imagenEnchanceR)
if showAll: si.mostrar_imagen(edgesR)

edgesA = fs.cannyHSV(imagenEnchanceA)
if showAll: si.mostrar_imagen(edgesA)


# edges2 = fs.laplace(imagenEnchance)
# si.mostrar_imagen(edges2)


# apertura = fs.apertura(edges2, 2)
# si.mostrar_imagen(apertura)

# dilatacion = fs.dilatacion(apertura, 3)
# si.mostrar_imagen(dilatacion)

# cerradura = fs.cerradura(edges2, 8)
# si.mostrar_imagen(cerradura)

# erosion = fs.erosion(edges2, 3)
# si.mostrar_imagen(erosion)




# Deteccion de circulos
imagenCirculosR = fs.detectCircles(imagen, edgesR)
if showAll: si.mostrar_imagen(imagenCirculosR)

imagenCirculosA = fs.detectCircles(imagen, edgesA)
if showAll: si.mostrar_imagen(imagenCirculosA)



# Dteccion de triangulos
imagenTriangulosR = fs.detectTriangles(imagen, edgesR)
if showAll: si.mostrar_imagen(imagenTriangulosR)

imagenTriangulosA = fs.detectTriangles(imagen, edgesA)
if showAll: si.mostrar_imagen(imagenTriangulosA)


# Dteccion de triangulos
imagenSquaresR = fs.detectSquares(imagen, edgesR)
if showAll: si.mostrar_imagen(imagenSquaresR)

imagenSquaresA = fs.detectSquares(imagen, edgesA)
si.mostrar_imagen(imagenSquaresA)




