import cv2
import numpy as np
import showImg as si
import funciones as fs

images = [cv2.imread(f"materialSenales/{i}.ppm") for i in range(1, 13)]

showAll = False


imagen = images[10]

si.mostrar_imagen(imagen)


# # Eliminar ruido
# imagen = cv2.GaussianBlur(imagen, (7,7), 0)
# si.mostrar_imagen(imagen)


# Dejamos solo los colores relevantes
imagenEnchanceR = fs.elimOtherColors(imagen, True, showAll, 3)
if showAll: si.mostrar_imagen(imagenEnchanceR)

imagenEnchanceA = fs.elimOtherColors(imagen, False, showAll, 3)
if showAll: si.mostrar_imagen(imagenEnchanceA)

# Canny
edgesR = fs.cannyHSV(imagenEnchanceR)
if showAll: si.mostrar_imagen(edgesR)

edgesA = fs.cannyHSV(imagenEnchanceA)
if showAll: si.mostrar_imagen(edgesA)


# Deteccion de circulos
imagenCirculosR = fs.detectCircles(imagen, edgesR)
if showAll: si.mostrar_imagen(imagenCirculosR)

imagenCirculosA = fs.detectCircles(imagen, edgesA)
if showAll: si.mostrar_imagen(imagenCirculosA)



# Dteccion de triangulos
imagenTriangulosR = fs.detectTriangles(imagen, edgesR)
if showAll: si.mostrar_imagen(imagenTriangulosR)


imagenSquaresA, imageDeleted = fs.detectSquares(imagen, edgesA)
if showAll: si.mostrar_imagen(imagenSquaresA)









# Dejamos solo los colores relevantes
imagenEnchanceR = fs.elimOtherColors(imagen, True, showAll, 9)
if showAll: si.mostrar_imagen(imagenEnchanceR)

imagenEnchanceA = fs.elimOtherColors(imagen, False, showAll, 9)
if showAll: si.mostrar_imagen(imagenEnchanceA)

# Canny
edgesR = fs.cannyHSV(imagenEnchanceR)
if showAll: si.mostrar_imagen(edgesR)

edgesA = fs.cannyHSV(imagenEnchanceA)
if showAll: si.mostrar_imagen(edgesA)





# Deteccion de circulos
imagenCirculosR = fs.detectCircles(imagenSquaresA, edgesR)
if showAll: si.mostrar_imagen(imagenCirculosR)

imagenCirculosA = fs.detectCircles(imagenSquaresA, edgesA)
if showAll: si.mostrar_imagen(imagenCirculosA)



# Dteccion de triangulos
imagenTriangulosR = fs.detectTriangles(imagenSquaresA, edgesR)
if showAll: si.mostrar_imagen(imagenTriangulosR)


imagenSquaresA = fs.detectSquares(imagenSquaresA, edgesA)
si.mostrar_imagen(imagenSquaresA)




