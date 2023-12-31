import cv2
import numpy as np
import showImg as si
import funciones as fs

images = [cv2.imread(f"materialSenales/{i}.ppm") for i in range(1, 13)]

showAll = False


imagen = images[10]

si.mostrar_imagen(imagen)


image_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

# Separa los canales HSV
h, s, v = cv2.split(image_hsv)

# Aplica la ecualización de histograma al canal de valor (v)
equalized_v = cv2.equalizeHist(v)

# Combina los canales HSV con el canal de valor ecualizado
equalized_image_hsv = cv2.merge([h, s, equalized_v])

# Convierte la imagen de vuelta a BGR
equalized_image_bgr = cv2.cvtColor(equalized_image_hsv, cv2.COLOR_HSV2BGR)

if showAll:si.mostrar_imagen(equalized_image_bgr)




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
imagenCirculosR, maskDeleted0 = fs.detectCircles(imagen, edgesR)
if showAll: si.mostrar_imagen(imagenCirculosR)

imagenCirculosA, maskDeleted1 = fs.detectCircles(imagen, edgesA)
if showAll: si.mostrar_imagen(imagenCirculosA)



# Dteccion de triangulos
imagenTriangulosR, maskDeleted2 = fs.detectTriangles(imagen, edgesR)
if showAll: si.mostrar_imagen(imagenTriangulosR)


imagenSquaresA, maskDeleted3 = fs.detectSquares(imagen, edgesA)
if showAll: si.mostrar_imagen(imagenSquaresA)



maskDeletedFInal = cv2.bitwise_and(maskDeleted0, cv2.bitwise_and(maskDeleted1, cv2.bitwise_and(maskDeleted2, maskDeleted3)))



imagenDeleted = cv2.bitwise_and(imagen, maskDeletedFInal)
if showAll:si.mostrar_imagen(imagenDeleted)















# Dejamos solo los colores relevantes
imagenEnchanceR = fs.elimOtherColors(imagenDeleted, True, showAll, 9)
if showAll: si.mostrar_imagen(imagenEnchanceR)

imagenEnchanceA = fs.elimOtherColors(imagenDeleted, False, showAll, 9)
if showAll: si.mostrar_imagen(imagenEnchanceA)

# Canny
edgesR = fs.cannyHSV(imagenEnchanceR)
if showAll: si.mostrar_imagen(edgesR)

edgesA = fs.cannyHSV(imagenEnchanceA)
if showAll: si.mostrar_imagen(edgesA)





# Deteccion de circulos
imagenCirculosR, maskDeleted3 = fs.detectCircles(imagenSquaresA, edgesR)
if showAll: si.mostrar_imagen(imagenCirculosR)

imagenCirculosA, maskDeleted3 = fs.detectCircles(imagenSquaresA, edgesA)
if showAll: si.mostrar_imagen(imagenCirculosA)



# Dteccion de triangulos
imagenTriangulosR, maskDeleted3 = fs.detectTriangles(imagenSquaresA, edgesR)
if showAll: si.mostrar_imagen(imagenTriangulosR)


imagenSquaresA, maskDeleted3 = fs.detectSquares(imagenSquaresA, edgesA)
si.mostrar_imagen(imagenSquaresA)




