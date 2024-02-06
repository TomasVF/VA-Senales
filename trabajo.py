#hacer que al encontrar un cuadrado o un triangulo elimine en la imagen tambien, de esta forma no encuentra circulos donde no debería dentro de señales :)
# aun así, mejora alguna cosita pero da más problemas en otros lados


import cv2
import numpy as np
import showImg as si
import funciones as fs

images = [cv2.imread(f"materialSenales/{i}.ppm") for i in range(1, 13)]

showAll = True


imagen = images[0]



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

if showAll: si.mostrar_imagen(equalized_image_bgr)


if np.median(v) < 80:
    imagen = equalized_image_bgr











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

imagenResult = imagen.copy()

# Dteccion de triangulos
imagenTriangulosR, maskDeleted2, mask_edge = fs.detectTriangles(imagen, edgesR, imagenResult)
if showAll: si.mostrar_imagen(imagenTriangulosR)


imagenSquaresA, maskDeleted3, mask_edge2 = fs.detectSquares(imagen, edgesA, imagenTriangulosR)
if showAll: si.mostrar_imagen(imagenSquaresA)


maskEdgeA = cv2.bitwise_and(edgesA, mask_edge2)
maskEdgeR = cv2.bitwise_and(edgesR, mask_edge)




# Deteccion de circulos
imagenCirculosR, maskDeleted0 = fs.detectCircles(imagen, maskEdgeA, imagenSquaresA)
if showAll: si.mostrar_imagen(imagenCirculosR)

imagenCirculosA, maskDeleted1 = fs.detectCircles(imagen, maskEdgeR, imagenCirculosR)
if showAll: si.mostrar_imagen(imagenCirculosA)



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





# Dteccion de triangulos
imagenTriangulosR, maskDeleted2, mask_edge = fs.detectTriangles(imagenDeleted, edgesR, imagenCirculosA)
if showAll: si.mostrar_imagen(imagenTriangulosR)


imagenSquaresA, maskDeleted3, mask_edge2 = fs.detectSquares(imagenDeleted, edgesA, imagenTriangulosR)
if showAll: si.mostrar_imagen(imagenSquaresA)



maskEdgeA = cv2.bitwise_and(edgesA, mask_edge2)
maskEdgeR = cv2.bitwise_and(edgesR, mask_edge)



# Deteccion de circulos
imagenCirculosR, maskDeleted0 = fs.detectCircles(imagenDeleted, maskEdgeR, imagenSquaresA)
if showAll: si.mostrar_imagen(imagenCirculosR)

imagenCirculosA, maskDeleted1 = fs.detectCircles(imagenDeleted, maskEdgeA, imagenCirculosR)
if showAll: si.mostrar_imagen(imagenCirculosA)




maskDeletedFInal = cv2.bitwise_and(maskDeleted0, cv2.bitwise_and(maskDeleted1, cv2.bitwise_and(maskDeleted2, maskDeleted3)))


imagenDeleted = cv2.bitwise_and(imagenDeleted, maskDeletedFInal)
if showAll:si.mostrar_imagen(imagenDeleted)
















# Dejamos solo los colores relevantes
imagenEnchanceR = fs.elimOtherColorsSimple(imagenDeleted, True, showAll, 3)
if showAll: si.mostrar_imagen(imagenEnchanceR)

imagenEnchanceA = fs.elimOtherColorsSimple(imagenDeleted, False, showAll, 3)
if showAll: si.mostrar_imagen(imagenEnchanceA)

# Canny
edgesR = fs.cannyHSV(imagenEnchanceR)
if showAll: si.mostrar_imagen(edgesR)

edgesA = fs.cannyHSV(imagenEnchanceA)
if showAll: si.mostrar_imagen(edgesA)




# Dteccion de triangulos
imagenTriangulosR, maskDeleted2, mask_edge = fs.detectTriangles(imagenDeleted, edgesR, imagenCirculosA)
if showAll: si.mostrar_imagen(imagenTriangulosR)


imagenSquaresA, maskDeleted3, mask_edge2 = fs.detectSquares(imagenDeleted, edgesA, imagenTriangulosR)
if showAll: si.mostrar_imagen(imagenSquaresA)



maskEdgeA = cv2.bitwise_and(edgesA, mask_edge2)
maskEdgeR = cv2.bitwise_and(edgesR, mask_edge)



# Deteccion de circulos
imagenCirculosR, maskDeleted0 = fs.detectCircles(imagenDeleted, maskEdgeR, imagenSquaresA)
if showAll: si.mostrar_imagen(imagenCirculosR)

imagenCirculosA, maskDeleted1 = fs.detectCircles(imagenDeleted, maskEdgeA, imagenCirculosR)
if showAll: si.mostrar_imagen(imagenCirculosA)




si.mostrar_imagen(imagenCirculosA)
