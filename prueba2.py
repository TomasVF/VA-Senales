import cv2
import numpy as np
import matplotlib.pyplot as plt


images = [cv2.imread(f"materialSenales/{i}.ppm") for i in range(1, 13)]


imagen = images[5]

# Convertir la imagen a espacio de color HSV
imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

# Definir un rango de colores rojos en el espacio HSV
rango_inferior = np.array([0, 100, 100], dtype=np.uint8)
rango_superior = np.array([10, 255, 255], dtype=np.uint8)

# Crear una máscara para los colores rojos
mascara_roja = cv2.inRange(imagen_hsv, rango_inferior, rango_superior)

# Aplicar la máscara a la imagen original
resaltado_rojo = cv2.bitwise_and(imagen, imagen, mask=mascara_roja)

# Mostrar las imágenes original y resaltada en rojo
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(resaltado_rojo, cv2.COLOR_BGR2RGB))
plt.title('Resaltado en Rojo')
plt.axis('off')

plt.show()
