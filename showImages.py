import cv2

def mostrar_imagen_ppm(ruta_imagen):
    # Leer la imagen
    imagen = cv2.imread(ruta_imagen)

    if imagen is not None:
        # Mostrar la imagen
        cv2.imshow("Imagen", imagen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"No se pudo leer la imagen en {ruta_imagen}")

# Ruta de la imagen en formato PPM
ruta_imagen = "materialSenales/00023.ppm"

# Llamar a la función para mostrar la imagen
mostrar_imagen_ppm(ruta_imagen)
