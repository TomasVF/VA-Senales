import cv2
import numpy as np
import showImg as si


def erosion(imagen, kernelSize=5):
    # Definir un kernel para la erosión
    kernel = np.ones((kernelSize, kernelSize), np.uint8)

    # Aplicar erosión
    erosion_resultado = cv2.erode(imagen, kernel, iterations=1)
    return erosion_resultado


def dilatacion(imagen, kernelSize=5):
    # Definir un kernel para la erosión
    # Definir un kernel para la dilatación
    kernel = np.ones((kernelSize, kernelSize), np.uint8)

    # Aplicar dilatación
    dilatacion_resultado = cv2.dilate(imagen, kernel, iterations=1)

    return dilatacion_resultado



def apertura(imagen, kernelSize = 5):
    # Definir un kernel para la operación de apertura
    kernel = np.ones((kernelSize, kernelSize), np.uint8)

    # Aplicar dilatación seguida de erosión (operación de apertura)
    apertura_resultado = cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)
    return apertura_resultado


def cerradura(imagen, kernelSize = 5):
     # Definir un kernel para la operación de cerradura
    kernel = np.ones((kernelSize, kernelSize), np.uint8)

    # Aplicar erosión seguida de dilatación (operación de cerradura)
    cerradura_resultado = cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel)
    return cerradura_resultado


def laplace(imagen):
    v_channel = imagen[:, :, 2]
    # Aplicar el operador Laplaciano
    bordes_laplaciana = cv2.Laplacian(v_channel, cv2.CV_64F)

    # Convertir los valores negativos a positivos
    bordes_laplaciana = np.abs(bordes_laplaciana)

    # Convertir a 8 bits para mostrar
    bordes_laplaciana = np.uint8(bordes_laplaciana)
    return bordes_laplaciana

def detectCircles(imagen, edges):

    imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200])  # Rango bajo para el matiz, saturación y valor
    upper_white = np.array([180, 30, 255])  # Rango alto para el matiz, saturación y valor

    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([120, 255, 255])

    # Define the range of color for red in the HSV color space
    lower_red1 = np.array([0, 100, 80])  # Lower range for hue, saturation, and value
    upper_red1 = np.array([10, 255, 255])  # Upper range for hue, saturation, and value

    # Define another range for red at the top of the spectrum
    lower_red2 = np.array([160, 100, 80])  # Lower range for hue, saturation, and value
    upper_red2 = np.array([180, 255, 255])  # Upper range for hue, saturation, and value

    # Apply HoughCircles to detect circles in the Canny edges
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
                            param1=1500, param2=16, minRadius=10, maxRadius=50)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Iterar a través de los círculos detectados
        filtered_circles = []
        for i in circles[0, :]:
            # Calcular la distancia entre el centro del círculo actual y los círculos existentes
            distances = np.sqrt(np.sum((circles[0, :, :2] - i[:2])**2, axis=1))

            # Contar cuántos círculos cercanos hay
            close_circles_count = np.sum(distances < 30)  # Ajusta el valor del umbral según sea necesario

            # Si hay mas de 10 círculos cercanos, procesar el círculo actual
            if close_circles_count > 3:
                filtered_circles.append(i)
            else:
                mask1 = np.zeros_like(imagen, dtype=np.uint8)
                cv2.circle(mask1, (i[0], i[1]), i[2], (255, 255, 255), thickness=-1)
                imagenSoloInteriorCirculo = cv2.bitwise_and(imagen_hsv, mask1)
                mascara_roja2 = cv2.inRange(imagenSoloInteriorCirculo, lower_red2, upper_red2)
                mascara_roja1 = cv2.inRange(imagenSoloInteriorCirculo, lower_red1, upper_red1)
                mascara_roja = cv2.bitwise_or(mascara_roja1, mascara_roja2)
                roiR = mascara_roja[i[1]-i[2]:i[1]+i[2], i[0]-i[2]:i[0]+i[2]]
                red_pixel_percentage = np.sum(roiR == 255) / float(roiR.size)
                if red_pixel_percentage > 0.5 and np.std(roiR) > 126:
                    filtered_circles.append(i)


        # If circles are found, draw them on the original image
        if filtered_circles is not None:
            accepted_circles = []
            for i in filtered_circles:
                mask1 = np.zeros_like(imagen, dtype=np.uint8)
                cv2.circle(mask1, (i[0], i[1]), i[2], (255, 255, 255), thickness=-1)
                imagenSoloInteriorCirculo = cv2.bitwise_and(imagen_hsv, mask1)

                mask2 = np.zeros_like(imagen, dtype=np.uint8)
                cv2.circle(mask2, (i[0], i[1]), i[2], (255, 255, 255), thickness=10)
                imagenSoloCircunferencia = cv2.bitwise_and(imagen_hsv, mask2)


                mascara_roja22 = cv2.inRange(imagenSoloCircunferencia, lower_red2, upper_red2)
                mascara_roja12 = cv2.inRange(imagenSoloCircunferencia, lower_red1, upper_red1)
                mascara_roja02 = cv2.bitwise_or(mascara_roja12, mascara_roja22)

                # a es el porcentaje de pixeles rojos dentro de la circunferencia de thickness 10
                a = np.sum(mascara_roja02[i[1]-i[2]:i[1]+i[2], i[0]-i[2]:i[0]+i[2]] == 255) / np.sum(mask2 == (255, 255, 255))


                mascara_roja2 = cv2.inRange(imagenSoloInteriorCirculo, lower_red2, upper_red2)
                mascara_roja1 = cv2.inRange(imagenSoloInteriorCirculo, lower_red1, upper_red1)
                mascara_roja = cv2.bitwise_or(mascara_roja1, mascara_roja2)

                mascara_blanco = cv2.inRange(imagenSoloInteriorCirculo, lower_white, upper_white)

                mascara_azul = cv2.inRange(imagenSoloInteriorCirculo, lower_blue, upper_blue)


                # Extract region of interest (ROI) for each circle
                roiR = mascara_roja[i[1]-i[2]:i[1]+i[2], i[0]-i[2]:i[0]+i[2]]

                roiA = mascara_azul[i[1]-i[2]:i[1]+i[2], i[0]-i[2]:i[0]+i[2]]

                roiB = mascara_blanco[i[1]-i[2]:i[1]+i[2], i[0]-i[2]:i[0]+i[2]]

                # Calculate the percentage of red pixels within the circle
                red_pixel_percentage = np.sum(roiR == 255) / float(roiR.size)

                blue_pixel_percentage = np.sum(roiA == 255) / float(roiA.size)
                
                white_pixel_percentage = np.sum(roiB == 255) / float(roiB.size)


                # si.mostrar_imagen(imagen[i[1]-i[2]:i[1]+i[2], i[0]-i[2]:i[0]+i[2]])

                # Draw the outer circle in green if red pixel percentage is above a threshold, otherwise draw in red
                if red_pixel_percentage > 0.5:
                    color = (0, 0, 255)
                    label = "Stop"
                    accepted_circles.append([i, color, label])
                elif blue_pixel_percentage > 0.5:
                    color = (255, 0, 0)
                    label = "Obligacion"
                    accepted_circles.append([i, color, label])
                elif (white_pixel_percentage+red_pixel_percentage) > 0.6 and a > 0.15:
                    color = (0, 255, 0)
                    label = "Prohibicion"
                    accepted_circles.append([i, color, label])
                else:
                    continue
                

        if accepted_circles is not None:
            final_circles = []
            for i in accepted_circles:
                cont = False
                if final_circles is None: 
                    final_circles.append(i)
                    continue
                for exist in final_circles:
                    distance = np.sqrt(np.sum((i[0][:2] - exist[0][:2])**2))
                    if distance > 0 and distance < 100:
                        cont = True
                        continue
                if cont:
                    continue
                final_circles.append(i)

            for i in final_circles:
                circ = i[0]
                color = i[1]
                label = i[2]

                cv2.circle(imagen, (circ[0], circ[1]), circ[2], color, 2)

                # Draw the center of the circle
                cv2.circle(imagen, (circ[0], circ[1]), 2, (0, 0, 255), 3)


                cv2.putText(imagen, label, (circ[0]-int(circ[2]/2), circ[1]+int(circ[2]/2)+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    return imagen


def detectTriangles(imagen, edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterar sobre los contornos y filtrar los que parecen ser triángulos
    triangles = []
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 3:  # El contorno es un triángulo

            side_lengths = [
                cv2.norm(approx[1] - approx[0]),
                cv2.norm(approx[2] - approx[1]),
                cv2.norm(approx[0] - approx[2])
            ]

            # Calcular el área del triángulo
            area = cv2.contourArea(approx)

            # Verificar si el área es mayor que el valor mínimo
            if area >= 1000 and all(abs(side_lengths[i] - side_lengths[(i + 1) % 3]) < 0.1 * sum(side_lengths) for i in range(3)):
                triangles.append(approx)

    for triangle in triangles:

        # Obtener las coordenadas de los vértices del triángulo
        vertex1 = tuple(triangle[0][0])
        vertex2 = tuple(triangle[1][0])
        vertex3 = tuple(triangle[2][0])

        # Organizar los vértices en orden de coordenada y
        vertices_sorted = sorted([vertex1, vertex2, vertex3], key=lambda vertex: vertex[1])

        d1 = vertices_sorted[1][1] - vertices_sorted[0][1]

        d2 = vertices_sorted[2][1] - vertices_sorted[1][1]

        # Obtener la máscara del triángulo
        mask_triangle = np.zeros_like(imagen, dtype=np.uint8)
        cv2.drawContours(mask_triangle, [triangle], -1, (255, 255, 255), thickness=cv2.FILLED)

        # Contar píxeles negros en la región del triángulo
        black_pixel_count = cv2.countNonZero(cv2.bitwise_not(cv2.cvtColor(mask_triangle, cv2.COLOR_BGR2GRAY)))




        M = cv2.moments(triangle)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Escribir el nombre debajo del triángulo
        font = cv2.FONT_HERSHEY_SIMPLEX

        if(d2>d1):
            cv2.putText(imagen, 'Ceda', (cx - 20, cy + 50), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(imagen, 'Peligro', (cx - 20, cy + 50), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # Dibujar los triángulos encontrados en la imagen original
    image_with_triangles = cv2.drawContours(imagen, triangles, -1, (0, 255, 0), 2)

    return image_with_triangles


def detectSquares(imagen, edges):

    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([120, 255, 255])

    # Find contours in the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterating over the contours and filtering those that appear to be squares
    squares = []
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        x, y, w, h = cv2.boundingRect(contour)

        square_image = imagen[y:y+h, x:x+w]

        imagen_hsv = cv2.cvtColor(square_image, cv2.COLOR_BGR2HSV)

        mascara_azul = cv2.inRange(imagen_hsv, lower_blue, upper_blue)

        porcentaje_pixeles_azules = np.sum(mascara_azul==255)/ float(square_image.size)

        

        if len(approx) == 4:  # The contour is a quadrilateral (could be a square)
            # Check if the angles are approximately 90 degrees
            rect = cv2.minAreaRect(contour)
            angles = rect[-1]
            if 80 <= abs(angles) <= 100:
                # Calculate the area of the quadrilateral
                area = cv2.contourArea(approx)

                # Check if the area is greater than the minimum value
                if area >= 100:
                    # Check the aspect ratio to see if it's roughly a square
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    if 0.5 <= aspect_ratio <= 1.1:
                        if porcentaje_pixeles_azules > 0.1:
                            squares.append(approx)
    # Draw the detected squares on the original image
    image_with_squares = cv2.drawContours(imagen, squares, -1, (0, 255, 0), 2)

    # Create a mask for the detected squares
    mask_squares = np.zeros_like(imagen, dtype=np.uint8)
    cv2.drawContours(mask_squares, squares, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Invert the mask to keep the areas outside the squares
    mask_inverse = cv2.bitwise_not(mask_squares)
    result_image = cv2.bitwise_and(imagen, mask_inverse)


    for square in squares:
        # Get the bounding box of the square
        x, y, w, h = cv2.boundingRect(square)

        # Calculate the center of the bounding box
        cx = x + w // 2
        cy = y + h + 20  # Adjust the distance below the square

        # Put text below the square
        cv2.putText(image_with_squares, 'Indication', (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    return image_with_squares, result_image



def cannyHSV(imagen):

    # Apply Canny edge detection 
    edges = cv2.Canny(imagen, 50, 150)
    return edges

# Eliminamos los colores que no aparecen en señales
def elimOtherColors(img, red, showAll, value):
    imagen_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if red==False :
        lower_blue = np.array([100, 100, 50])
        upper_blue = np.array([120, 255, 255])
        mascara_azul = cv2.inRange(imagen_hsv, lower_blue, upper_blue)
        mascara_combinada = mascara_azul
    else:
        # Definir el rango de color para el rojo en el espacio de color HSV
        lower_red1 = np.array([0, 100, 80])  # Rango bajo para el matiz, saturación y valor
        upper_red1 = np.array([10, 255, 255])  # Rango alto para el matiz, saturación y valor
        mascara_roja1 = cv2.inRange(imagen_hsv, lower_red1, upper_red1)

        # Definir otro rango para el rojo que se encuentra en la parte superior del espectro
        lower_red2 = np.array([160, 100, 80])  # Rango bajo para el matiz, saturación y valor
        upper_red2 = np.array([180, 255, 255])  # Rango alto para el matiz, saturación y valor
        mascara_roja2 = cv2.inRange(imagen_hsv, lower_red2, upper_red2)

        # Combinar ambas máscaras para cubrir todo el rango de colores rojos
        mascara_combinada = cv2.bitwise_or(mascara_roja1, mascara_roja2)

    if showAll: si.mostrar_imagen(mascara_combinada)
        


    # # Aplicar erode para eliminar lo que no sean lineas verticales
    # kernel = np.array([[1], [1], [1], [1], [1], [1], [1]])
    # mascara_erode = cv2.erode(mascara_combinada, kernel, iterations=1)
    # si.mostrar_imagen(mascara_erode)

    # # Aplicar erode para eliminar lo que no sean lineas horizontales
    # kernel = np.array([[1, 1, 1, 1, 1, 1, 1]])
    # mascara_erode2 = cv2.erode(mascara_combinada, kernel, iterations=1)
    # si.mostrar_imagen(mascara_erode2)


    # # mezclamos las imagenes resultantes de los dos erodes
    # mascara_erode_final = cv2.bitwise_or(mascara_erode, mascara_erode2)
    # si.mostrar_imagen(mascara_erode_final)


    # # Aplicar dilation para mejorar la máscara suavizada
    # kernel = np.ones((4, 4), np.uint8)
    # mascara_final = cv2.dilate(mascara_erode_final, kernel, iterations=1)

    # Aplicar desenfoque para eliminar zonas con tonos no uniformes
    mascara_suavizada = cv2.medianBlur(mascara_combinada, 5)
    if showAll: si.mostrar_imagen(mascara_suavizada)

    # Aplicar dilation para mejorar la máscara suavizada
    mascara_final = dilatacion(mascara_suavizada, 4)
    if showAll: si.mostrar_imagen(mascara_final)


    lower_white = np.array([0, 0, 200])  # Rango bajo para el matiz, saturación y valor
    upper_white = np.array([180, 30, 255])  # Rango alto para el matiz, saturación y valor
    mascara_blanco = cv2.inRange(imagen_hsv, lower_white, upper_white)

    mascara_prueba = cv2.bitwise_and((cv2.bitwise_or(mascara_combinada, mascara_blanco)), mascara_final)
    if showAll: si.mostrar_imagen(mascara_prueba)

    # Aplicar desenfoque para eliminar zonas con tonos no uniformes
    mascara_suavizada = cv2.medianBlur(mascara_prueba, value)
    if showAll: si.mostrar_imagen(mascara_suavizada)

    mascaraladf = cerradura(mascara_suavizada, 18)

    return mascaraladf
