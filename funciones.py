import cv2
import numpy as np
import showImg as si

 # Función que aplica erosion a una imagen
def erosion(imagen, kernelSize=5):
    # Definir un kernel para la erosión
    kernel = np.ones((kernelSize, kernelSize), np.uint8)

    # Aplicar erosión
    erosion_resultado = cv2.erode(imagen, kernel, iterations=1)
    return erosion_resultado

 # Función que aplica dilatación a una imagen
def dilatacion(imagen, kernelSize=5):
    # Definir un kernel para la erosión
    # Definir un kernel para la dilatación
    kernel = np.ones((kernelSize, kernelSize), np.uint8)

    # Aplicar dilatación
    dilatacion_resultado = cv2.dilate(imagen, kernel, iterations=1)

    return dilatacion_resultado


 # Función que aplica apertura a una imagen
def apertura(imagen, kernelSize = 5):
    # Definir un kernel para la operación de apertura
    kernel = np.ones((kernelSize, kernelSize), np.uint8)

    # Aplicar dilatación seguida de erosión (operación de apertura)
    apertura_resultado = cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)
    return apertura_resultado

 # Función que aplica cierre a una imagen
def cierre(imagen, kernelSize = 5):
     # Definir un kernel para la operación de cerradura
    kernel = np.ones((kernelSize, kernelSize), np.uint8)

    # Aplicar erosión seguida de dilatación (operación de cerradura)
    cerradura_resultado = cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel)
    return cerradura_resultado

 # Función que aplica laplace a una imagen
def laplace(imagen):
    v_channel = imagen[:, :, 2]
    # Aplicar el operador Laplaciano
    bordes_laplaciana = cv2.Laplacian(v_channel, cv2.CV_64F)

    # Convertir los valores negativos a positivos
    bordes_laplaciana = np.abs(bordes_laplaciana)

    # Convertir a 8 bits para mostrar
    bordes_laplaciana = np.uint8(bordes_laplaciana)
    return bordes_laplaciana


 # Función que busca circulos en una imagen
def detectCircles(imagen, edges, va=False):

    # Mascara que se utilizará más adelante
    mask_circles = np.zeros_like(imagen, dtype=np.uint8)

    # la imagen se utiliza en HSV
    imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # rangos de pixeles blancos
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 30, 255])

    # rangos de pixeles azules
    lower_blue = np.array([100, 100, 80])
    upper_blue = np.array([120, 255, 255])

    # rangos de pixeles rojos
    lower_red1 = np.array([0, 100, 80]) 
    upper_red1 = np.array([10, 255, 255])

    # rangos de pixeles rojos
    lower_red2 = np.array([160, 100, 80])
    upper_red2 = np.array([180, 255, 255])

    # Se aplica HoughCircles
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
                            param1=1500, param2=13, minRadius=5, maxRadius=50)
    
    # Si se ha detectado algún círculo se hacen varias comprobaciones para asegurarse de que es una señal
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Iterar a través de los círculos detectados
        filtered_circles = []
        for i in circles[0, :]:
            # Calcular la distancia entre el centro del círculo actual y los círculos existentes
            distances = np.sqrt(np.sum((circles[0, :, :2] - i[:2])**2, axis=1))

            # Contar cuántos círculos cercanos hay
            close_circles_count = np.sum(distances < 30)


            if va == True:
                maskadsf = np.zeros_like(imagen, dtype=np.uint8)
                cv2.circle(maskadsf, (i[0], i[1]), i[2], (255, 255, 255), thickness=-1)
                print(close_circles_count)
                si.mostrar_imagen(maskadsf)

            # Si hay mas de 2 círculos cercanos, se acepta el círculo
            if close_circles_count > 2:
                filtered_circles.append(i)
            else:
                # Si no puede que sea una señal de stop así que se hacen nuevas comprobaciones
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


        # En filered circles se encuentran todos los círculos que tienen más de dos círculos cercanos o se consideran stops
        # ahora se comprobará que estos círculos cumplan las características de color de las señales 
        if filtered_circles is not None:
            accepted_circles = []
            for i in filtered_circles:

                # imagenSoloInteriorCirculo contiene el círculo que se ha encontrado en la imagen
                mask1 = np.zeros_like(imagen, dtype=np.uint8)
                cv2.circle(mask1, (i[0], i[1]), i[2], (255, 255, 255), thickness=-1)
                imagenSoloInteriorCirculo = cv2.bitwise_and(imagen_hsv, mask1)

                # imagenSoloCircunferencia contiene la circunferencia con un grosor de 10 del círculo encontrado
                mask2 = np.zeros_like(imagen, dtype=np.uint8)
                cv2.circle(mask2, (i[0], i[1]), i[2], (255, 255, 255), thickness=10)
                imagenSoloCircunferencia = cv2.bitwise_and(imagen_hsv, mask2)


                mascara_roja22 = cv2.inRange(imagenSoloCircunferencia, lower_red2, upper_red2)
                mascara_roja12 = cv2.inRange(imagenSoloCircunferencia, lower_red1, upper_red1)
                mascara_roja02 = cv2.bitwise_or(mascara_roja12, mascara_roja22)

                # a es el porcentaje de pixeles rojos dentro de la circunferencia de grosor 10
                # como las señales criculares tienen rojo a lo largo de la circunferencia esto servirá como un filtro
                a = np.sum(mascara_roja02[i[1]-i[2]:i[1]+i[2], i[0]-i[2]:i[0]+i[2]] == 255) / np.sum(mask2 == (255, 255, 255))


                # Se crean máscaras de color rojo blanco y azul
                mascara_roja2 = cv2.inRange(imagenSoloInteriorCirculo, lower_red2, upper_red2)
                mascara_roja1 = cv2.inRange(imagenSoloInteriorCirculo, lower_red1, upper_red1)
                mascara_roja = cv2.bitwise_or(mascara_roja1, mascara_roja2)

                mascara_blanco = cv2.inRange(imagenSoloInteriorCirculo, lower_white, upper_white)

                mascara_azul = cv2.inRange(imagenSoloInteriorCirculo, lower_blue, upper_blue)


                # Se extrae la región de interés (ROI) para cada círculo
                roiR = mascara_roja[i[1]-i[2]:i[1]+i[2], i[0]-i[2]:i[0]+i[2]]

                roiA = mascara_azul[i[1]-i[2]:i[1]+i[2], i[0]-i[2]:i[0]+i[2]]

                # Se separa la imagen azul para comprobar la simetría de píxeles azules en los círculos
                ancho, alto = np.shape(roiA)

                mitad_superior = roiA[:ancho // 2, :]
                mitad_inferior = roiA[ancho // 2:, :]
                mitad_izquierda = roiA[:, :alto // 2]
                mitad_derecha = roiA[:, alto // 2:]

                

                if np.sum(mitad_inferior == 255) == 0:
                    simetria_azul_X = 0
                else:
                    simetria_azul_X = np.sum(mitad_superior == 255) / np.sum(mitad_inferior == 255)

                if np.sum(mitad_derecha == 255) == 0:
                    simetria_azul_Y = 0
                else:
                    simetria_azul_Y = np.sum(mitad_izquierda == 255) / np.sum(mitad_derecha == 255)


                roiB = mascara_blanco[i[1]-i[2]:i[1]+i[2], i[0]-i[2]:i[0]+i[2]]

                # Se calcula el porcentaje de píxeles de cada color para el ROI
                red_pixel_percentage = np.sum(roiR == 255) / float(roiR.size)

                blue_pixel_percentage = np.sum(roiA == 255) / float(roiA.size)
                
                white_pixel_percentage = np.sum(roiB == 255) / float(roiB.size)


                # si.mostrar_imagen(imagen[i[1]-i[2]:i[1]+i[2], i[0]-i[2]:i[0]+i[2]])

                # Se comprueban los porcentajes de color y el valor a para identificar las señales y su tipo
                if red_pixel_percentage > 0.6 and white_pixel_percentage > 0.03 and white_pixel_percentage < 0.1 and np.std(imagenSoloInteriorCirculo) > 5:
                 
                    color = (0, 0, 255)
                    label = "Stop"
                    accepted_circles.append([i, color, label])
                elif blue_pixel_percentage > 0.45 and simetria_azul_X > 0.8 and simetria_azul_Y > 0.8 and white_pixel_percentage > 0.01:
                    color = (255, 0, 0)
                    label = "Obligacion"
                    accepted_circles.append([i, color, label])
                elif white_pixel_percentage > 0.3 and red_pixel_percentage > 0.2 and a > 0.16:
                    color = (0, 255, 0)
                    label = "Prohibicion"
                    accepted_circles.append([i, color, label])
                else:
                    continue
            
        # accepted_circles contiene varios círculos para la misma señal en algunas situaciones, ahora nos quedaremos solo con
        # uno de los círculos, que será el más grande
        if accepted_circles is not None:
            final_circles = []
            for i in accepted_circles:
                cont = False
                for j in accepted_circles:
                    distance = np.sqrt(np.sum((i[0][:2] - j[0][:2])**2))
                    # si hay un círculo ceracano
                    if distance > 0 and distance < 100:
                        # si tiene un radio mayor al actual el actual se descarta
                        if i[0][2] < j[0][2]:
                            cont = True
                            break
                        else:
                            # si el actual tiene el radio mayor y no hay ya un círculo cercano aceptado se acepta el actual
                            for exists in final_circles:
                                distance2 = np.sqrt(np.sum((i[0][:2] - exists[0][:2])**2))
                                if distance2 > 0 and distance2 < 100:
                                    cont = True
                                    break
                if cont:
                    continue
                final_circles.append(i)

            # Se dibujan los círculos finales en la imagen
            for i in final_circles:
                circ = i[0]
                color = i[1]
                label = i[2]

                cv2.circle(imagen, (circ[0], circ[1]), circ[2], color, 2)

                
                cv2.circle(imagen, (circ[0], circ[1]), 2, (0, 0, 255), 3)


                cv2.putText(imagen, label, (circ[0]-int(circ[2]/2), circ[1]+int(circ[2]/2)+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Se dibuja en negro en la mascara que se había creado inicialmente, esto se aplicará para que las siguientes iteraciones no encuentren el mismo círculo
                cv2.circle(mask_circles, (i[0][0], i[0][1]), i[0][2], (255, 255, 255), thickness=cv2.FILLED)



    return imagen, cv2.bitwise_not(mask_circles)

# Función que busca triángulos en una imagen
def detectTriangles(imagen, edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterar sobre los contornos y filtrar los que parecen ser triángulos
    triangles = []
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 3:  # Si tiene tres vértices es un triángulo

            side_lengths = [
                cv2.norm(approx[1] - approx[0]),
                cv2.norm(approx[2] - approx[1]),
                cv2.norm(approx[0] - approx[2])
            ]

            # Calcular el área del triángulo
            area = cv2.contourArea(approx)

            # Verificar si el área es mayor que el valor mínimo y que el triángulo es aproximadamente equilátero
            if area >= 1000 and all(abs(side_lengths[i] - side_lengths[(i + 1) % 3]) < 0.1 * sum(side_lengths) for i in range(3)):
                triangles.append(approx)

    # Create a mask for the detected squares
    mask_triangles = np.zeros_like(imagen, dtype=np.uint8)
    mask_triangles2 = np.zeros_like(edges, dtype=np.uint8)

    # Dibujar los triángulos encontrados en la imagen original
    image_with_triangles = cv2.drawContours(imagen, triangles, -1, (0, 255, 0), 2)


    for triangle in triangles:

        # Obtener las coordenadas de los vértices del triángulo
        vertex1 = tuple(triangle[0][0])
        vertex2 = tuple(triangle[1][0])
        vertex3 = tuple(triangle[2][0])

        # Organizar los vértices en orden de coordenada y
        vertices_sorted = sorted([vertex1, vertex2, vertex3], key=lambda vertex: vertex[1])

        d1 = vertices_sorted[1][1] - vertices_sorted[0][1]

        d2 = vertices_sorted[2][1] - vertices_sorted[1][1]

        cv2.drawContours(mask_triangles, triangles, -1, (255, 255, 255), thickness=cv2.FILLED)

        cv2.drawContours(mask_triangles2, triangles, -1, (255, 255, 255), thickness=cv2.FILLED)


        M = cv2.moments(triangle)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Escribir el nombre debajo del triángulo
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Se diferencia entre ceda y peligro según si el triángulo apunta hacia arriba o hacia abajo

        if(d2>d1):
            cv2.putText(image_with_triangles, 'Ceda', (cx - 20, cy + 50), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(image_with_triangles, 'Peligro', (cx - 20, cy + 50), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)



    return image_with_triangles, cv2.bitwise_not(mask_triangles), cv2.bitwise_not(mask_triangles2)

# Función que busca cuadrados en la imagen
def detectSquares(imagen, edges):

    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([120, 255, 255])

    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    squares = []
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        x, y, w, h = cv2.boundingRect(contour)

        square_image = imagen[y:y+h, x:x+w]

        imagen_hsv = cv2.cvtColor(square_image, cv2.COLOR_BGR2HSV)

        mascara_azul = cv2.inRange(imagen_hsv, lower_blue, upper_blue)

        porcentaje_pixeles_azules = np.sum(mascara_azul==255)/ float(square_image.size)

        

        if len(approx) == 4:  # Si tiene cuatro vértices es un cuadrado
            # Se comprueba si los ángulos del cuadrado son aprox 90
            rect = cv2.minAreaRect(contour)
            angles = rect[-1]
            if 80 <= abs(angles) <= 95:
                # Se calcula el area
                area = cv2.contourArea(approx)

                # El area tiene que ser mayor a un valor
                if area >= 100:
                    # Las unícas señales cuadradas son azules así que se comprueba el porcentaje de píxeles azules
                    if porcentaje_pixeles_azules > 0.1:
                            counts = True
                            # Obtiene las coordenadas de los vértices
                            vertices = approx.reshape(-1, 2)
                            # Se comprueba que lso vértices estén suficientemente alinéados con otro vértice en el eje x e y
                            for i in range(4):
                                min1 = []
                                min2 = []
                                for j in range(4):
                                    if j == i: continue
                                    min1.append(np.abs(vertices[i][0] - vertices[j][0]))
                                    min2.append(np.abs(vertices[i][1] - vertices[j][1]))
                                if 1000*np.min(min1)/area > 40 or 1000*np.min(min2)/area > 40 :
                                    counts = False
                            if counts:
                                squares.append(approx)
    # Se pintan los cuadrados detectados en la imagen
    image_with_squares = cv2.drawContours(imagen, squares, -1, (0, 255, 0), 2)

    # La imagen con los cuadrados encontrados borrados
    mask_squares = np.zeros_like(imagen, dtype=np.uint8)
    cv2.drawContours(mask_squares, squares, -1, (255, 255, 255), thickness=cv2.FILLED)

    mask_squares2 = np.zeros_like(edges, dtype=np.uint8)
    cv2.drawContours(mask_squares2, squares, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Invert the mask to keep the areas outside the squares
    mask_inverse = cv2.bitwise_not(mask_squares)
    mask_inverse2 = cv2.bitwise_not(mask_squares2)


    for square in squares:
        # Get the bounding box of the square
        x, y, w, h = cv2.boundingRect(square)

        # Calculate the center of the bounding box
        cx = x + w // 2
        cy = y + h + 20  # Adjust the distance below the square

        # Put text below the square
        cv2.putText(image_with_squares, 'Indication', (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    return image_with_squares, mask_inverse, mask_inverse2


# Función que aplica canny a una imagen
def cannyHSV(imagen):

    # Apply Canny edge detection 
    edges = cv2.Canny(imagen, 50, 150)
    return edges

# Función que elimina todos los colores no necesarios de la imagen y aplica varios filtros para quedarnos solo con las partes necesarias
# red == True si queremos quedarnos solo con las señales rojas y false si queremos las azules
def elimOtherColors(img, red, showAll, value):
    imagen_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Se crean las máscaras de color necesarias
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
        

    # Aplicar desenfoque para eliminar zonas con tonos no uniformes
    mascara_suavizada = cv2.medianBlur(mascara_combinada, 5)
    if showAll: si.mostrar_imagen(mascara_suavizada)

    # Aplicar dilation para mejorar la máscara suavizada
    mascara_final = dilatacion(mascara_suavizada, 4)
    if showAll: si.mostrar_imagen(mascara_final)


    # Máscara de color blanco que se le aplica a la imagen dilatada para volver a hacer un filtro de color pero esta vez aceptando el color blanco también ya que 
    # las señales suelen tener color blanco en el interior
    lower_white = np.array([0, 0, 200])  
    upper_white = np.array([180, 30, 255])  
    mascara_blanco = cv2.inRange(imagen_hsv, lower_white, upper_white)

    mascara_prueba = cv2.bitwise_and((cv2.bitwise_or(mascara_combinada, mascara_blanco)), mascara_final)
    if showAll: si.mostrar_imagen(mascara_prueba)

    # Aplicar desenfoque para eliminar zonas con tonos no uniformes
    mascara_suavizada = cv2.medianBlur(mascara_prueba, value)
    if showAll: si.mostrar_imagen(mascara_suavizada)


    # Se aplica un cierre par aevitar trozos que corten una señal
    mascaraladf = cierre(mascara_suavizada, 18)

    return mascaraladf



# La misma función que su otra versión pero esta aplica menos filtros, a veces la imagen ya esta en buen estado y los filtros hacen que no se detecte alguna cosa
def elimOtherColorsSimple(img, red, showAll, value):
    imagen_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if red==False :
        lower_blue = np.array([100, 100, 50])
        upper_blue = np.array([120, 255, 255])
        mascara_azul = cv2.inRange(imagen_hsv, lower_blue, upper_blue)
        mascara_combinada = mascara_azul
    else:
        
        lower_red1 = np.array([0, 100, 80]) 
        upper_red1 = np.array([10, 255, 255]) 
        mascara_roja1 = cv2.inRange(imagen_hsv, lower_red1, upper_red1)

       
        lower_red2 = np.array([160, 100, 80])  
        upper_red2 = np.array([180, 255, 255])  
        mascara_roja2 = cv2.inRange(imagen_hsv, lower_red2, upper_red2)

       
        mascara_combinada = cv2.bitwise_or(mascara_roja1, mascara_roja2)

    if showAll: si.mostrar_imagen(mascara_combinada)
        


    # Aplicar desenfoque para eliminar zonas con tonos no uniformes
    mascara_suavizada = cv2.medianBlur(mascara_combinada, 5)
    if showAll: si.mostrar_imagen(mascara_suavizada)

    return mascara_suavizada