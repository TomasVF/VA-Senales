import cv2
import numpy as np


def showImgs(images):
    # Asumir que las imágenes son del mismo tamaño
    rows, cols, _ = images[0].shape

    # Crear una cuadrícula para organizar las imágenes
    grid_cols = 4
    grid_rows = 3
    grid = np.zeros((rows * grid_rows, cols * grid_cols, 3), dtype=np.uint8)

    # Poblar la cuadrícula con las imágenes
    for i in range(grid_rows):
        for j in range(grid_cols):
            index = i * grid_cols + j
            if index < len(images):
                grid[i * rows:(i + 1) * rows, j * cols:(j + 1) * cols] = images[index]


    # Crear la ventana y redimensionarla automáticamente
    cv2.namedWindow('Cuadrícula de Imágenes', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Cuadrícula de Imágenes', 800, 600)  # Establece el tamaño inicial (puedes ajustarlo según tus necesidades)

    # Mostrar la cuadrícula de imágenes
    cv2.imshow('Cuadrícula de Imágenes', grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


images = [cv2.imread(f"materialSenales/{i}.ppm") for i in range(1, 13)]


showImgs(images)