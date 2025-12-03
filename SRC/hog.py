#clase para manejar histogramas de gradientes orientados (HOG)import cv2, y sus funcionalidades básicas, y graficos
import numpy as np
import matplotlib.pyplot as plt
import cv2

class HOG:
    def __init__(self, win_size=(64, 128), block_size=(16, 16), block_stride=(8, 8),
                 cell_size=(8, 8), nbins=9):
        # Convertir los argumentos a tuplas de enteros si no lo son
        win_size = tuple(map(int, win_size))
        block_size = tuple(map(int, block_size))
        block_stride = tuple(map(int, block_stride))
        cell_size = tuple(map(int, cell_size))
        
        # Inicializar el descriptor HOG con los argumentos corregidos
        self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

    def compute(self, image):
        """
        Calcula el descriptor HOG para una imagen dada.

        :param image: Imagen de entrada en escala de grises.
        :return: Descriptor HOG como un array numpy.
        """
        if len(image.shape) != 2:
            # Asegurarse de que la imagen esté en escala de grises
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        hog_descriptor = self.hog.compute(image)
        return hog_descriptor.flatten()

