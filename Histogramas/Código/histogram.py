'''
    Reconocedor de dígitos escritos a mano - histogram.py

    Programa que crea histogramas de las imágenes de los dígitos 
    y retorna una lista con los histogramas de cada imagen.
    Se calculan los histogramas verticales y horizontales de cada imagen.
      

    Copyright (C) 2022  Roy Garcia Alvarado - rvga1311@estudiantec.cr & Abiel Porras Garro - abielpg@estudiantec.cr & Elias Castro Montero - eliasc5@estudiantec.cr & Fabián Rojas Arguedas - fabian.sajor26@estudiantec.cr 

    Ultima modificacion: 2022-10-25
    Responsables: Roy Garcia Alvarado - rvga1311@estudiantec.cr & Abiel Porras Garro - abielpg@estudiantec.cr & Elias Castro Montero - eliasc5@estudiantec.cr & Fabián Rojas Arguedas - fabian.sajor26@estudiantec.cr 
    Resumen: Optimización de código y comentarios.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

from cProfile import label
from itertools import count
from math import ceil
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import os


inputPath = "Samples"
outputPath = "Histograms"


def getHistogram():
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    images = os.listdir(inputPath)

    for name in images:
        imagePath = inputPath + "/" + name
        image = cv.imread(imagePath, 0)

        total_x = ceil(image.shape[0] / 4)
        total_y = ceil(image.shape[1] / 4)
        axis = np.zeros(total_x + total_y)
        labels = np.zeros(total_x + total_y)
        pos = 0

        for i in range(0, total_y):
            initial_col = i * 4
            axis[pos] = np.count_nonzero(cv.bitwise_not(
                image[:, initial_col:initial_col+4]))
            labels[pos] = pos
            pos += 1

        for i in range(0, total_x):
            initial_row = i * 4
            axis[pos] = np.count_nonzero(cv.bitwise_not(
                image[initial_row:initial_row+4]))
            labels[pos] = pos
            pos += 1

        plt.cla()
        plt.bar(labels, axis)
        plt.savefig(f"{outputPath}/histogram_{name}")


def get_histogram(img):
    image = cv.imread(img, 0)
    kernel = np.ones((3, 3), np.uint8)
    image = cv.dilate(image, kernel, iterations=2)
    image = cv.erode(image, kernel, iterations=1)

    total_x = image.shape[0]
    total_y = image.shape[1]
    axis = np.zeros(total_x + total_y)
    pos = 0

    for i in range(0, total_y, 4):
        axis[pos] = np.count_nonzero(image[:, i:i+4])
        pos += 1

    for i in range(0, total_x, 4):
        axis[pos] = np.count_nonzero(image[i:i+4])
        pos += 1

    return axis
