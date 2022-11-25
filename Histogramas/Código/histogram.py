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
