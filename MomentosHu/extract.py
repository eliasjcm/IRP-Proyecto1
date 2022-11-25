'''
    Reconocedor de dígitos escritos a mano - extract.py

    Este código representa funcionalidades necesarias para la extracción de los
    diferentes especimenes de una huella digital por cada hoja de usuario recolectada.
    Este programa fue realizado usando openCV que está bajo 
    la licencia Apache 2 (siendo está compatible con GLP V3). De esta 
    biblioteca se usaron las funciones basicas para el manipulamiento de
    imagenes


    Copyright (C) 2022  Roy Garcia Alvarado - rvga1311@estudiantec.cr & Abiel Porras Garro - abielpg@estudiantec.cr & Elias Castro Montero - eliasc5@estudiantec.cr & Fabián Rojas Arguedas - fabian.sajor26@estudiantec.cr 

    Ultima modificacion: 2022-11-24
    Responsables: Roy Garcia Alvarado - rvga1311@estudiantec.cr & Abiel Porras Garro - abielpg@estudiantec.cr & Elias Castro Montero - eliasc5@estudiantec.cr & Fabián Rojas Arguedas - fabian.sajor26@estudiantec.cr 
    Resumen: Creacion del codigo para la extraccion de los especimenes de una huella digital por cada hoja de usuario recolectada. Limpieza de codigo.

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

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from math import atan2, degrees
import crop_center as cc

margin = 20


def get_angle(point_1, point_2):
    angle = atan2(point_1[0] - point_2[0], point_1[1] - point_2[1])
    return degrees(angle)


def alignAux(img, point1, point2):
    angle = get_angle(point2, point1)
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols//2, rows//2), angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows),
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255))
    return dst


def getPointLeft(img, stroke=0, height=None):
    if height == None:
        height, width = img.shape[:2]
    else:
        width = img.shape[1]
    points = []
    for x in reversed(range(height)):
        for y in range(0, width//6):
            if len(points) <= 2:
                if img[x][y] == 255:
                    points.append([x, y])
                    break
            else:
                break
    point = min(points, key=lambda x: x[1], default=None)
    return [point[0]-stroke, point[1]+stroke]


def getPointRight(img, stroke=0, height=None):
    if height == None:
        height, width = img.shape[:2]
    else:
        width = img.shape[1]
    points = []
    for x in reversed(range(height)):
        for y in reversed(range(width//6*5, width)):
            if len(points) <= 2:
                if img[x][y] == 255:
                    points.append([x, y])
                    break
            else:
                break
    point = max(points, key=lambda x: x[1], default=None)
    return [point[0]-stroke, point[1]-stroke]


def getPointTop(img, startPoint, stroke=0):
    points = []
    while not (img[startPoint[0]][startPoint[1]] == 0):
        for x in reversed(range(0, startPoint[0])):
            if img[x][startPoint[1]] == 0:
                points.append([x+stroke+1, startPoint[1]])
                startPoint[1] += 1
                break
    return min(points, key=lambda x: x[0], default=None)


def show_wait_destroy(winname, img):
    plt.imshow(img, cmap=plt.cm.gray)
    plt.title(winname)
    plt.show()


def align(img, file):

    height, width = img.shape[:2]

    gray = cv2.bitwise_not(img)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 15, -2)

    horizontal = np.copy(bw)

    cols = horizontal.shape[1]
    horizontal_size = cols // 30

    horizontalStructure = cv2.getStructuringElement(
        cv2.MORPH_RECT, (horizontal_size, 1))

    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    point1 = getPointLeft(horizontal)
    point2 = getPointRight(horizontal)

    while (point2[1]-point1[1] > width*0.9) or (point1[0]+100 < point2[0] or point2[0]+100 < point1[0]):
        horizontal = horizontal[0:point2[0]-10, 0:width]
        point1 = getPointLeft(horizontal)
        point2 = getPointRight(horizontal)

    img = alignAux(img, point1, point2)

    img = img[margin:height-margin, margin:width-margin]

    return img


def getVerticals(gray):

    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 15, -2)

    vertical = np.copy(bw)

    rows = vertical.shape[0]
    verticalsize = rows // 30

    verticalStructure = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, verticalsize))

    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    return vertical


def getRects(verticals, file):
    stroke = 0
    rectList = []
    heightPoint = verticals.shape[0]
    startLeftPoint = getPointLeft(verticals, stroke, heightPoint)
    endLeftPoint = getPointTop(verticals, startLeftPoint, stroke)
    topLenght = startLeftPoint[0] - endLeftPoint[0]

    for i in range(5):
        startLeftPoint = getPointLeft(verticals, stroke, heightPoint)
        rightPoint = getPointRight(verticals, stroke, heightPoint)[1]
        rectList.append([[startLeftPoint[0]-topLenght, startLeftPoint[1]], [startLeftPoint[0] -
                        topLenght, rightPoint], startLeftPoint, [startLeftPoint[0], rightPoint]])
        heightPoint = startLeftPoint[0]-topLenght-5

    return rectList


def makeFolders(pageType):
    if not os.path.exists("numbers"):
        os.mkdir("numbers")
    if pageType == 4:
        for i in range(0, 5):
            if os.path.exists(f"numbers/{i}"):
                shutil.rmtree(f"numbers/{i}")
            os.mkdir(f"numbers/{i}")
    elif pageType == 9:
        for i in range(5, 10):
            if os.path.exists(f"numbers/{i}"):
                shutil.rmtree(f"numbers/{i}")
            os.mkdir(f"numbers/{i}")


def main():
    print("""
Extractor de dígitos de plantillas con números escritos a mano
Todos los archivos de codigo fuente usados para este proyecto están bajo la lincencia GNU General Public License version 3. Se permite la copia, distribucion y modificacion del software siempre que realice un seguimiento de los cambios/fechas en los archivos fuente. Cualquier modificación o software, incluido (a través del compilador) código con licencia GPL, también debe estar disponible bajo la GPL junto con las instrucciones de compilación e instalación.

Copyright (C) 2022  Roy Garcia Alvarado - rvga1311@estudiantec.cr & Abiel Porras Garro - abielpg@estudiantec.cr & Elias Castro Montero - eliasc5@estudiantec.cr & Fabián Rojas Arguedas - fabian.sajor26@estudiantec.cr"

Licencia: https://opensource.org/licenses/gpl-3.0.html
    """)

    folderList = ["Page1", "Page2"]

    for folder in folderList:
        number = 0
        if folder == "Page1":
            number = 4
        elif folder == "Page2":
            number = 9

        makeFolders(number)

        id = 0
        for file in os.listdir(folder):

            img = cv2.imread(f"{folder}/" + file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cc.filters(img)

            print(f"Procesando: {file}")
            gray = align(img, file)
            rectList = getRects(getVerticals(gray), file)

            for i in range(len(rectList)):
                cc.crop_image(gray, rectList[i][0][1], rectList[i][1][1],
                              rectList[i][0][0], rectList[i][2][0], number-i, id)

            id += 51


if __name__ == "__main__":
    main()
