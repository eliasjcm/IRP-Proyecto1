'''
    Reconocedor de dígitos escritos a mano - crop_center.py

    Se encarga de recortar las imágenes de las plantillas de los números
    y guardarlas en la carpeta numbers. Se utiliza para entrenar el modelo
    de reconocimiento de dígitos.


    Copyright (C) 2022  Roy Garcia Alvarado - rvga1311@estudiantec.cr & Abiel Porras Garro - abielpg@estudiantec.cr & Elias Castro Montero - eliasc5@estudiantec.cr & Fabián Rojas Arguedas - fabian.sajor26@estudiantec.cr 

    Ultima modificacion: 2022-11-10
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



import numpy as np
import cv2

kernel = np.ones((3, 3), np.uint8)


def filters(img):

    img = cv2.GaussianBlur(img, (5, 5), 0)

    img = cv2.bitwise_not(img)
    for i in range(3):
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)

    img = cv2.bitwise_not(img)

    return img


def center_image(img):
    minX = minY = maxX = maxY = -1
    height, width = img.shape
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    whiteCount = 0
    blackFlag = False
    for j in range(width-2):
        whiteFlag = False
        for i in range(height-2):
            if img[i, j] == 0:
                blackFlag = True
                whiteFlag = True
                whiteCount = 0
                minX = j if minX == -1 else min(minX, j)
                maxX = j if maxX == -1 else max(maxX, j)
                minY = i if minY == -1 else min(minY, i)
                maxY = i if maxY == -1 else max(maxY, i)
        if not whiteFlag and blackFlag:
            whiteCount += 1
        if whiteCount > 7:
            break

    newImage = img[minY:maxY, minX:maxX]
    newImage = cv2.copyMakeBorder(
        newImage, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    newImage = cv2.resize(newImage, (50, 67), interpolation=cv2.INTER_AREA)
    _, newImage = cv2.threshold(
        newImage, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return newImage


def crop_image(img: np.array, x1, x2, y1, y2, num, nameID):
    varName = nameID
    distanceX = (x2 - x1)//17
    distanceY = (y2 - y1)//3

    for i in range(3):
        for j in range(17):
            crop = img[(y1+12) + i*distanceY:(y1-5) + (i+1)*distanceY,
                       (x1+13) + j*distanceX:(x1-1) + (j+1)*distanceX]


            crop = cv2.bitwise_not(crop)
            crop = cv2.resize(crop, (50, 67), interpolation=cv2.INTER_AREA)
            crop = cv2.erode(crop, kernel, iterations=1)
            crop = cv2.Canny(crop, 50, 200)
            _, crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            if np.mean(crop) != 0:
                cv2.imwrite(f'numbers/{num}/'+str(num) + '_' +
                            str(varName) + '.png', crop)
            varName += 1
