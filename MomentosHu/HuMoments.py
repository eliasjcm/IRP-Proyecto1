'''
    Reconocedor de dígitos escritos a mano - HuMoments.py

    Este programa utiliza los momentos de Hu para reconocer dígitos escritos a mano.
    Los momentos de Hu son invariantes a rotaciones, traslaciones y escalas.
    

    Copyright (C) 2022  Roy Garcia Alvarado - rvga1311@estudiantec.cr & Abiel Porras Garro - abielpg@estudiantec.cr & Elias Castro Montero - eliasc5@estudiantec.cr & Fabián Rojas Arguedas - fabian.sajor26@estudiantec.cr 

    Ultima modificacion: 2022-11-02
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


import os
from random import randint
import numpy as np
import cv2
from tabulate import tabulate
from math import copysign, log10


def getHuMoments():
    print('Getting Hu Moments...')

    huMoments = []
    for dir in os.listdir("HuImagesTest"):
        huMomentsByImage = []
        count = 0
        for image in os.listdir(f"HuImagesTest/{dir}"):
            huMomentsByImage.append([image])
            img = cv2.imread(
                f"HuImagesTest/{dir}/{image}", cv2.IMREAD_GRAYSCALE)
            _, im = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

            # Calculate Moments
            moments = cv2.moments(im)
            # Calculate Hu Moments
            huMomentsByImage[count] += list(cv2.HuMoments(moments))
            for i in range(1, 8):
                huMomentsByImage[count][i] = abs(-1 *
                                                 copysign(
                                                     1.0, huMomentsByImage[count][i]) * log10(abs(huMomentsByImage[count][i])))

            #huMomentsByImage[count] += list(huMoments)
            count += 1
        huMoments.append(huMomentsByImage)
    return huMoments


def getHuMomentsByImage(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    moments = cv2.moments(img)
    huMomentsByImage = cv2.HuMoments(moments)

    huMoments = []

    for i in range(7):
        huMoments.append(abs(-1 * copysign(1.0,
                         huMomentsByImage[i]) * log10(abs(huMomentsByImage[i]))))
    return huMoments


def tabulateResult(huMoments):
    for i in range(len(huMoments)):
        print(f"\nTable for {i}")
        print(tabulate(huMoments[i], headers=[
              "Image", "Hu1", "Hu2", "Hu3", "Hu4", "Hu5", "Hu6", "Hu7"]))


def saveImg(img, name):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(f"HuImagesTest/{name}", img)


def modify_image():
    print('Modificando imagenes...')

    for dir in os.listdir(f"HuImagesTest"):
        for img in os.listdir(f"HuImagesTest/{dir}"):
            if "_" in img:
                os.remove(f"HuImagesTest/{dir}/{img}")

    for dir in os.listdir("HuImagesTest"):
        for image in os.listdir(f"HuImagesTest/{dir}"):
            print("Processing image: ", image)
            img = cv2.imread(f"HuImagesTest/{dir}/{image}")
            name = image.split('.')[0]

            # resize image
            newDimension = randint(30, 80)
            newImg = cv2.resize(img, (newDimension, newDimension+17),
                                interpolation=cv2.INTER_AREA)
            saveImg(newImg, f"{dir}/{name}_resize.png")

            # rotate image
            rows, cols = img.shape[:2]
            M = cv2.getRotationMatrix2D(
                ((cols-1)/2.0, (rows-1)/2.0), randint(-70, 70), 1)
            newImg = cv2.warpAffine(img, M, (cols, rows))
            newImg = cv2.copyMakeBorder(
                newImg, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            saveImg(newImg, f"{dir}/{name}_rotate.png")

            # move image
            rows, cols = img.shape[:2]
            newImg = cv2.copyMakeBorder(
                img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            M = np.float32([[1, 0, randint(-20, 20)],
                           [0, 1, randint(-20, 20)]])
            rows, cols = newImg.shape[:2]
            newImg = cv2.warpAffine(newImg, M, (cols, rows))
            saveImg(newImg, f"{dir}/{name}_move.png")


if __name__ == "__main__":
    modify_image()
    huMoments = getHuMoments()
    tabulateResult(huMoments)

    # https://learnopencv.com/shape-matching-using-hu-moments-c-python/
