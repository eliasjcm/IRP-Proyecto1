'''
    Reconocedor de dígitos escritos a mano - main.py

    Este programa utiliza histogramas para reconocer dígitos escritos a mano.
    Para ello, se utiliza la librería OpenCV para obtener los histogramas de cada
    imagen, y se comparan con los histogramas de las imágenes de entrenamiento.
    El programa se divide en 3 partes:
        - Separación de imágenes en imágenes de entrenamiento y de prueba
        - Cálculo de estadísticas
        - Prueba del programa
        


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


import os
from random import choice, seed
import shutil
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from histogram import get_histogram
count = 0


def splitImages():
    print("\nSeparando imagenes...")

    if not os.path.exists("test"):
        os.mkdir("test")

    if not os.path.exists("train"):
        os.mkdir("train")

    for dir in os.listdir("numbers"):
        print(f"Procesando directorio: {dir}")
        numbers = os.listdir(f"numbers/{dir}")
        amount = int(len(numbers)*0.3)

        if not os.path.exists(f"test/{dir}"):
            os.mkdir(f"test/{dir}")
        if not os.path.exists(f"train/{dir}"):
            os.mkdir(f"train/{dir}")

        for i in os.listdir(f"test/{dir}"):
            os.remove(f"test/{dir}/{i}")
        for i in os.listdir(f"train/{dir}"):
            os.remove(f"train/{dir}/{i}")

        for i in range(amount):
            digit = choice(numbers)
            numbers.remove(digit)

            shutil.copy(f"numbers/{dir}/{digit}", f"test/{dir}/{digit}")

        for file in numbers:
            shutil.copy(f"numbers/{dir}/{file}", f"train/{dir}/{file}")


def calc_stats():
    print("\nCalculando estadisticas...")
    means = []

    stds = []
    this_num_histograms = []
    np_histograms = []

    for dir in os.listdir("train"):
        this_num_histograms = []

        print(f"Procesando directorio: {dir}")
        for file in os.listdir(f"train/{dir}"):
            this_num_histograms.append(get_histogram(f"train/{dir}/{file}"))

        np_histograms = np.array(this_num_histograms)
        means.append(np_histograms.mean(axis=0))
        stds.append(np_histograms.std(axis=0))

    return means, stds


def test(means, stds):
    print("\nProbando...")

    total_errors = 0
    total_accerted = 0
    for dir in os.listdir("test"):
        print(f"Procesando directorio: {dir}")
        correct_count = 0
        incorrect_count = 0
        for file in os.listdir(f"test/{dir}"):

            result = recognize_num(f"test/{dir}/{file}", means, stds)

            if result == int(dir):
                correct_count += 1
            else:
                incorrect_count += 1

        print(f"El número {int(dir)} fue correcto {correct_count} veces")
        print(
            f"El número {int(dir)} fue incorrecto {incorrect_count} veces")
        print("El número fue correcto en un {0:.2f}% de las veces".format(
            (correct_count/(correct_count+incorrect_count))*100))

        total_errors += incorrect_count
        total_accerted += correct_count

    print("\n Resultados: ")
    print(f"El total de errores fue de: {total_errors}")
    print(f"El total de aciertos fue de: {total_accerted}")
    print(f"Total de imagenes: {total_errors+total_accerted}")
    print(
        f"El porcentaje de aciertos fue de: {round(total_accerted/(total_accerted+total_errors)*100,2)}%")


def recognize_num(img, means, stds):
    histogram = get_histogram(img)
    numbers_probabilities = []
    possible_numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    initial_possible_numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    std_arr_ratio = 2

    while len(initial_possible_numbers) > 1 and std_arr_ratio >= 1:
        numbers_probabilities = []

        for x in possible_numbers:
            mean_arr = means[x]
            std_arr = stds[x]

            results = abs(histogram - mean_arr) <= std_arr*std_arr_ratio
            numbers_probabilities.append(np.count_nonzero(results))

        possible_numbers = initial_possible_numbers.copy()
        initial_possible_numbers = []

        initial_possible_numbers = [possible_numbers[i] for i in range(len(numbers_probabilities)) if numbers_probabilities[i] >= len(
            means[i])*0.7 and numbers_probabilities[i] == max(numbers_probabilities)]

        if initial_possible_numbers != []:
            possible_numbers = np.array(initial_possible_numbers)

        std_arr_ratio -= 0.05

    if len(possible_numbers) != 0:
        return possible_numbers[0]
    return -1


if __name__ == "__main__":
    print("""
Reconocedor de dígitos escritos a mano utilizando Histogramas
Todos los archivos de codigo fuente usados para este proyecto están bajo la lincencia GNU General Public License version 3. Se permite la copia, distribucion y modificacion del software siempre que realice un seguimiento de los cambios/fechas en los archivos fuente. Cualquier modificación o software, incluido (a través del compilador) código con licencia GPL, también debe estar disponible bajo la GPL junto con las instrucciones de compilación e instalación.

Copyright (C) 2022  Roy Garcia Alvarado - rvga1311@estudiantec.cr & Abiel Porras Garro - abielpg@estudiantec.cr & Elias Castro Montero - eliasc5@estudiantec.cr & Fabián Rojas Arguedas - fabian.sajor26@estudiantec.cr"

Licencia: https://opensource.org/licenses/gpl-3.0.html
    """)
    # read seed from arguments if present
    if len(sys.argv) > 1:
        seed(int(sys.argv[1]))
    if not os.path.exists("numbers"):
        print("No se encontro el directorio 'numbers'. Por favor, ejecuta el script 'extract.py' para crear las imagenes")
        exit()
    splitImages()
    means, stds = calc_stats()
    test(means, stds)
