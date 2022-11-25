# IRP-Proyecto 1
Introducción al Reconocimiento de Patrones - Proyecto 1

## Instituto Tecnologico de Costa Rica
### Escuela de Computacion

### IC8046 - Introducción al Reconocimiento de Patrones

Profesor: Eduardo Adolfo Canessa Montero

Estudiantes: 
+ Elias Castro Montero - 2020098930 - eliasc5@estudiantec.cr
+ Roy Vinicio Garcia Alvarado - 2020109283 - rvga1311@estudiantec.cr
+ Fabián Rojas Arguedas - 2019380107 - fabian.sajor26@estudiantec.cr
+ Abiel Porras Garro - 2020209597 - abielpg@estudiantec.cr

Semestre II 2022

# Proyecto

El objetivo de este proyecto es crear un sistema de reconocimiento de dígitos a mano  y comparar diferentes métodos para realizarlo, así como también se pretende entender su funcionamiento con el fin de obtener el conocimiento suficiente para poder ser aplicado y adaptado en diferentes situaciones. Se muestran tres diferentes métodos reconocedores: Histogramas, Momentos invariantes de Hu y Redes Neuronales. Los primeros dos se tomaron de las tareas de investigación 4 y 5 realizadas para el curso de Introducción al Reconocimiento de Patrones en el segundo semestre del año 2022, el tercer método se basó en el código desarrollado por [Patel](https://youtu.be/iqQgED9vV7k). Con los tres métodos, se pretende ver una comparación y concluir cuál es el mejor y el peor.

# Licencia

Todos los archivos de codigo fuente usados para este proyecto están bajo la lincencia GNU General Public License version 3. Se permite la copia, distribucion y modificacion del software siempre que realice un seguimiento de los cambios/fechas en los archivos fuente. Cualquier modificación o software, incluido (a través del compilador) código con licencia GPL, también debe estar disponible bajo la GPL junto con las instrucciones de compilación e instalación.

Copyright (C) 2022  Roy Garcia Alvarado - rvga1311@estudiantec.cr & Abiel Porras Garro - abielpg@estudiantec.cr & Elias Castro Montero - eliasc5@estudiantec.cr & Fabián Rojas Arguedas - fabian.sajor26@estudiantec.cr"

[Licencia](https://github.com/rvga1311/IRP-FingerprintRecognition/blob/main/LICENSE)

# Datos
El sistema fue desarrollado en Python 3.10.5 utilizando Jupyter Notebooks, la biblioteca OpenCV para el procesamiento de imagenes, y Tensorflow para el desarrollo de la red neuronal.

## Requerimientos para usar el sistema
+ Python 3.8 o superior
+ Biblioteca ```OpenCV``` para el procesamiento y manipulación de imágenes.
+ Biblioteca ```numpy``` para el manejo de matrices (formato en el que las imágenes son almacenadas).
+ Biblioteca ```os``` para el manejo de archivos y carpetas.
+ Biblioteca ```shutil``` como complemento para el manejo de archivos y carpetas.
+ Biblioteca ```tensorflow``` para la implementación de las redes neuronales.
+ Biblioteca ```random ``` para calcular los datasets de imágenes de manera pseudoaleatoria.
+ Biblioteca ```pandas ``` para mostrar datos probabilísticos salidos de la red neuronal.
+ Biblioteca ```IPython.display``` para mostrar renderizar en consola la salida de ```pandas ```.
+ Biblioteca ```seaborn``` útil para mostrar la matriz de confusión. La anterior permite visualizar de mejor manera los aciertos y fallos, producto de la red neuronal. 
+ Herramienta ``Jupyter``, debe instalar Anaconda para poder hacer uso de esta herramienta.

## Nota importante
El sistema fue desarrollado en Windows 11. En caso de que el sistema sea ejecutado en otros sistemas operativos, por lo tanto, es posible que funciones pertenecientes a la biblioteca ```os``` deban ser reemplazadas por su homóloga respectiva.
