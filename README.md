# MOVIMIENTO DEL CURSOR MEDIANTE INTERFAZ CEREBRO COMPUTADOR HÍBRIDA

Secuencias de comando de la ICCH desarrollada 

# Archivos disponibles

## Modelo.py

Esta es la secuencias de comandos principal se puede diseñar una interfaz mediante el uso de la clase Modelo.

Para variar los parametros de la interfaz diseñada se hace uso del método Parametros().

Para entrenar o cargar la interfaz se hace uso del método Procesamiento(proceso), en el cual proceso corresponde al proceso que se desea realizar mediante la STR 'entrenar' o 'cargar'.

## Funciones.py

Aquí se encuentran todas las funciones desarrolladas que utiliza la clase Modelo.

## Evalucion.py

Imprime las graficas de evaluación utilizadas en el trabajo de grado

## Interfaz.py

Intefaz grafica de usuario que permite la selección del sujeto y el proceso, además de realizar el proceso selecionado con los datos del sujeto escogido.

despues de realizar la carga o el entrenamiento muestra la matriz de confusión del clasificador entrenado o cargado

debido a limitaciones de tiempo no se terminó varios botones no cumplen con ninguna funsión.

## Principal.kv

Contiene la información de la disposición de la interfaz grafica.

# directorios

## Parametros

Aquí se encuentran los datos de entrenamiento de todos los sujetos entrenados, además de los archivos Rendimiento.csv, y Evaluacion.csv que contien un resumen de las metricas obtenidas por los entrenamientos

## Iconos

Aquí se encuentran los iconos utilizados en la interfaz grafica de usuario.


# Generalidades

La ICCH fue desarrollada en python 3.9.6 con las siguientes librerias instaladas:
- Numpy
- scipy
- Scikit-learn
- Seaborn
- Tensorflow

para la interfaz grafica de usuario se utilizó:
- Kivy

Se comprobó que los algoritmos funcionan para python 3.10.9. 
Sin embargo, para el caso de python 3.11 varias librerias aun no han sido actualizadas para ejecutarse en esta versión.
