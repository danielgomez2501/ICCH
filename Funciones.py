"""
Funciones para la app

Created on Fri Jan 14 16:56:27 2022

version 0.3

@author: Daniel
"""

# ----------------------------------------------------------------------------
# Librerías

# general
from scipy.io import loadmat  # para  abrir archivos .mat
import numpy as np  # para operaciones matemáticas
import math  # para aproximaciones
import pandas as pd
import pickle  # para guardar los datos
import os  # interactuar con el sistema operativo
import matplotlib.pyplot as plt  # gráficas

# dividir la base de datos
from sklearn.model_selection import train_test_split
# from tkinter import filedialog #para cuadros de dialogo
from sklearn.utils import resample

# para filtro
from scipy import signal

# para el fastICA
from sklearn.decomposition import FastICA  # implementación de FastICA

# para la RNC
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
# Retrocompativilidad con antiguo
# from tensorflow.keras.optimizers.legacy import Adam

# Para matrices de confusión
from sklearn.metrics import confusion_matrix
from tensorflow.math import argmax  # para convertir de one hot a un vector
import seaborn as sns  # para el mapa de calor

# Para generar multiples procesos
import multiprocessing

#############################################################################
# ----------------------------------------------------------------------------
"""
Funciones para abrir datos, guardarlos, hacer el enventanado y dividir
los datos de test y prueba
"""


def NombreCanal(nombre):
    """
    Traduce los nombres estándar de EEG a los del dataset.

    Parameters
    ----------
    nombre: STR, correspondiente al nombre del canal según estándar
    internacional.
    Returns
    -------
    switch: STR, con el nombre del canal correspondiente en la base
    de datos.
    """

    # Se retorna el número del canal dependiendo del nombre de entrada
    switch = {
        'EMG_1': "ch1",
        'EMG_2': "ch2",
        'EMG_3': "ch3",
        'EMG_4': "ch4",
        'EMG_5': "ch5",
        'EMG_6': "ch6",
        'EMG_ref': "ch7",
        'FP1': "ch1",
        'AF7': "ch2",
        'AF3': "ch3",
        'AFz': "ch4",
        'F7': "ch5",
        'F5': "ch6",
        'F3': "ch7",
        'F1': "ch8",
        'Fz': "ch9",
        'FT7': "ch10",
        'FC5': "ch11",
        'FC3': "ch12",
        'FC1': "ch13",
        'T7': "ch14",
        'C5': "ch15",
        'C3': "ch16",
        'C1': "ch17",
        'Cz': "ch18",
        'TP7': "ch19",
        'CP5': "ch20",
        'CP3': "ch21",
        'CP1': "ch22",
        'CPz': "ch23",
        'P7': "ch24",
        'P5': "ch25",
        'P3': "ch26",
        'P1': "ch27",
        'Pz': "ch28",
        'PO7': "ch29",
        'PO3': "ch30",
        'POz': "ch31",
        'FP2': "ch32",
        'AF4': "ch33",
        'AF8': "ch34",
        'F2': "ch35",
        'F4': "ch36",
        'F6': "ch37",
        'F8': "ch38",
        'FC2': "ch39",
        'FC4': "ch40",
        'FC6': "ch41",
        'FT8': "ch42",
        'C2': "ch43",
        'C4': "ch44",
        'C6': "ch45",
        'T8': "ch46",
        'CP2': "ch47",
        'CP4': "ch48",
        'CP6': "ch49",
        'TP8': "ch50",
        'P2': "ch51",
        'P4': "ch52",
        'P6': "ch53",
        'P8': "ch54",
        'PO4': "ch55",
        'PO8': "ch56",
        'O1': "ch57",
        'Oz': "ch58",
        'O2': "ch59",
        'Iz': "ch60",
    }

    return switch.get(nombre, "Nombre Errado")


# ----------------------------------------------------------------------------
#


def NombresCanales(direccion_eeg, direccion_emg):
    """
    Saca los nombres de los canales en el dataset.

    Parameters
    ----------
    direccion_eeg: STRING, Dirección de los datos de EEG de la base de
        datos en formato .MAT.
    direccion_emg: STRING, Dirección de los datos de EEG de la base de 
        datos en formato .MAT.
    
    Returns
    -------
    nombres: DICCIONARIO, contiene los nombres correspondientes a los
        canales.
        nombres = 'Canales EEG': nombres_eeg
                  'Canales EMG': nombres_emg
    """

    # Cargar datos de las direcciones
    annots_emg = loadmat(direccion_emg)
    annots_eeg = loadmat(direccion_eeg)

    # para sacar los nombres de cada canal
    nombres_emg = [''] * (len(annots_emg['dat'][0][0][0][0]))

    # Sacar los nombres de los canales
    i = 0
    for nombre in annots_emg['dat'][0][0][0][0]:
        nombres_emg[i] = str(nombre[0])
        i += 1

    nombres_eeg = [''] * (len(annots_eeg['dat'][0][0][0][0]))

    i = 0
    for nombre in annots_eeg['dat'][0][0][0][0]:
        nombres_eeg[i] = str(nombre[0])
        i += 1
    del i

    nombres = {'Canales EEG': nombres_eeg,
               'Canales EMG': nombres_emg}
    return nombres


# ----------------------------------------------------------------------------
#


def TraducirNombresCanales(nombres):
    """
    Traduce una lista de nombres de canales estándar a los del dataset

    Parameters
    ----------
    nombres: LISTA, Contiene los nombres de los sensores usados de 
        acuerdo al estándar utilizado.
    
    Returns
    -------
    traduccion: LISTA, contiene los nombres correspondientes a los 
        de la base de datos
    """

    traduccion = [''] * (len(nombres))

    i = 0
    for canal in nombres:
        traduccion[i] = NombreCanal(canal)
        i += 1

    return traduccion


# ----------------------------------------------------------------------------
#


def GuardarPkl(datos, direccion, tipo='pkl'):
    """
    Guarda un objeto en la dirección en fomáto .pkl ó .obj.

    Parameters
    ----------
    datos: OBJETO, Contiene los datos a guardar.

    direccion: STR, el nombre del archivo a guardar, terminar en .pkl
        ó .obj

    tipo: STR, tipo de archivo a guardar, puede ser 'pkl' ó 'obj', con
        'pkl' como predeterminado.

    Returns
    -------

    """
    # revisa si la terminación es diferente a .pkl o .obj
    if (direccion[-4:] != '.obj') and (direccion[-4:] != '.pkl'):
        # concatena el tipo seleccionado
        direccion = direccion + '.' + tipo

    with open(direccion, 'wb') as file:
        pickle.dump(datos, file)
    print(direccion + ' guardado')

    pass


# ----------------------------------------------------------------------------
#


def AbrirPkl(direccion):
    """
    Abre un objeto .pkl de dada dirección.

    Parameters
    ----------
    direccion: STR, dirección del archivo a abrir, terminar en .pkl

    Returns
    -------
    datos: OBJETO, los datos del objeto cargado

    """

    with open(direccion, 'rb') as file:
        datos = pickle.load(file)

    return datos


#############################################################################
# ----------------------------------------------------------------------------
#
# Funciones para divergente
#


def DisenarFiltro(
        tipo_filtro, tipo_banda, orden_filtro, frecuencia_corte,
        frecuencia_muestreo):
    """
    Diseña un filtro con los parametros dados.

    Parameters
    ----------
    tipo_filtro: STRING, indíca el tipo de filtro a utilizar el tipo de
        filtro se es igual al que se utiliza en la función iirfilter de
        scipy

    tipo_banda: STRING, indica el tipo de banda de paso o rechazo a
        utilizar es igual al que se utiliza en la función iirfilter de
        scipy

    orden_filtro: INT, indica el orden del filtro a diseñar, solo se
        utiliza en algunos tipos de filtros

    frecuencia_corte: ARRAY, indica donde se encuentra la frecuencia a
        cortar puede ser la banda de paso

    frecuencia_muestreo: INT, indica el valor de frecuencia de muestreo (Hz)

    Returns
    -------
    filtro: SOS, filtro diseñado en formato SOS

    """

    wn = 2 * frecuencia_corte / frecuencia_muestreo

    filtro = signal.iirfilter(orden_filtro, wn, btype=tipo_banda,
                              analog=False, ftype=tipo_filtro, output='sos')

    return filtro


def ClasesOneHot(
        nombre_clases, num_clases, final_grabacion, banderas, one_hot):
    """
    Determina las clases One-Hot de cada una de las muestras.

    Parameters
    ----------
    nombre_clases: LIST: STR, lista con los nombres de las clases
    num_clases: INT, indica el número de clases totales
    final_grabacion: INT, indica el índice en el cual se tiene el
        final de la grabación del dataset
    banderas: ARRAY, contiene las banderas indican inicio y final de
        la ejecución de la actividad, disponible en el dataset
    one_hot: ARRAY, Matriz one-hot de todas las actividades,
        correspondiente al intervalo dado por las banderas

    Returns
    -------
    clases_OH: PD DATAFRAME, dataframe con las clases de los datos
        en formato OH, indicado para cada uno de los índices del dataset

    """

    # dataframe para las clases one-hot
    clases_oh = pd.DataFrame(columns=nombre_clases, dtype=np.int8)
    # arreglo para guardar si pertenece o no a la clase
    clase_verdad = np.zeros([final_grabacion + 1], dtype='int8')

    # para iterar entre las clases
    # i = 0
    for i in range(num_clases):
        # crear un vector con 0 y 1 de acuerdo a si corresponde o no a
        # la clase
        # reinicia el vector a ceros
        clase_verdad = clase_verdad * 0

        # index = 0
        for index in range(len(banderas) - 1):
            clase_verdad[banderas[index]:banderas[index + 1]] = one_hot[i, index]
            # index =+ 1

        if i >= num_clases - 1:
            clase_verdad[0:banderas[0]] = 1
            clase_verdad[banderas[-1]:] = 1

        clases_oh[nombre_clases[i]] = clase_verdad
        # i += 1
    return clases_oh


def AplicarFiltro(canales, filtro, annots):
    """
    Aplica un filtro SOS sobre los canales dados.

    Parameters
    ----------
    canales: LIST, lista con los nombres de los canales a utilizar

    filtro: SOS, filtro diseñado en formato SOS

    annots: DATASET, Contiene los datos del dataset ya sean los de
        EMG o EEG

    Returns
    -------
    senales_filt: DATAFRAME, datraframe que contiene las señales de
        la lista de canales filtradas

    """
    
    # Crear diccionario con llaves los canales
    senales_filt = dict.fromkeys(canales)
    for k in canales:
        senales_filt[k] = signal.sosfiltfilt(filtro, annots[k].T[0])
    
    # Metodo antigüo
    # # Señales en un dataframe gigante
    # senales_filt = pd.DataFrame(columns=canales)

    # # Señales de los canales seleccionados:
    # for k in canales:
    #     senales_filt[k] = signal.sosfilt(filtro, annots[k].T[0])

    return senales_filt


def SubmuestreoClases(
        senales, canales, clases, nombre_clases, inicio_grabacion, 
        final_grabacion, m, filtro):
    """
    Realiza el submuestreo en un intervalo dado.

    Parameters
    ----------
    num_canales: INT, índica el número de canales utilizados
    inicio_grabacion: INT, índica la muestras donde inicia la grabación
    final_grabacion: INT, indica la muestra donde termina la grabación
    m: INT, Factor de submuestreo, se da por la ecuación y(n)=x(Mn)
    senales: DATAFRAME, contiene las señales previamente filtradas
    canales:  LIST, lista con los nombres de los canales a utilizar

    Returns
    -------
    senales_subm: ARRAY, contiene las señales submuestreadas
    clases_m: ARRAY, contiene las clases de las señales submuestreadas.
    """
    
    # filtro_dlti = signal.sos2zpk(filtro)
    # filtro_dlti = signal.dlti(filtro_dlti[0], filtro_dlti[1], filtro_dlti[1])
    
    # el número de muestras que tendrá el vector final
    muestras = int((final_grabacion - inicio_grabacion)/m)
    
    senales_subm = dict.fromkeys(canales)
    for canal in canales:
        senales_subm[canal] = signal.resample(
            senales[canal][inicio_grabacion:final_grabacion], muestras,
            axis=0, window = 'hamming', domain='time')
    del senales   
        # senales_subm[canal] = signal.decimate(
        #     senales[canal][inicio_grabacion:final_grabacion], m, n=None, 
        #     ftype=filtro_dlti, axis=-1, zero_phase=True)
    
    # # Revisar que se calculen de forma correcta el numero de clases y
    # # el numero de canales
    # num_canales = np.shape(senales)[1]
    
    # # para las señales
    # # matriz vacia
    # senales_subm = np.zeros([
    #     num_canales, math.ceil((final_grabacion - inicio_grabacion) / m)
    # ])
    # for j in range(num_canales):
    #     senales_subm[j, :] = senales[canales[j]][
    #                          inicio_grabacion:final_grabacion:m
    #                          ]
    
    # para las clases
    # Calcular numero de clases
    num_clases = np.shape(clases)[1]    
    # matriz vacia
    clases_m = np.zeros([
        num_clases, math.ceil((final_grabacion - inicio_grabacion) / m)
    ])
    for j in range(num_clases):
        clases_m[j, :] = clases[nombre_clases[j]][
                             inicio_grabacion:final_grabacion:m
                             ]
    # combertir las clases en dataframe
    clases_subm = pd.DataFrame(
        clases_m.T, columns=nombre_clases, dtype='int8')

    return senales_subm, clases_subm


def HacerSubmuestreo(
        num_canales, inicio_grabacion, final_grabacion, m, senales_filt, canales):
    """
    Realiza el submuestreo en un intervalo dado.

    Parameters
    ----------
    num_canales: INT, índica el número de canales utilizados
    inicio_grabacion: INT, índica la muestras donde inicia la grabación
    final_grabacion: INT, indica la muestra donde termina la grabación
    m: INT, Factor de submuestreo, se da por la ecuación y(n)=x(Mn)
    senales_filt: DATAFRAME, contiene las señales previamente filtradas
    canales:  LIST, lista con los nombres de los canales a utilizar

    Returns
    -------
    senales_subm: ARRAY, contiene las señales submuestreadas

    """
    senales_subm = np.zeros([
        num_canales, math.ceil((final_grabacion - inicio_grabacion) / m)
    ])

    # para las señales
    for j in range(num_canales):
        senales_subm[j, :] = senales_filt[canales[j]][
                             inicio_grabacion:final_grabacion:m
                             ]

    return senales_subm


def HacerEnventanado(
        num_ventanas, num_canales, num_clases, tam_ventana, paso_ventana,
        paso_general, inicio_grabacion, senales_subm, clases_OH, sacar_clases):
    """
    Realiza el enventanado de las señales submuestreadas.

    Parameters
    ----------
    num_ventanas: INT, indica el numero de venatanas utilizadas

    num_canales: INT, indica el numero de canales utilizados

    num_clases: INT, indica el numero de clases

    tam_ventana: INT, indica el tamño de la ventana en numero de
        muestras

    paso_ventana: INT, indica el paso entre ventanas en numero de
        muestras

    paso_general: INT, indica el paso entre ventanas en numero de
        muestras de la frecuencia de muestreo original

    inicio_grabacion: INT, indica la muestra donde inicia la grabación
        del dataset

    senales_subm: ARRAY, matriz que contiene los datos de submuestreo

    clases_OH: DATAFRAME, contiene los datos de las clase a las que
        pertenencen cada una de las muestras

    sacar_clases: BOOL, True determina hacer una matriz con las clases
        de las ventanas enformato One-Hot.

    Returns
    -------
    ventanas: ARRAY, contiene las señales en ventanas

    clases_ventanas_OH: ARRAY, contiene las clases de las ventanas en
        formato One-Hot, solo lo retorna si sacar_clases es True

    """

    # Las ventanas en este caso en un arreglo de np
    ventanas = np.zeros((num_ventanas, num_canales, tam_ventana))

    if sacar_clases:

        # para tener las clase en formato OH de las ventanas
        clases_ventanas_oh = np.zeros((num_ventanas, num_clases), dtype='int8')

        # Variable para ver en cual ventana se va
        v = 0
        while v < num_ventanas:
            ventanas[v, :, :] = senales_subm[
                                :, paso_ventana * v:paso_ventana * v + tam_ventana
                                ]
            # la clase de la ventana es decidida por la clase a la que 
            # pertenece la primera muestra de la ventana
            clases_ventanas_oh[v, :] = clases_OH.iloc[
                                       int(v * paso_ventana), :
                                       ]
            # en la versión anterior se calculaba a partir de las señales sin 
            # submuestrear
            # clases_ventanas_oh[v, :] = clases_OH.iloc[
            #                            int(Finicio_grabacion + v * paso_general), :
            #                            ]
            v += 1

        return ventanas, clases_ventanas_oh

    else:
    # if ~sacar_clases:
        # Variable para ver en cual ventana se va
        v = 0
        while v < num_ventanas:
            ventanas[v, :, :] = senales_subm[
                                :, paso_ventana * v:paso_ventana * v + tam_ventana
                                ]
            v += 1

        return ventanas


def QuitarImaginacionMotora(datos, clases, clase_reposo, banderas, reclamador):
    """
    Eliminar las ventanas de imaginación motora para luego balancear la DB

    Parameters
    ----------
    datos: ARRAY, matriz que contiene los datos de las ventanas
    clases: ARRAY, matriz One-Hot correspondiente a las ventanas
    clase_reposo: ARRAY, vector con la clase en formato one-hot que
        causa el desbalanceo de los datos
    banderas: LIST, vector con las vanderas donde incian y terminan las
        activida
    reclamador: INT, indica el número de ventanas a tomar de la clase
        reposo por cada actividad, se calcula mediante:
        frecuencia muestreo * 3s (duración actividad) / paso ventana

    Returns
    ----------
    datos_sub: ARRAY, matriz que contiene los datos de las ventanas
        reducidos
    clases_sub: ARRAY, matriz que contiene las clases de las ventanas
        de los datos reducidos
    """
    # numero de ventanas totales
    num_ven = len(clases)
    # numero ventanas para la salida
    sub_ven = int(np.sum(np.sum(clases, axis=0) * ((clase_reposo - 1) * -1))
                  + reclamador * (1 + len(banderas) / 2))

    # donde se guardaran los datos
    datos_sub = np.zeros((sub_ven, np.shape(datos[0])[0], np.shape(datos[0])[1]))
    # donde se guardarán las clases
    clases_sub = np.zeros((sub_ven, np.shape(clases)[1]))

    # iniciadores
    clase_actual = np.array([69, 69])
    tomar = False
    monitor = 0
    i = 0
    j = 0
    # ciclo que recorre todos los datos
    while i < num_ven and j > sub_ven:
        # Revisa cambio de clase
        if not np.array_equal(clase_actual, clases[i]):
            clase_actual = clases[i]
            tomar = True
            monitor = 0
        # Revisa si debe tomar datos
        if not tomar:
            i = i + 1
        # Realiza la copia de datos
        if np.array_equal(clase_actual, clases[i]) and tomar:
            datos_sub[j] = datos[i]
            clases_sub[j] = clases[i]
            i = i + 1
            j = j + 1
            monitor = monitor + 1
        # Examina condiciones para tomar
        if (np.array_equal(clase_actual, clase_reposo)
                and (reclamador <= monitor) and tomar):
            tomar = False

    return datos_sub, clases_sub


def DescartarVentanas(
        datos, clases, clase_reposo, banderas, reclamador, descarte):
    """
    Eliminar las ventanas de senales ambiguas

    Las senales ambiguas, se las toma como las que suceden milisegundos
    despues de dar la pista visual antes de la respuesta como tal del
    usuario determinada por su tiempo de reaccion o el tiempo en que el
    cerebro logra reaccionar a dicha pista visual, otra parte es para
    el tiempo de reposo dado despues de una tarea el brazo del sujeto
    vuelve a la posicion inical y esto genera diferentes senales que
    son muy similares a las de las tareas como tal.
    mala clasificacion de las senales

    Parameters
    ----------
    datos: ARRAY, matriz que contiene los datos de las ventanas.
    clases: ARRAY, matriz One-Hot correspondiente a las ventanas.
    clase_reposo: ARRAY, vector con la clase en formato one-hot que
        causa el desbalanceo de los datos.
    banderas: LIST, vector con las vanderas donde incian y terminan las
        activida.
    reclamador: DICT, indica el numero de ventanas a tomar de la clase
        reposo por cada actividad, tine las siguientes llaves: 'Activo'
        y 'Reposo', se calcula mediante:
        frecuencia muestreo * (duracion actividad) / paso ventana
    descarte: DICT, indica el numero de ventanas a saltar para luego
        empesar a tomar los datos despues de una bandera, tine las
        siguientes llaves: 'Activo' y 'Reposo' en referencia al tiempo
        de espera para las senales de actividad de un movimiento o en
        el reposo, esto en numero de ventanas. se calcula mediante:
        frecuencia muestreo * (tiempo de respuesta) / paso ventana

    Returns
    ----------
    datos_sub: ARRAY, matriz que contiene los datos de las ventanas
        reducidos.
    clases_sub: ARRAY, matriz que contiene las clases de las ventanas
        de los datos reducidos.

    """
    # numero de ventanas totales
    num_ven = len(clases)

    datos_sub = []
    clases_sub = []

    # iniciadores
    # para la clase actual
    clase_actual = np.empty(0)
    # para determinar cuando tomar las ventanas
    tomar = False
    # para determinar si se siguen tomando los datos al compararlo
    monitor = 0
    # para saber si se está en la clase reposo
    reposo = False
    i = 0
    j = 0
    # ciclo que recorre todos los datos
    while i < (num_ven - 1):
        # Revisa cambio de clase
        if not np.array_equal(clase_actual, clases[i]):
            clase_actual = clases[i]
            tomar = True
            limite = False
            monitor = 0
            # ajuste para el salto por el tiempo de respuesta
            # revisar que sea la clase de reposo
            if np.array_equal(clase_actual, clase_reposo):
                i = i + descarte['Reposo']
                reposo = True
            else:
                i = i + descarte['Activo']
                reposo = False
        # Revisa si debe tomar datos
        if not tomar:
            i = i + 1
        # Realiza la copia de datos
        if i < num_ven:
            if np.array_equal(clase_actual, clases[i]) and tomar and not limite:
                # revisar si esto es lo suficiente mente rapido
                datos_sub.append(datos[i])
                clases_sub.append(clases[i])
                i = i + 1
                j = j + 1
                monitor = monitor + 1
                # Examina condiciones para tomar
                if reposo and (monitor >= reclamador['Reposo']):
                    tomar = False
                elif not reposo and (monitor >= reclamador['Activo']):
                    tomar = False

    datos_sub = np.array(datos_sub)
    clases_sub = np.array(clases_sub, dtype='int8')

    return datos_sub, clases_sub


def Balanceo(datos, clases, clase_reposo):
    """
    Balanceo de base de datos.

    Parameters
    ----------
    datos: ARRAY, matriz que contiene los datos de las ventanas

    clases: ARRAY, matriz One-Hot correspondiente de las clases de
        las ventanas

    clase_reposo: ARRAY, vector con la clase en formato one-hot que
        causa el desbalanceo de los datos

    Returns
    -------
    datos_sub: ARRAY, matriz que contiene los datos balanceados

    clases_sub: ARRAY, matriz que contiene las clases balanceadas

    """
    # ventanas totales
    num_ven = len(clases)

    # forma de las ventanas
    num_canales = np.shape(datos[0])[0]
    num_datos = np.shape(datos[0])[1]
    num_clases = np.shape(clases)[1]

    suma_clases = np.sum(clases, axis=0)

    # calculo del numero de ventanas de la clase que tiene menos ventanas
    num_ven_clase = int(np.min(suma_clases, axis=0))

    # calcula el numero de ventanas de actividad
    num_ven_actividad = int(np.sum(suma_clases * ((clase_reposo - 1) * -1)))
    # Calcula el numero de ventanas de reposo
    num_ven_reposo = int(suma_clases[np.argmax(clase_reposo)])

    # matrices donde se guardan las ventanas
    ven_actividad = np.zeros((num_ven_actividad, num_canales, num_datos))
    ven_reposo = np.zeros((num_ven_reposo, num_canales, num_datos))

    # matrices donde quedan las clases
    clases_actividad = np.zeros((num_ven_actividad, num_clases), dtype='int8')
    clases_reposo = np.ones((num_ven_reposo, num_clases), dtype='int8') * clase_reposo

    # separa los datos de reposo y actividad
    i = 0
    j = 0
    k = 0
    while i < num_ven:
        if np.array_equal(clases[i], clase_reposo):
            ven_reposo[j] = datos[i]
            j = j + 1
        else:
            ven_actividad[k] = datos[i]
            clases_actividad[k] = clases[i]
            k = k + 1
        i = i + 1

    ven_reposo, clases_reposo = resample(
        ven_reposo, clases_reposo, replace=False, n_samples=num_ven_clase,
        random_state=None, stratify=None)

    # concatenar los vectores de actividad e inactividad
    datos_sub = np.concatenate([ven_reposo, ven_actividad])
    clases_sub = np.concatenate([clases_reposo, clases_actividad])

    # mezclar de forma aleatorea los datos
    datos_sub, clases_sub = resample(
        datos_sub, clases_sub, replace=False, n_samples=None,
        random_state=None, stratify=None)

    return datos_sub, clases_sub


def BalanceDoble(datos_a, datos_b, clases, clase_reposo):
    """
    Balanceo de base de datos para los dos tipos de datos

    Parameters
    ----------
    datos_a: ARRAY, matriz que contiene los datos a de las ventanas 
    
    datos_b: ARRAY, matriz que contiene los datos b de las ventanas
    
    clases: Dataframe, matriz One-Hot correspondiente de las clases de
        las ventanas
    
    clase_reposo: ARRAY, vector con la clase en formato one-hot que
        causa el desbalanceo de los datos
    
    Returns
    -------
    datos_a_sub: ARRAY, matriz que contiene los datos a balanceados
    
    datos_b_sub: ARRAY, matriz que contiene los datos a balanceados
    
    clases_sub: ARRAY, matriz que contiene las clases balanceadas

    """
    # ventanas totales
    num_ven = len(clases)

    suma_clases = np.sum(clases, axis=0)

    # calculo del numero de ventanas de la clase que tiene menos ventanas
    num_ven_clase = int(np.min(suma_clases, axis=0))

    # calcula el numero de ventanas de actividad
    num_ven_actividad = int(np.sum(suma_clases * ((clase_reposo - 1) * -1)))
    # Calcula el numero de ventanas de reposo
    num_ven_reposo = int(suma_clases[np.argmax(clase_reposo)])

    indices_reposo = np.zeros((num_ven_reposo), dtype='int')
    indices_actividad = np.zeros((num_ven_actividad), dtype='int')

    # separa los datos de reposo y actividad
    i = 0
    j = 0
    k = 0
    while i < num_ven:
        if np.array_equal(clases[i], clase_reposo):
            indices_reposo[j] = int(i)
            j = j + 1
        else:
            indices_actividad[k] = int(i)
            k = k + 1
        i = i + 1

    indices_reposo = resample(
        indices_reposo, replace=False, n_samples=num_ven_clase,
        random_state=None, stratify=None)

    # concatenar los vectores de inactividad y actividad
    indices = np.concatenate([indices_reposo, indices_actividad])

    # mezclar de forma aleatorea los datos
    indices = resample(
        indices, replace=False, n_samples=None,
        random_state=None, stratify=None)

    datos_a_sub = datos_a[indices]
    datos_b_sub = datos_b[indices]
    clases_sub = clases[indices]

    return datos_a_sub, datos_b_sub, clases_sub


def TransformarICA(X, whiten, num_ventanas, num_ci, tam_ventana):
    """
    Realiza la transformación en de ICA.

    Esta transformación se la realiza con una matriz de tranformación 
    para blanqueo.
    
    Parameters
    ----------
    X: ARRAY, contiene las ventanas con las señales a calcular los 
        CI.
        
    whiten: ARRAY, con matriz de transformación para blanqueo de los
        datos.
    
    num_ventanas: INT, indica el numero de ventanas utilizadas para
        el entrenamiento

    num_ci: INT, indica el numero de componentes independientes a
        calcular

    tam_ventana: INT, indica el numero de muestras en cada ventana
    

    Returns
    -------
    x_ica: ARRAY, arreglo con los componentes independientes calculados

    """
    # para una transformación individual de las matrices
    x_white = np.zeros(
        (num_ventanas, num_ci, tam_ventana))

    # aplicación del blanqueo
    # para usarla x_blanca = np.dot(k,x.T)
    for i in range(num_ventanas):
        x_white[i] = np.dot(whiten, X[i])

    # ya que el whitening es falso se ignora el n_components
    ica = FastICA(
        algorithm='parallel', whiten=False, fun='exp', max_iter=500)

    x_ica = np.zeros(
        (num_ventanas, num_ci, tam_ventana))

    for i in range(num_ventanas):
        x_ica[i] = ica.fit_transform(x_white[i].T).T

    return x_ica


def AplicarICA(num_ventanas, num_ci, tam_ventana, ica, ventanas):
    """
    Realiza la transformación en de ICA.

    Parameters
    ----------
    num_ventanas: INT, indica el numero de ventanas utilizadas para
        el entrenamiento

    num_ci: INT, indica el numero de componentes independientes a
        calcular

    tam_ventana: INT, indica el numero de muestras en cada ventana

    ica: FASTICAMODEL, modelo utilizado para la transformación de 
        FastICA

    ventanas: ARRAY, contiene las ventanas con las señales a calcular
        los CI

    Returns
    -------
    ci: ARRAY, arreglo con los componentes independientes calculados

    """
    
    ci = np.zeros((num_ventanas, num_ci, tam_ventana))

    n = 0
    while n < num_ventanas:
        ci[n, :, :] = ica.transform(ventanas[n, :, :].T).T
        n += 1

    return ci



def ClasificadorEMG(num_ci, tam_ventana, num_clases):
    """
    Extructura de RNC para EMG.

    Parameters
    ----------
    num_ci: INT, indica el numero de componentes independientes
    que se utilizarán como entrada en la red

    tam_ventana: INT, indica el numero de muestras en cada ventana
    
    num_clases: INT, indica el numero de clases a clasificar, siendo
    tambien el numero de neuronas en la capa de salida

    Returns
    -------
    modelo_emg: ESTRUCTURA DE RNA, la extructura secuencial de la
    RNA sin entrenar.

    """
    # la red para EMG fue modificada dado que se contaban con entrada
    # de 4 x 325

    # Diseño de RNA convolucional
    modelo = Sequential()
    # primera capa, convoluciòn temporal
    modelo.add(
        Conv1D(10, 10, activation='relu', padding='valid', strides=1,
               input_shape=(num_ci, tam_ventana, 1)))
    # modelo_emg.add(BatchNormalization())
    # modelo_emg.add(Dropout(0.25))
    # segunda capa
    # modelo_emg.add(MaxPooling2D(pool_size=(2, 2)))
    # tercera capa, convoluciòn 
    modelo.add(
        Conv2D(20, (num_ci, 1), activation='relu', padding='valid', strides=(1, 1)))
    modelo.add(BatchNormalization())
    # modelo_emg.add(Dropout(0.50))
    # cuarta capa, terminan las convolucionales por lo cual se aplana todo
    # modelo_emg.add(Activation(tf.math.square))
    modelo.add(AveragePooling2D(pool_size=(1, 30), strides=(1, 15)))
    # modelo_emg.add(Activation(tf.math.log))
    modelo.add(Dropout(0.50))
    # Capa de convolución global
    modelo.add(
        Conv1D(1, 25, activation='relu', padding='valid', strides=1))
    # Terminan las capas convolucionales
    modelo.add(Flatten())
    # quinta capa
    modelo.add(Dense(16, activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.25))

    # sexta capa
    modelo.add(Dense(num_clases, activation='softmax'))

    # Se usa categorical por que son varias clases.
    # Loss mediante entropia cruzada.
    # Las metricas son las que se muestran durante el FIT pero no
    # afectan el entrenamiento.
    modelo.compile(
        optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy',
        metrics=[
            'accuracy', 'categorical_accuracy', 'categorical_crossentropy'
        ])
    
    """ ANTIGUA Estructura, modificada con nuevas ideas.
    # la red para EMG fue modificada dado que se contaban con entrada
    # de 4 x 325

    # Diseño de RNA convolucional
    modelo_emg = Sequential()
    # primera capa
    modelo_emg.add(
        Conv1D(8, 13, activation='relu', padding='valid', strides=1,
               input_shape=(num_ci, tam_ventana, 1)))
    modelo_emg.add(BatchNormalization())
    modelo_emg.add(Dropout(0.25))
    # segunda capa
    modelo_emg.add(AveragePooling2D(pool_size=(1, 20), strides = (1, 10)))
    # tercera capa
    modelo_emg.add(
        Conv2D(8, (num_ci, 1), activation='relu', padding='valid', strides=(1, 1)))
    modelo_emg.add(BatchNormalization())
    modelo_emg.add(Dropout(0.50))
    # cuarta capa, terminan las convolucionales por lo cual se aplana todo
    modelo_emg.add(MaxPooling2D(pool_size=(1, 10), strides = (1, 5)))
    # Terminan las capas convolucionales
    modelo_emg.add(Flatten())
    # quinta capa
    modelo_emg.add(Dense(8, activation='relu'))
    modelo_emg.add(BatchNormalization())
    modelo_emg.add(Dropout(0.50))
    # sexta capa
    modelo_emg.add(Dense(num_clases, activation='softmax'))

    # Se usa categorical por que son varias clases.
    # Loss mediante entropia cruzada.
    # Las metricas son las que se muestran durante el FIT pero no
    # afectan el entrenamiento.
    modelo_emg.compile(
        optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy',
        metrics=[
            'accuracy', 'categorical_accuracy', 'categorical_crossentropy'
        ])
    """

    return modelo


def ClasificadorEEG(num_ci, tam_ventana, num_clases):
    """
    Extructura RNC para EEG

    Parameters
    ----------
    num_ci: INT, indica el numero de componentes independientes
    que se utilizarán como entrada en la red

    tam_ventana: INT, indica el numero de muestras en cada ventana
    
    num_clases: INT, indica el numero de clases a clasificar, siendo
    tambien el numero de neuronas en la capa de salida

    Returns
    -------
    modelo_eeg: ESTRUCTURA DE RNA, la extructura secuencial de la
    RNA sin entrenar.

    """
    # Diseño de RNA convolucional
    modelo = Sequential()
    # primera capa, convoluciòn temporal
    modelo.add(
        Conv1D(8, 16, activation='relu', padding='valid', strides=1,
               input_shape=(num_ci, tam_ventana, 1)))
    # modelo_emg.add(BatchNormalization())
    # modelo_emg.add(Dropout(0.25))
    # segunda capa
    # modelo_emg.add(MaxPooling2D(pool_size=(2, 2)))
    # tercera capa, convoluciòn 
    modelo.add(
        Conv2D(8, (num_ci, 1), activation='relu', padding='valid', strides=(1, 1)))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.50))
    # cuarta capa, terminan las convolucionales por lo cual se aplana todo
    # modelo_emg.add(Activation(tf.math.square))
    modelo.add(AveragePooling2D(pool_size=(1, 16), strides=(1, 8)))
    # modelo_emg.add(Activation(tf.math.log))
    #modelo.add(Dropout(0.50))
    # Capa de convolución global
    modelo.add(
        Conv1D(1, 16, activation='relu', padding='valid', strides=1))
    # Terminan las capas convolucionales
    modelo.add(Flatten())
    # quinta capa
    modelo.add(Dense(8, activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.25))

    # sexta capa
    modelo.add(Dense(num_clases, activation='softmax'))

    # Se usa categorical por que son varias clases.
    # Loss mediante entropia cruzada.
    # Las metricas son las que se muestran durante el FIT pero no
    # afectan el entrenamiento.
    modelo.compile(
        optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy',
        metrics=[
            'accuracy', 'categorical_accuracy', 'categorical_crossentropy'
        ])
    
    """
    # Variación adaptada
    # Diseño de RNA convolucional
    modelo_eeg = Sequential()
    # primera capa
    modelo_eeg.add(
        Conv1D(8, 16, activation='relu', padding='same', strides=1,
               input_shape=(num_ci, tam_ventana, 1)))
    modelo_eeg.add(BatchNormalization())
    modelo_eeg.add(Dropout(0.25))
    # segunda capa
    modelo_eeg.add(MaxPooling2D(pool_size=(1, 16), strides = (1, 8)))
    # tercera capa
    modelo_eeg.add(
        Conv2D(16, (num_ci, 1), activation='relu', padding='valid', strides=1))
    modelo_eeg.add(BatchNormalization())
    modelo_eeg.add(Dropout(0.50))
    # cuarta capa, terminan las convolucionales donde se aplana todo
    modelo_eeg.add(MaxPooling2D(pool_size=(1, 16), strides = (1, 8)))
    # Terminan las capas convolucionales
    modelo_eeg.add(Flatten())
    # quinta capa
    modelo_eeg.add(Dense(8, activation='relu'))
    modelo_eeg.add(BatchNormalization())
    modelo_eeg.add(Dropout(0.50))
    # sexta capa
    modelo_eeg.add(Dense(num_clases, activation='softmax'))

    # Se usa categorical por que son varias clases.
    # Loss mediante entropia cruzada.
    # Las metricas son las que se muestran durante el FIT pero no
    # afectan el entrenamiento.
    modelo_eeg.compile(
        optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy',
        metrics=[
            'accuracy', 'categorical_accuracy', 'categorical_crossentropy'
        ])
    # modelo_eeg.summary()
    
    # Antiguo
    # Diseño de RNA convolucional
    modelo_eeg = Sequential()
    # primera capa
    modelo_eeg.add(
        Conv1D(4, 13, activation='relu', padding='same', strides=1,
               input_shape=(num_ci, tam_ventana, 1)))
    modelo_eeg.add(BatchNormalization())
    modelo_eeg.add(Dropout(0.25))
    # segunda capa
    modelo_eeg.add(MaxPooling2D(pool_size=(1, 20), strides = (1, 10)))
    # tercera capa
    modelo_eeg.add(
        Conv2D(8, (13, 3), activation='relu', padding='valid', strides=1))
    modelo_eeg.add(BatchNormalization())
    modelo_eeg.add(Dropout(0.50))
    # cuarta capa, terminan las convolucionales donde se aplana todo
    modelo_eeg.add(MaxPooling2D(pool_size=(1, 10), strides = (1, 5)))
    # Terminan las capas convolucionales
    modelo_eeg.add(Flatten())
    # quinta capa
    modelo_eeg.add(Dense(8, activation='relu'))
    modelo_eeg.add(BatchNormalization())
    modelo_eeg.add(Dropout(0.50))
    # sexta capa
    modelo_eeg.add(Dense(num_clases, activation='softmax'))

    # Se usa categorical por que son varias clases.
    # Loss mediante entropia cruzada.
    # Las metricas son las que se muestran durante el FIT pero no
    # afectan el entrenamiento.
    modelo_eeg.compile(
        optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy',
        metrics=[
            'accuracy', 'categorical_accuracy', 'categorical_crossentropy'
        ])
    """
    return modelo

def ClasificadorCanales(num_ci, tam_ventana, num_clases):
    """
    Extructura RNC para EEG

    Parameters
    ----------
    num_ci: INT, indica el numero de componentes independientes
    que se utilizarán como entrada en la red

    tam_ventana: INT, indica el numero de muestras en cada ventana
    
    num_clases: INT, indica el numero de clases a clasificar, siendo
    tambien el numero de neuronas en la capa de salida

    Returns
    -------
    modelo_eeg: ESTRUCTURA DE RNA, la extructura secuencial de la
    RNA sin entrenar.

    """
    # # desactivar el uso de GPU (no hay suficiente memoria de GPU para entrenar)
    # try:
    #     # Disable all GPUS
    #     tf.config.set_visible_devices([], 'GPU')
    #     visible_devices = tf.config.get_visible_devices()
    #     for device in visible_devices:
    #         assert device.device_type != 'GPU'
    # except:
    #     # Invalid device or cannot modify virtual devices once initialized.
    #     print('No se pudo desactivar la GPU')
    #     pass
    
    # Diseño de RNA convolucional
    modelo = Sequential()
    # primera capa, convoluciòn temporal
    modelo.add(
        Conv1D(8, 16, activation='relu', padding='valid', strides=1,
               input_shape=(num_ci, tam_ventana, 1)))
    # modelo.add(BatchNormalization())
    # modelo.add(Dropout(0.25))
    # segunda capa
    # modelo.add(MaxPooling2D(pool_size=(2, 2)))
    # segunda capa, convoluciòn 
    modelo.add(
        Conv2D(16, (num_ci, 1), activation='relu', padding='valid', strides=(1, 1)))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.50))
    # tercera capa, terminan las convolucionales por lo cual se aplana todo
    modelo.add(AveragePooling2D(pool_size=(1, 16), strides=(1, 8)))
    #modelo.add(Dropout(0.50))
    # Capa de convolución global
    modelo.add(
        Conv1D(1, 16, activation='relu', padding='valid', strides=1))
    # Terminan las capas convolucionales
    modelo.add(Flatten())
    # quinta capa
    modelo.add(Dense(8, activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(Dropout(0.25))

    # sexta capa
    modelo.add(Dense(num_clases, activation='softmax'))

    # Se usa categorical por que son varias clases.
    # Loss mediante entropia cruzada.
    # Las metricas son las que se muestran durante el FIT pero no
    # afectan el entrenamiento.
    modelo.compile(
        optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy',
        metrics=['categorical_accuracy'])
    
    return modelo
    
#############################################################################
# ----------------------------------------------------------------------------
#
# Funciones para 0.61 en adelante
#

def ExtraerDatos(directorio, sujeto, tipo):
    """Se extraen los datos del dataset

    Parameters
    ----------
    directorio: STRING, Dirección del directorio donde se encuentran
        las bases de datos.
    sujeto: INT, Numero del sujeto al cual sacar los datos.
    tipo: STRING, Tipo de señales, ya sea "EEG" o "EMG" en mayusculas.

    Returns
    -------
    datos: DICCIONARIO, contiene los datos del sujeto elejido, cada una
        de las llaves tiene una lista con tres datos que corresponden a
        cada una de las seciones de entrenamiento.
        datos = 'Inicio grabacion': inicio_grabacion
                'Final grabacion': final_grabacion
                'Frecuencia muestreo': frec_muestreo
                'Banderas': banderas
                'One hot': one_hot
    """

    # listas con los datos
    inicio_grabacion = []
    final_grabacion = []
    banderas = []
    one_hot = []

    for sesion in range(1, 4):
        # dirreccion
        direccion = directorio + '/Subjet_' + str(sujeto) + '/' + tipo + '_session' + str(sesion) + '_sub' + str(
            sujeto) + '_reaching_realMove.mat'
        # Los datos
        annots = loadmat(direccion)
        # Incio de la grabación
        inicio_grabacion.append(annots['mrk'][0][0][5][0][0][0][0][0])
        # Final de la grabación
        final_grabacion.append(annots['mrk'][0][0][5][0][0][0][0][1])
        # Banderas que indican incio y final de la ejecución de la actividad
        banderas.append(annots['mrk'][0][0][0][0])
        # Matriz one-hot de todas las actividades, correspondiente al intervalo dado por las banderas.
        one_hot.append(annots['mrk'][0][0][3])

    # Frecuencia de muestreo 2500 Hz
    frec_muestreo = annots['mrk'][0][0][2][0][0]

    # Los datos
    datos = {'Inicio grabacion': inicio_grabacion,
             'Final grabacion': final_grabacion,
             'Frecuencia muestreo': frec_muestreo,
             'Banderas': banderas,
             'One Hot': one_hot}

    return datos


def Submuestreo(
        directorio, tipo, datos, sujeto, sesion, canales, nombre_clases,
        filtro, m):
    """Se realisa submuestreo y aplica filtro

    Cabe destacar que esta función no realiza interpolación de los
    datos a la hora de hacer el submuestreo.

    Parameters
    ----------
    directorio: STRING, Dirección del directorio donde se encuentran
        las bases de datos.
    tipo: STRING, Tipo de señales, ya sea "EEG" o "EMG" en mayusculas.
    datos: DICCIONARIO, contiene los datos del sujeto elejido, cada una
        de las llaves tiene una lista con tres datos que corresponden a
        cada una de las seciones de entrenamiento.
        datos = 'Inicio grabacion': inicio_grabacion
                'Final grabacion': final_grabacion
                'Frecuencia muestreo': frec_muestreo
                'Banderas': banderas
                'One hot': one_hot
    sujeto: INT, Numero del sujeto al cual sacar los datos.
    sesion: INT, numero de la sesión a la cual hacer el submuestreo, de
        1 a 3.
    canales: LISTA, contiene los nombres de los canales.
    clases: ARRAY, contiene las clases a las que pertenecen cada uno de
        los datos.
    nombre_clases: LISTA, contiene los nombres de las clases.
    filtro: SOS, filtro diseñado en formato SOS.
    m: INT, Factor de submuestreo, se da por la ecuación y(n)=x(Mn)

    Returns
    -------
    senales_subm: ARRAY, contiene las señales submuestreadas

    """
    # dirreccion
    direccion = directorio + '/Subjet_' + str(sujeto) + '/' + tipo + '_session' + str(sesion) + '_sub' + str(
        sujeto) + '_reaching_realMove.mat'
    # Los datos
    annots = loadmat(direccion)
    # variables a calcular
    # num_canales = len(canales)
    sesion = int(sesion) - 1  # ya que el indice comienza con cero

    # Determinar clases
    # Tomar la clase de onehot y asignarla a la clases oh de forma que cada
    # indice corresponda con las banderas. 
    # Dataframe para las clases one-hot
    clases = ClasesOneHot(
                nombre_clases, len(nombre_clases),
                datos['Final grabacion'][sesion], datos['Banderas'][sesion],
                datos['One Hot'][sesion])

    senales_filt = AplicarFiltro(canales, filtro, annots)
    
    # Aplicar el filtro a los datos y guradarlo en el data frame
    # print('Aplicando filtros a las señales ...')
    if m == 1:
        # para las señales
        senales_subm = dict.fromkeys(canales)
        for k in canales:    
            senales_subm[k] = senales_filt[k][
                datos['Inicio grabacion'][sesion]:datos['Final grabacion'][sesion]]
        
        # para las clases
        # matriz vacia
        clases_m = np.zeros([
            np.shape(clases)[1], math.ceil(
                datos['Final grabacion'][sesion] - datos['Inicio grabacion'][sesion])
        ])
        for j in range(np.shape(clases)[1]):
            clases_m[j, :] = clases[nombre_clases[j]][
                                 datos['Inicio grabacion'][sesion]:datos['Final grabacion'][sesion]
                                 ]
        # combertir las clases en dataframe
        clases_subm = pd.DataFrame(
            clases_m.T, columns=nombre_clases, dtype='int8')
        
    else:
        # Sub muestreo
        # Ecuación para el sub muestreo: y(n)=x(Mn)
        # Calcular la frecuencia de sub muestreo
        # frec_submuestreo = int(datos['Frecuencia muestreo'] / m)
        # Variable donde guardar el submuestreo
        # senales_subm = HacerSubmuestreo(
        #    num_canales, datos['Inicio grabacion'][sesion],
        #    datos['Final grabacion'][sesion], m, senales_filt, canales)
        # senales = dict.fromkeys(canales)
        # for canal in canales:
        #     senales[canal] = annots[canal].T[0] 
        # se aplica submuestreo a las clases tambien.
        senales_subm, clases_subm = SubmuestreoClases(
            senales_filt, canales, clases, nombre_clases, datos['Inicio grabacion'][sesion],
            datos['Final grabacion'][sesion], m, filtro)
    
    return senales_subm, clases_subm


def Enventanado(
        senales, clases_OH, datos, sesion, tam_ventana_ms, paso_ms,
        frec_submuestreo, num_canales, num_clases):
    """Se realisa el enventanado de las señales

    Parameters
    ----------
    senales: ARRAY, matriz que contiene los datos de las señales.
    clases_OH: DATAFRAME, contiene los datos de las clase a las que
        pertenencen cada una de las muestras.
    datos: DICCIONARIO, contiene los datos del sujeto elejido, cada una
        de las llaves tiene una lista con tres datos que corresponden a
        cada una de las seciones de entrenamiento.
        datos = 'Inicio grabacion': inicio_grabacion
                'Final grabacion': final_grabacion
                'Frecuencia muestreo': frec_muestreo
                'Banderas': banderas
                'One hot': one_hot
    sesion: INT, numero de la sesión a la cual aplicar el enventanado,
        de 0 a 2.
    tam_ventana_ms: INT, tamaño en ms de las ventanas.
    paso_ms: INT, distancia en ms entre las ventanas.
    frec_submuestreo: INT, el valor de la frecuencia a la cual los
        datos fueron submuestreados en Hz.
    num_canales: INT, valor de el numero de canales a usar
    num_clases: INT, valor del nuemro de clases a usar

    Returns
    -------
    ventanas: ARRAY, contiene las señales en ventanas.

    clases_ventanas: ARRAY, contiene las clases de las ventanas en
        formato One-Hot, solo lo retorna si sacar_clases es True.

    """

    # Paso de ventanas para la frecencia de muestreo original
    paso_ventana_general = int(
        paso_ms * 0.001 * datos['Frecuencia muestreo'])
    # Variable para calcular el numero de ventanas totales
    # se calcula con la ultima vandera ya que hay unos segundos de 
    # incatividad hasta el final de la pruebas
    num_ventanas = int(((datos['Banderas'][sesion][-1]
                         - datos['Inicio grabacion'][sesion])
                        / (datos['Frecuencia muestreo'] * paso_ms)) * 1000)

    # Para determinar el tamaño de las ventanas
    tam_ventana = int(tam_ventana_ms * 0.001 * frec_submuestreo)
    # Para el tamaño del paso de la ventana en muestras segundo
    paso_ventana = int(paso_ms * 0.001 * frec_submuestreo)
    # Las ventanas en este caso en un arreglo de np

    ventanas, clases_ventanas = HacerEnventanado(
        num_ventanas, num_canales, num_clases, tam_ventana,
        paso_ventana, paso_ventana_general,
        datos['Inicio grabacion'][sesion], senales, clases_OH,
        sacar_clases=True)

    return ventanas, clases_ventanas


def Division(
        ventanas, clases_ventanas, porcen_prueba, porcen_validacion,
        num_clases, todasclases=False):
    """Se realisa la divición del dataset junto al balanceo de los datos

    Parameters
    ----------
    ventanas: ARRAY, contiene las señales en ventanas
    clases_ventanas: ARRAY, contiene las clases de las ventanas en 
        formato One-Hot
    porcen_prueba: INT, Porcentaje a dividir para datos de prueba
    porcen_validacion: INT, porcentaje a dividir para datos de 
        validación.
    num_clases: INT, numero de clases usadas.
    todasclases: BOOL, indica si el balanceo de los datos se debe 
        aplicar a todas las clases, en el caso de False unicamente se 
        plica el balanceo a la clase de reposo.
    
    Returns
    -------
    train: ARRAY, matriz que contiene las ventanas balanceadas de 
        entrenamiento
    class_train: ARRAY, matriz que contiene las clases de las ventanas 
        balanceadas de entrenamiento 
    validation: ARRAY, matriz que contiene las ventanas balanceadas de 
        validación
    class_validation: ARRAY, matriz que contiene las clases de las 
        ventanas balanceadas de validación
    test: ARRAY, matriz que contiene las ventanas balanceadas de prueba
    class_test: ARRAY, matriz que contiene las clases de las ventanas
        balanceadas de prueba
    
    """
    # Divición entrenamiento y prueba
    train_un, test_un, class_train_un, class_test_un = train_test_split(
        ventanas, clases_ventanas, test_size=porcen_prueba,
        random_state=1, shuffle=False)
    # Divición entrenamiento y validación
    train_un, validation_un, class_train_un, class_validation_un = train_test_split(
        train_un, class_train_un, test_size=porcen_validacion,
        random_state=1, shuffle=True)

    # Balanceo de base de datos mediate submuestreo aleatoreo
    # aplicado unicamente a datos de entrenamiento

    # La inicialización del balance se hace para conservar las 
    # variables anteriores y poder compararlas
    clases = np.identity(num_clases, dtype='int8')
    # inicialización
    train, class_train = Balanceo(
        train_un, class_train_un, clases[-1])
    validation, class_validation = Balanceo(
        validation_un, class_validation_un, clases[-1])
    test, class_test = Balanceo(
        test_un, class_test_un, clases[-1])

    # En el caso de que se requiera realizar en todas las clases
    if todasclases:
        for i in range(num_clases - 1):
            train, class_train = Balanceo(
                train, class_train, clases[i])
            validation, class_validation = Balanceo(
                validation, class_validation, clases[i])
            test, class_test = Balanceo(
                test, class_test, clases[i])

    return train, class_train, validation, class_validation, test, class_test


def SelecionarCanales(
        rendimiento, direccion, tipo='EMG', determinar=False, num_canales=3):
    """Selecion automatica de canales mediante XCDC

    Este metodo de selección de momento se plantea que sea lo del
    articulo "Cross-Correlation Based Discriminant Criterion for
    Channel Selection in Motor Imagery BCI Systems" 

    Parameters
    ----------
    registros: DICT, continen los registros de las señales.
    determinar : BOOL, opcinal, en el caso de que sea True se 
    implementa el metodo de selección de canales, cuando es False los
        canales selecionados se toman de la lista de 
        seleccion_canales. Pred: False.
    selecion_canales : LIST, Contienen los nombres de los canales a 
        seleccionar. Pred: None.

    Returns
    -------
    registros_train: DICT, contiene las señales de los canales 
        seleccionadas.
        
    selecion_canales : LIST, solo es retornado si determinar = True, 
        contiene los canales seleccionado por el metodo de selecciòn
        de caracteristicas empleado.

    """
    # sacar la lista de canales disponibles
    canales = rendimiento.keys()
    
    # revisar que el numero de canales elegidos esté disponible
    if num_canales > len(canales) or num_canales <= 0 or type(num_canales) != int:
        print('El número de canales escogidos es incorrecto.')
        print('Automaticamente el número de canales escogidos pasa a ser tres.')
        num_canales = 3

        # convertir a pandas
    if determinar:
        # crear dataframes con los canales como indice
        evaluacion = pd.DataFrame(index=canales,columns=['categorical_accuracy', 'loss'])
        # agregar la información del rendimiento en los dataframes
        for canal in canales:
            lista_exactitud = [valor['categorical_accuracy'] for valor in rendimiento[canal]]
            lista_perdida = [valor['loss'] for valor in rendimiento[canal]]
            # evaluar el rendimiento de cada canal
            evaluacion['categorical_accuracy'][canal] = np.mean(lista_exactitud)
            evaluacion['loss'][canal] = np.mean(lista_perdida)
        
        # Guardar la evalucion del rendimiento
        evaluacion.to_csv(direccion + 'evaluacion_' + tipo)
    
    else:
        evaluacion = pd.read_csv(direccion + 'evaluacion_' + tipo)
    
    # ordenar los canales de menor a mayor
    seleccion_canales = evaluacion.sort_values(
        by=['categorical_accuracy'], ascending=False).index.tolist()

    return seleccion_canales[:num_canales]

def ICA(
        train, validation, test, senales_subm, num_ci, tam_ventana,
        paso_ventana):
    """Se realisa la divición del dataset junto al balanceo de los datos

    Parameters
    ----------
    train: ARRAY, matriz que contiene las ventanas balanceadas de 
        entrenamiento.
    validation: ARRAY, matriz que contiene las ventanas balanceadas de 
        validación.
    test: ARRAY, matriz que contiene las ventanas balanceadas de prueba.
    senales_subm: ARRAY, matriz que contiene los datos de submuestreo.
    num_ci: INT, contiene el numero de componentes independientes a 
        calcular, siempre es mayor que 4.
    tam_ventana: INT, indica el tamaño de ventana en numero de muestras.
    paso_ventana: INT, indica el paso entre ventanas en numero de
        muestras.
    
    Returns
    -------
    train_ica: ARRAY, matriz que contiene las ventanas de entrenamiento
        aplicadas la transformación ICA.
    validation_ica: ARRAY, matriz que contiene las ventanas de
        validacion aplicadas la transformación ICA.
    test_ica: ARRAY, matriz que contiene las ventanas de prueba
        aplicadas la transformación ICA.
    ica_total: OBJ, objeto para la transformar ica ya entrenado.
    whiten: ARRAY, matriz de transformación para blanqueamiento.
    
    """
    # Revisar que funcione correctamente la idea es determinar el 
    # numero de ventanas
    num_ventanas_entrenamiento = len(train)
    num_ventanas_validacion = len(validation)
    num_ventanas_prueba = len(test)

    # Calcular el ultimo valor de entrenamiento
    lim_entre = paso_ventana * num_ventanas_prueba + tam_ventana

    # para el vector de todos los datos de entrenamiento
    ica_total = FastICA(
        n_components=num_ci, algorithm='parallel',
        whiten='arbitrary-variance', fun='exp', max_iter=500)
    ica_total.fit(senales_subm[:, :lim_entre].T)
    # obtener la matriz de blanqueo
    whiten = ica_total.whitening_
    # aplicar transformaciones
    train_ica = TransformarICA(
        train, whiten, num_ventanas_entrenamiento,
        num_ci, tam_ventana)
    validation_ica = TransformarICA(
        validation, whiten, num_ventanas_validacion,
        num_ci, tam_ventana)
    test_ica = TransformarICA(
        test, whiten, num_ventanas_prueba,
        num_ci, tam_ventana)

    return train_ica, validation_ica, test_ica, ica_total, whiten


def NICA(
        train, validation, test, num_ci, tam_ventana,
        paso_ventana):
    """Se realisa la divición del dataset junto al balanceo de los datos

    variación utiliza los datos despues del balanceo para calcular la 
    matriz de whitening y de ahí calcula el ICA para cada ventana individual

    Parameters
    ----------
    train: ARRAY, matriz que contiene las ventanas balanceadas de 
        entrenamiento.
    validation: ARRAY, matriz que contiene las ventanas balanceadas de 
        validación.
    test: ARRAY, matriz que contiene las ventanas balanceadas de prueba.
    num_ci: INT, contiene el numero de componentes independientes a 
        calcular, siempre es mayor que 4.
    tam_ventana: INT, indica el tamaño de ventana en numero de muestras.
    paso_ventana: INT, indica el paso entre ventanas en numero de
        muestras.
    
    Returns
    -------
    train_ica: ARRAY, matriz que contiene las ventanas de entrenamiento
        aplicadas la transformación ICA.
    validation_ica: ARRAY, matriz que contiene las ventanas de
        validacion aplicadas la transformación ICA.
    test_ica: ARRAY, matriz que contiene las ventanas de prueba
        aplicadas la transformación ICA.
    ica_total: OBJ, objeto para la transformar ica ya entrenado.
    whiten: ARRAY, matriz de transformación para blanqueamiento.
    
    """
    # Revisar que funcione correctamente la idea es determinar el 
    # numero de ventanas
    num_ventanas_entrenamiento = len(train)
    num_ventanas_validacion = len(validation)
    num_ventanas_prueba = len(test)
    print('Ventanas de entrenamiento: ' + str(num_ventanas_entrenamiento))
    print('Ventanas de validación: ' + str(num_ventanas_validacion))
    print('Ventanas de prueba: ' + str(num_ventanas_prueba))

    # para el vector de todos los datos de entrenamiento
    ica_total = FastICA(
        n_components=num_ci, algorithm='parallel',
        whiten='arbitrary-variance', fun='exp', max_iter=500)

    # Nuevo ICA
    senales = np.reshape(
        np.concatenate(train), [np.shape(train[0])[0],
                                num_ventanas_entrenamiento * tam_ventana], order='F')
    ica_total.fit(senales.T)

    # obtener la matriz de blanqueo
    whiten = ica_total.whitening_

    # aplicar transformaciones
    train_ica = TransformarICA(
        train, whiten, num_ventanas_entrenamiento,
        num_ci, tam_ventana)
    validation_ica = TransformarICA(
        validation, whiten, num_ventanas_validacion,
        num_ci, tam_ventana)
    test_ica = TransformarICA(
        test, whiten, num_ventanas_prueba,
        num_ci, tam_ventana)

    return train_ica, validation_ica, test_ica, ica_total, whiten


def VICA(
        train, validation, test, senales_subm, num_ci, tam_ventana,
        paso_ventana):
    """Se realisa la divición del dataset junto al balanceo de los datos

    Parameters
    ----------
    train: ARRAY, matriz que contiene las ventanas balanceadas de 
        entrenamiento.
    validation: ARRAY, matriz que contiene las ventanas balanceadas de 
        validación.
    test: ARRAY, matriz que contiene las ventanas balanceadas de prueba.
    senales_subm: ARRAY, matriz que contiene los datos de submuestreo.
    num_ci: INT, contiene el numero de componentes independientes a 
        calcular, siempre es mayor que 4.
    tam_ventana: INT, indica el tamaño de ventana en numero de muestras.
    paso_ventana: INT, indica el paso entre ventanas en numero de
        muestras.
    
    Returns
    -------
    train_ica: ARRAY, matriz que contiene las ventanas de entrenamiento
        aplicadas la transformación ICA.
    validation_ica: ARRAY, matriz que contiene las ventanas de
        validacion aplicadas la transformación ICA.
    test_ica: ARRAY, matriz que contiene las ventanas de prueba
        aplicadas la transformación ICA.
    ica_total: OBJ, objeto para la transformar ica ya entrenado.
    whiten: ARRAY, matriz de transformación para blanqueamiento.
    
    """
    # Revisar que funcione correctamente la idea es determinar el 
    # numero de ventanas
    num_ventanas_entrenamiento = len(train)
    num_ventanas_validacion = len(validation)
    num_ventanas_prueba = len(test)

    # Calcular el ultimo valor de entrenamiento
    lim_entre = paso_ventana * num_ventanas_prueba + tam_ventana

    # para el vector de todos los datos de entrenamiento
    ica_total = FastICA(
        n_components=num_ci, algorithm='parallel',
        whiten='arbitrary-variance', fun='exp', max_iter=500)
    ica_total.fit(senales_subm[:, :lim_entre].T)
    # obtener la matriz de blanqueo
    whiten = ica_total.whitening_
    # aplicar transformaciones
    train_ica = AplicarICA(
        num_ventanas_entrenamiento, num_ci, tam_ventana, ica_total, train)
    validation_ica = AplicarICA(
        num_ventanas_validacion, num_ci, tam_ventana, ica_total, validation)
    test_ica = AplicarICA(
        num_ventanas_prueba, num_ci, tam_ventana, ica_total, test)

    return train_ica, validation_ica, test_ica, ica_total, whiten


def FICA(
        train, validation, test, num_ci, tam_ventana,
        paso_ventana):
    """Se realisa la divición del dataset junto al balanceo de los datos

    Parameters
    ----------
    train: ARRAY, matriz que contiene las ventanas balanceadas de 
        entrenamiento.
    validation: ARRAY, matriz que contiene las ventanas balanceadas de 
        validación.
    test: ARRAY, matriz que contiene las ventanas balanceadas de prueba.
    num_ci: INT, contiene el numero de componentes independientes a 
        calcular, siempre es mayor que 4.
    tam_ventana: INT, indica el tamaño de ventana en numero de muestras.
    paso_ventana: INT, indica el paso entre ventanas en numero de
        muestras.
    
    Returns
    -------
    train_ica: ARRAY, matriz que contiene las ventanas de entrenamiento
        aplicadas la transformación ICA.
    validation_ica: ARRAY, matriz que contiene las ventanas de
        validacion aplicadas la transformación ICA.
    test_ica: ARRAY, matriz que contiene las ventanas de prueba
        aplicadas la transformación ICA.
    ica_total: OBJ, objeto para la transformar ica ya entrenado.
    whiten: ARRAY, matriz de transformación para blanqueamiento.
    
    """
    # Revisar que funcione correctamente la idea es determinar el 
    # numero de ventanas
    num_ventanas_entrenamiento = len(train)
    num_ventanas_validacion = len(validation)
    num_ventanas_prueba = len(test)

    # para el vector de todos los datos de entrenamiento
    ica_total = FastICA(
        n_components=num_ci, algorithm='parallel',
        whiten='arbitrary-variance', fun='exp', max_iter=500)

    # Nuevo ICA
    senales = np.reshape(
        np.concatenate(train), [np.shape(train[0])[0],
                                num_ventanas_entrenamiento * tam_ventana], order='F')
    ica_total.fit(senales.T)

    # obtener la matriz de blanqueo
    whiten = ica_total.whitening_
    # aplicar transformaciones

    train_ica = AplicarICA(
        num_ventanas_entrenamiento, num_ci, tam_ventana, ica_total, train)
    validation_ica = AplicarICA(
        num_ventanas_validacion, num_ci, tam_ventana, ica_total, validation)
    test_ica = AplicarICA(
        num_ventanas_prueba, num_ci, tam_ventana, ica_total, test)

    return train_ica, validation_ica, test_ica, ica_total, whiten


def CrearDirectorio(direc):
    """Crea una carpeta en el directorio dado

    Parameters
    ----------
    direc: STR, El nombre de la carpeta a crear

    Returns
    -------
    
    """
    try:
        os.mkdir(direc)
    except OSError:
        print('Error en crear %s' % direc)


def Directo(path):
    """Para crear las carpetas donde se guardan los datos
    
    se crean las sigueintes carpetas;
        /Clasificador   -> Datos de los clasificadores
        /Procesamiento  -> Datos de los filtros e ICA
        /General        -> Datos generales de la interfaz

    Parameters
    ----------
    path: STR, El directorio donde crear las carpetas

    Returns
    -------
    
    """
    path_a = path + '/Clasificador'
    path_b = path + '/Procesamiento'
    path_c = path + '/General'
    # Crea nuevos directorios para la siguiente id
    CrearDirectorio(path_a)
    CrearDirectorio(path_b)
    CrearDirectorio(path_c)


def Directorios(sujeto):
    """Crea los directorios donde se guardan los datos

    La extructuar de los directorios es la siguiente:
    /Parametros     -> Carpeta donde se guardan los datos
        /Sujeto_n   -> Carpeta que guarda los datos del sujeto n
            /###    -> Carpeta id de la vez que se entrenó
                /Clasificador   -> Datos de los clasificadores
                /Procesamiento  -> Datos de los filtros e ICA
                /General        -> Datos generales de la interfaz

    Parameters
    ----------
    sujeto: STR, corresponde al numero del sujeto elegido.
    
    Returns
    -------
    path: STR, Dirección de la id donde se guardan los datos.
    ubi: STR, identificación (id) del entrenamiento.
    
    """
    # Revisar si la carpeta parametros existe
    if os.path.exists('Parametros'):
        path = 'Parametros/Sujeto_' + str(sujeto)
        # revisa si la carpeta del sujeto existe
        if os.path.exists(path):
            # Revisa cual es la mayor id creada
            direc = os.listdir(path)
            # revisa que la lista no esté vacia
            if direc:
                # revisa que todos los valores de la lista sean 
                # numericos y se componen por tres caracteres
                if all(carpetas.isnumeric() for carpetas in direc) and all(len(carpetas) == 3 for carpetas in direc):
                    print('Se crea nueva sesión')
                    # la sigiente sesión
                    siguiente = int(max(direc)) + 1
                    # la dirección de la id a guardar
                    path = path + '/' + format(siguiente, '03')
                    CrearDirectorio(path)
                    Directo(path)

                # para el caso de que existan carpetas distientas a las 
                # del formato
                else:
                    print('Revisa y crea una nueva sesion')
                    # Iteración con todos los nombres de las carpetas
                    maximo = '000'
                    for carpetas in direc:
                        # Cumpen las condiciones para ser una id
                        if (carpetas.isnumeric()) and (len(carpetas) == 3):
                            # revisa el valor maximo y lo almacena
                            if maximo < carpetas: maximo = carpetas

                    # Determinar la siguiente id
                    siguiente = int(maximo) + 1
                    path = path + '/' + format(siguiente, '03')
                    CrearDirectorio(path)
                    Directo(path)

            else:
                print('No hay entrenamientos')
                path = path + '/000'
                CrearDirectorio(path)
                Directo(path)

        else:
            print('No existe el Sujeto')
            CrearDirectorio(path)
            path = path + '/000'
            CrearDirectorio(path)
            Directo(path)

    # Crea todos los directorios
    else:
        print('No existe Parametros')
        path = 'Parametros/'
        CrearDirectorio(path)
        path = path + 'Sujeto_' + str(sujeto)
        CrearDirectorio(path)
        path = path + '/000'
        CrearDirectorio(path)
        Directo(path)

    # Id del entrenamiento
    ubi = path[-3:]

    return path, ubi


def GuardarMetricas(metricas):
    """Para guardar las metricas

    Guarda las metricas en el archivo Rendimiento.csv ubicado en la
    carpeta de Parametros, en el caso de que el archivo ya haya sido 
    creado, este concatena las metricas añadidas, en caso contrario,
    crea el archivo.

    Parameters
    ----------
    metricas: DICT, contiene las metricas a guardar
    
    Returns
    -------
    
    """
    # Convertir los datos en dataframe
    datos = pd.DataFrame(metricas, index=[0])
    directo = 'Parametros/Rendimiento.csv'

    # en el caso de que el archivo ya exista
    if os.path.exists(directo):
        # el mode = 'a' es para concatenar los datos nuevos
        datos.to_csv(directo, header=False, index=False, mode='a')
    # para cuando no existe
    else:
        datos.to_csv(directo, index=False)
        
def GuardarConfiguracion(configuracion):
    """Para guardar la configuracion

    Guarda la configuración en el archivo Configuracion.csv ubicado en la
    carpeta de Parametros, en el caso de que el archivo ya haya sido 
    creado, este concatena las metricas añadidas, en caso contrario,
    crea el archivo.

    Parameters
    ----------
    metricas: DICT, contiene las metricas a guardar
    
    Returns
    -------
    
    """
    # Convertir los datos en dataframe
    datos = pd.DataFrame(configuracion, index=[0])
    directo = 'Parametros/Configuracion.csv'

    # en el caso de que el archivo ya exista
    if os.path.exists(directo):
        # el mode = 'a' es para concatenar los datos nuevos
        datos.to_csv(directo, header=False, index=False, mode='a')
    # para cuando no existe
    else:
        datos.to_csv(directo, index=False)


def DeterminarDirectorio(sujeto, tipo, tam_ventana = None):
    """Determina la ubicación del directorio con los datos

    Determina cual es el entrenamiento que logró mejor precisión, esto 
    mediante la busqueda en el archivo Rendimiento.csv.

    Parameters
    ----------
    sujeto: INT, corresponde al numero del sujeto elegido.
    tipo: STR, Indica el tipo de señales a determinar puede ser 'EEG', 
        'EMG' o 'Combinada'.
    tam_ventana: INT, determinar el direcotrio para un tamaño de
        ventana especifico.
    
    Returns
    -------
    path: STR, Dirección de la id donde se guardan los datos.
    ubi: STR, identificación (id) del entrenamiento.
    existe: BOOL, indica la existencia de datos a cargar.
    """
    # Ubicación del archivo
    directo = 'Parametros/Rendimiento.csv'
    # El archivo existe
    if os.path.exists(directo):
        datos = pd.read_csv(directo)
        # revisar que datos corresponden al sujeto actual
        data = datos[datos['Sujeto'] == sujeto]
        # Revisar unicamente a la mejor presición combinada
        data = data[data['Tipo de señales'] == tipo]

        # revisa que el dataframe no esté vacío
        if not data.empty:
            existe = True
            # Cambia el formato de ubi que se pierde al cargar
            if tam_ventana is None:
                ubi = format(data['Id'][data['Exactitud'].idxmax()], '03')
                path = 'Parametros/Sujeto_' + str(sujeto) + '/' + ubi
            # En el caso de que se de un tamaño de ventana
            else:
                config = pd.read_csv('Parametros/Configuracion.csv')
                # determina que ids concuerdan con ese tamaño de ventana
                ids = config.loc[
                    (config['tamaño ventana ms'] == tam_ventana), ['Id']
                    ].squeeze(axis=1)
                # revisa que el dataframe no esté vacío
                if not ids.empty:
                    pista = data[data['Id'].isin(ids)]
                    ubi = format(pista['Id'][pista['Exactitud'].idxmax()], '03')
                    path = 'Parametros/Sujeto_' + str(sujeto) + '/' + ubi
                # En el caso de que el dataframe esté vacío
                else:
                    existe = False
                    ubi = None
                    path = None
                    print('No se encuentra una ventana igual para el sujeto' + str(sujeto))
                    
        # En el caso de que el dataframe esté vacío
        else:
            existe = False
            ubi = None
            path = None
            print('No se encuentran datos del sujeto ' + str(sujeto))
    # En el caso de que no haya ningún archivo
    else:
        existe = False
        ubi = None
        path = None
        print('No se encuetran datos de entrenamiento')

    return path, ubi, existe


def Clasificador(
        train, class_train, validation, class_validation, test, class_test, path,
        tipo, num_ci, tam_ventana, nombre_clases, num_clases, epocas, lotes):
    """Diseña y entrena un clasificador

    Guarda los datos de entrenamiento mediante puntos de control 
    disponibles con tensorflow, e imagenes que resumen el rendimiento

    Parameters
    ----------
    train: ARRAY, matriz que contiene las ventanas balanceadas de 
        entrenamiento.
    class_train: ARRAY, matriz que contiene las clases de las ventanas 
        balanceadas de entrenamiento.
    validation: ARRAY, matriz que contiene las ventanas balanceadas de 
        validación.
    class_validation: ARRAY, matriz que contiene las clases de las 
        ventanas balanceadas de validación.
    test: ARRAY, matriz que contiene las ventanas balanceadas de prueba
    class_test: ARRAY, matriz que contiene las clases de las ventanas
        balanceadas de prueba.
    path: STR, Dirección de la id donde se guardan los datos.
    tipo: STR, Tipo de señales, ya sea "EEG" o "EMG" en mayusculas.
    num_ci: INT, Numero de canales de entrada, en terminos de 
        componetes independientes.
    tam_ventana: INT, indica el tamaño de ventana en numero de muestras.
    nombre_clases: LISTA, contiene los nombres de las clases.
    num_clases: INT, valor del nuemro de clases a usar.
    epocas: INT, número de epocas (epochs) para el entrenamiento
    lotes: INT, número de lotes (batch) para el enttrenamiento

    Returns
    -------
    modelo: OBJ: tf.keras.Sequential, contiene la CNN entrenada.
    cnn: OBJ: keras.callbacks.History, contiene historial de grabación
        del modelo.
    eva: DICT, contiene las metricas del modelo sobre los datos de 
        prueba.
    confusion: DICT, Contiene los datos de las matrices de confución 
        del modelo entrenado.
        confusion = 'Validacion': confusion_val
                    'Prueba': confusion_pru
    prediccion: np.ARRAY, Contiene las clases predichas por el modelo
        aplicado a los datos de prueba.
    
    """
    def Entrenamiento(return_dict, modelo):
            # return_dict, modelo, train, validation, class_train, 
            # class_validation, epocas, lotes):
        """Para Crear un proceso donde se realice el entrenamieto
        
        Parameters
        ----------
        return_dict : TYPE
            DESCRIPTION.
        modelo : TYPE
            DESCRIPTION.
        train : TYPE
            DESCRIPTION.
        validation : TYPE
            DESCRIPTION.
        class_train : TYPE
            DESCRIPTION.
        class_validation : TYPE
            DESCRIPTION.
        epocas : TYPE
            DESCRIPTION.
        lotes : TYPE
            DESCRIPTION.

        Returns
        -------
        modelo : TYPE
            DESCRIPTION.
        historial : TYPE
            DESCRIPTION.
        
        """
        print("Se entrena el modelo")
        historial = []
        # historial = modelo.fit(
            # train, class_train, shuffle=True, epochs=epocas, batch_size=lotes,
            # validation_data=(validation, class_validation))
        # print("Terminó el entrenamiento")
        return_dict = {'modelo': modelo, 'historial': historial}
        return return_dict
    
    # desactivar el uso de GPU (no hay suficiente memoria de GPU para entrenar)
    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        print('No se pudo desactivar la GPU')
        pass
   
    # ajustar la drección de guardado
    clasificador_path = path + '/Clasificador/' + tipo + '/'
    # Crear punto de control del entrenamiento
    # (se cambia a guardar todo el modelo)
    # checkpoint_path = clasificador_path + tipo + "_cp.ckpt"
    # Dirección para guardar el modelo
    modelo_path = clasificador_path + tipo + "_modelo.h5"

    # Modelo
    if tipo == 'EMG':
        modelo = ClasificadorEMG(num_ci, tam_ventana, num_clases)
        # modelo = ClasificadorCanales(num_ci, tam_ventana, num_clases)
    elif tipo == 'EEG':
        modelo = ClasificadorEEG(num_ci, tam_ventana, num_clases)
    modelo.summary()

    # Crea un callback que guarda los pesos del modelo
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_path, save_weights_only=True, verbose=1)
    
    # revisar la creación de un proceso para ejecutar el entrenaamiento
    # o en el peor de lo casos toda la clasificación
    # entrenamiento en un nuevo proceso
    # para que se retornen variables del proceso
    # organizador = multiprocessing.Manager()
    # return_dict = organizador.dict()
    # proceso = multiprocessing.Process(
    #     target=Entrenamiento, args=(
    #         return_dict, modelo))
    #         # return_dict, modelo, train, validation, class_train, 
    #         # class_validation, epocas, lotes))
    # # iniciar la ejecución de proceso de entrenamiento
    # proceso.start()
    # # Esperar a que termine la ejecución del proceso
    # proceso.join()
    # # recuperar los datos realizados en el proceso
    # cnn = return_dict['historial']
    # modelo_b = return_dict['modelo']
    
    # Entrenamiento del modelo 
    cnn = modelo.fit(
        train, class_train, shuffle=True, epochs=epocas, batch_size=lotes,
        validation_data=(validation, class_validation))
    # agregar a modelo.fit() si se reactivan los puntos de control
    #    callbacks=[cp_callback])
    
    # guardar todo el modelo no solo los parametros
    # puede que ocupe más espacio en el disco
    modelo.save(modelo_path)

    eva = modelo.evaluate(
        test, class_test, verbose=1, return_dict=True)
    print("La precición del modelo: {:5.2f}%".format(
        100 * eva['categorical_accuracy']))

    # Para las matrices de confución
    # A los datos de validación
    prediccion_val = modelo.predict(validation)
    confusion_val = confusion_matrix(
        argmax(class_validation, axis=1), argmax(prediccion_val, axis=1))
    # Aplicar a los datos de prueba
    prediccion = modelo.predict(test)
    confusion_pru = confusion_matrix(
        argmax(class_test, axis=1), argmax(prediccion, axis=1))

    confusion = {
        'Validacion': confusion_val,
        'Prueba': confusion_pru
    }

    return modelo, cnn, eva, confusion, prediccion


def grafica_clasifi(axcla, cnn, fontsize=12, senales='EEG', tipo='loss'):
    """Para hacer las graficas del entrenamiento rapido
    """
    # para que el eje x vaya desde 1 hasta la cantidad de epocas
    # axcla.set_xticks(range(1, len(cnn.history[tipo])+1))
    # Se agregan los datos a las graficas
    axcla.plot(cnn.history[tipo])
    axcla.plot(cnn.history['val_' + tipo])
    axcla.margins(x=0)
    if tipo == 'loss':
        axcla.set_title('Perdida del modelo de ' + senales)
        axcla.set_ylabel('Perdida')
    elif tipo == 'categorical_accuracy':
        axcla.set_title('Exactitud del modelo de ' + senales)
        axcla.set_ylabel('Exactitud')
    axcla.set_xlabel('Epocas')
    axcla.legend(['Entrenamiento', 'Validación'])


def Graficas(path, cnn, confusion, nombre_clases, tipo):
    """Diseña la y entrena un clasificador

    Parameters
    ----------
    path: STR, Dirección de la id donde se guardan los datos.
    cnn: OBJ; keras.callbacks.History, Historial de entrenamiento del 
        modelo.
    confusion: np.ARRAY, Contiene los datos de la matriz de confución 
        del modelo entrenado.
    nombre_clases: LISTA, contiene los nombres de las clases.
    tipo: STR, Tipo de señales, ya sea "EEG" o "EMG" en mayusculas.

    Returns
    -------

    """
    # Imprimir la matriz de confución de los modelos por separado
    # El dataframe
    cm = pd.DataFrame(
        confusion['Prueba'], index=nombre_clases, columns=nombre_clases)
    cm.index.name = 'Verdadero'
    cm.columns.name = 'Predicho'
    # La figura
    fig_axcm = plt.figure(figsize=(10, 8))
    axcm = fig_axcm.add_subplot(111)
    sns.heatmap(
        cm, cmap="Blues", linecolor='black', linewidth=1, annot=True,
        fmt='', xticklabels=nombre_clases, yticklabels=nombre_clases,
        cbar_kws={"orientation": "vertical"}, annot_kws={"fontsize": 13}, ax=axcm)
    axcm.set_title(
        'Matriz de confusión de CNN para ' + tipo + ' - prueba', fontsize=21)
    axcm.set_ylabel('Verdadero', fontsize=16)
    axcm.set_xlabel('Predicho', fontsize=16)
    # para validación
    # El dataframe
    cm_val = pd.DataFrame(
        confusion['Validacion'], index=nombre_clases, columns=nombre_clases)
    cm_val.index.name = 'Verdadero'
    cm_val.columns.name = 'Predicho'
    # La figura
    fig_axcm_val = plt.figure(figsize=(10, 8))
    axcm_val = fig_axcm_val.add_subplot(111)
    sns.heatmap(
        cm_val, cmap="Blues", linecolor='black', linewidth=1, annot=True,
        fmt='', xticklabels=nombre_clases, yticklabels=nombre_clases,
        cbar_kws={"orientation": "vertical"}, annot_kws={"fontsize": 13}, ax=axcm_val)
    axcm_val.set_title(
        'Matriz de confusión de CNN para ' + tipo + ' - validación', fontsize=21)
    axcm_val.set_ylabel('Verdadero', fontsize=16)
    axcm_val.set_xlabel('Predicho', fontsize=16)

    # Graficas de entrenamiento
    tamano_figura = (10, 8)
    figcla, (axcla1, axcla2) = plt.subplots(
        nrows=2, ncols=1, figsize=tamano_figura)
    figcla.suptitle(
        'Información sobre el entrenamiento del clasificador', fontsize=21)
    # figcla.set_xticks(range(1, len(cnn.history[tipo])))
    grafica_clasifi(axcla1, cnn, fontsize=13, senales=tipo, tipo='categorical_accuracy')
    grafica_clasifi(axcla2, cnn, fontsize=13, senales=tipo, tipo='loss')

    plt.tight_layout()
    # Guardar graficas:
    path = path + '/General/'
    fig_axcm.savefig(path + 'CM_Pru_' + tipo + '.png', format='png')
    fig_axcm_val.savefig(path + 'CM_Val_' + tipo + '.png', format='png')
    figcla.savefig(path + 'Entrenamiento_' + tipo + '.png', format='png')


def PresicionClases(confusion_val):
    """Calculo de la presición de cada clase junto a la excatitud 
    general del calsificadorde acuerdo con la CM.

    Parameters
    ----------
    confusion_val: np.ARRAY, contiene los datos de la matriz de
        confusión.

    Returns
    -------
    precision_clase = np.ARRAY, contiene la presición de cada clase,
        calculada a partir de la matriz de confución.
    exactitud = FLOAT, indica el valor de la exactitud calculado a 
        partir de la matriz de confución.
    
    """
    # calculo de pesos de acuerdo a la precisión
    precision_clase = np.zeros(len(confusion_val))
    # la exactitud
    exactitud = 0
    for i in range(len(confusion_val)):
        if sum(confusion_val[:, i]) == 0:
            precision_clase[i] = 0
        else:
            precision_clase[i] = confusion_val[i, i] / sum(confusion_val[:, i])
            # va sumando los valores correctamente clasificados
            exactitud = exactitud + confusion_val[i, i]

    # Calcula la exactitud
    exactitud = exactitud / confusion_val.sum()

    return precision_clase, exactitud


def ExactitudClases(confusion):
    """Calculo de la exactitud de cada clase junto a la excatitud 
    general del calsificadorde acuerdo con la CM.
    
    De acuerdo con la eq. (8,3) de Fundamentals of Machine Learning for 
    Predictive Data Analytics de John D. Kelleher et. al.
    
    classification accuracy = (TP+TN) / (TP+TN+FP+FN) 

    Parameters
    ----------
    confusion: np.ARRAY, contiene los datos de la matriz de
        confusión.

    Returns
    -------
    exactitud_clase = np.ARRAY, contiene la exactitud de cada clase,
        calculada a partir de la matriz de confución.
    
    """
    num_clases = len(confusion)
    # calculo de pesos de acuerdo a la exactitud
    exactitud_clase = np.zeros(num_clases)

    # total de datos = TP + TN + FP `FN
    total = np.sum(confusion)

    for i in range(num_clases):
        # crea una matriz del tamaño de la cm llena de bool true
        mascara = np.full(confusion.shape, True, dtype='bool')
        mascara[:, i] *= False
        mascara[i, :] *= False

        TP = confusion[i, i]
        TN = np.sum(confusion * mascara)
        # Eq. (8,3): cls. accuracy = (TP+TN) / (TP+TN+FP+FN) 
        exactitud_clase[i] = (TP + TN) / total

    return exactitud_clase


def CalculoPesos(confusion_val_emg, confusion_val_eeg):
    """Calculo de peso para la convinación de los clasificadores

    Parameters
    ----------
    confusion_val_emg: np.ARRAY, Contiene los datos de la matriz de
        confusiónn de los datos de validación de las señales de EMG.
    confusion_val_emg: np.ARRAY, Contiene los datos de la matriz de
        confusiónn de los datos de validación de las señales de EMG.

    Returns
    -------
    w = np.ARRAY, Matriz que contiene los pesos para la convinación de 
        los clasificadores.
    
    """
    precision_emg,_ = PresicionClases(confusion_val_emg)
    precision_eeg,_ = PresicionClases(confusion_val_eeg)
    
    # exactitud_emg = ExactitudClases(confusion_val_emg)
    # exactitud_eeg = ExactitudClases(confusion_val_eeg)

    # calculo del vector de deción eq. 5.45 kuncheva
    # u[j] = sum from i=1 to L (w[i,j] * d[i,j]) 

    # matriz de pesos
    w = [precision_emg, precision_eeg]

    return w


def GraficaMatrizConfusion(confusion_combinada, nombre_clases, path):
    """Grafica y guarda la matriz de confusión combinada

    Parameters
    ----------
    confusion_combinada: np.ARRAY, Contiene los datos de la matriz de
        confusión de los datos combinados.
    nombre_clases: LISTA, contiene los nombres de las clases..
    path: STR, Dirección de la id donde se guardan los datos.

    Returns
    -------
    
    """
    # Matriz de confusión
    cm_combinada = pd.DataFrame(
        confusion_combinada, index=nombre_clases, columns=nombre_clases)
    cm_combinada.index.name = 'Verdadero'
    cm_combinada.columns.name = 'Predicho'
    # Figura
    fig_axcm_combinada = plt.figure(figsize=(10, 8))
    axcm_combinada = fig_axcm_combinada.add_subplot(111)
    sns.set(font_scale=1.7)
    sns.heatmap(
        cm_combinada, cmap="Purples", linecolor='black', linewidth=1,
        annot=True, fmt='', xticklabels=nombre_clases,
        yticklabels=nombre_clases, cbar_kws={"orientation": "vertical"},
        annot_kws={"fontsize": 13}, ax=axcm_combinada)
    axcm_combinada.set_title(
        'Matriz de confusión de clasificadores combinados', fontsize=21)
    axcm_combinada.set_ylabel('Verdadero', fontsize=16)
    axcm_combinada.set_xlabel('Predicho', fontsize=16)
    # Guardar datos
    path = path + '/General/'
    fig_axcm_combinada.savefig(path + 'CM_Combinada.png', format='png')
    pass


# ----------------------------------------------------------------------------
# Funciones creadas para las recomendaciones del profesor
def Ventanas(
        registros, clases_regis, num_canales, num_clases, reclamador, descarte,
        tam_ventana, paso_ventana, salto):
    """Para realizar en enventanado de un registro
    """
    tipo_ventana = 'hamming'
    
    # if tipo_ventana == 'hamming':
    #     ventana = signal.windows.hamming(tam_ventana, sym=True)
    # elif tipo_ventana == 'bartlett':
    #     ventana = signal.windows.bartlett(tam_ventana, sym=True)
    
    # crea el tipo de ventana OCURRIO UN ERROR DE COMPATIVILIDA CON MATCH
    match tipo_ventana:
        case 'hamming':
            ventana = signal.windows.hamming(tam_ventana, sym=True)
        case 'bartlett':
            ventana = signal.windows.bartlett(tam_ventana, sym=True)
    
    #
    clase_reposo = np.eye(num_clases, dtype='int8')[:,-1]
    num_vent_reposo = int(reclamador['Reposo']/paso_ventana)
    num_vent_actividad = int(descarte['Reposo']/paso_ventana)
    # Calcula el numero total de ventanas, usando el tamaño de las sesiones
    num_vent = int(
        (num_vent_actividad + num_vent_reposo)*(
            clases_regis[0].shape[1]
            + clases_regis[1].shape[1]
            + clases_regis[2].shape[1]))
    
    # enventanado final
    ventanas = np.empty([num_vent, num_canales, tam_ventana])
    clases = np.empty([num_vent, num_clases], dtype='int8')
    v = 0
    for sesion in range(3):
        # para revisar cada registro
        for registro in range(clases_regis[sesion].shape[1]):
            # para ventanas de reposo
            for i in range(num_vent_reposo):
                # para el numero del canal
                k = 0
                for canal in registros.keys():
                    ventanas[v,k,:] = np.multiply(
                        registros[canal][sesion][
                            registro,
                            reclamador['Activo'] + paso_ventana*i:
                            reclamador['Activo'] + tam_ventana + paso_ventana*i], 
                        ventana)
                    k += 1
                clases[v,:] = clase_reposo
                v += 1
            del i, k
            # para ventanas de actividad
            for i in range(num_vent_actividad):
                k = 0
                for canal in registros.keys():
                    ventanas[v,k,:] = np.multiply(
                            registros[canal][sesion][
                            registro, salto + descarte['Activo'] + paso_ventana*i:
                            salto + descarte['Activo'] + tam_ventana + paso_ventana*i], 
                        ventana)
                    k += 1
                clases[v,:] = clases_regis[sesion][:,registro]
                v += 1
            del i, k
        del registro
    del sesion, v

    return ventanas, clases

def CICA(senales, num_ci):
    """Para calcular las matrices de transformación para ICA
    """
    from sklearn.decomposition import FastICA  # implementación de FastICA

    # Diseñar la matriz de confusioón
    ica_total = FastICA(
        n_components=num_ci, algorithm='parallel',
        whiten='arbitrary-variance', fun='exp', max_iter=500)
    ica_total.fit(senales.T)
    # obtener la matriz de blanqueo
    whiten = ica_total.whitening_
    return ica_total, whiten

# ----------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------
# Gracias por llegar hasta aquí, aprecio que revise todas estas funciones
