# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 12:54:34 2022

@author: Daniel
"""

'''
#-----------------------------------------------------------------------------
# Por hacer
#-----------------------------------------------------------------------------
se pasa a la version 0.60

Ajustar la forma de guardar y cargar los datos cambiando los directorios donde
se guardan. 

Cambiar el los electrodos de EEG para que se posicionen sobre la corteza
motora.

Tomar datos de las tres sesiones y hacer el balanceo de datos respectivo

Probar si quitando ICA en datos de EMG mejora la clasificación

Guardar datos de matrices de confución de los clasificadores de EEG y EMG para
poder cargarlos y calcular los pesos para la convinación de estos.

Reducir el numero de parametros a entrenar por cada red neuronal de forma que
el numero de parametros a entrenar sea unas 5 ó 10 veces menor que el numero 
de ventanas de entrenamiento
    
    El numero de ventanas:
        Ventanas totales: 12,827
        Ventanas de entrenamiento: 10,261
        Ventanas de pruba: 2,566
        

#-----------------------------------------------------------------------------
# Notas de verciones
#-----------------------------------------------------------------------------

0.10:   Primera version (solo realizaba hasta la extracción de caracteristicas).

0.20:   Se añadió la clasificación.

0.30:   Se crearon funciones para diversas tareas repetitivas o que se puedan
        modificar ciertos parametros, sin tener que volver a escribir codigo.

0.33:   Se re escribe el codigo de acuerdo a la guia de estilo para codigo de 
        python PEP 8, pero se omite la recomendación sobre el uso casi 
        obligatorio del idioma ingles, dado que consumiria demaciado tiempo
        traduccir todas las variables y comentarios.
        
0.40:   Se implementan los Puntos de control en el entrenamiento del 
        clasificador, a demás de que se reducen el numero de parametros
        entrenables de los clasificadores, y se modifica el llamado a la
        función encargada de la creación de la extructura del clasificador.
        
0.44:   Se corrigen lineas para la impreción de graficas de las señales, se 
        imprime el rendimiento del calsificador en terminos de matrices de 
        confunción, se combinan la salida de los dos clasificadores.
        

0.50:   Se realiza balance de la base de datos mediante el submuestreo de esta
        se actualiza la combinan la salida de los dos clasificadores.
        se dividen los datos en entrenamiento, validación y prueba.
    
0.51:   Se empieza a realizar guardado de la transformación FastICA para
        un calculo individal para cada una de las ventanas. Se agregan los 
        datos de validación de forma separada.

0.52:   Se busca re calcular el FastICA para que se lo pueda guardar,
        Se corrige el balance de datos y se actualizan las graficas.
        
0.53:   Se corrige el balanceo de datos para que correspondan los datos
        de EEG y EMG de forma que se pudan usar en la combinación.
        
0.54:   Se reducen ciertas lineas de codigo, y se cambian las funciones que
        dan la estructura a las redes neuronales, haciendo que reciban el
        parametro de numero de clases para la clasificación
        
'''
#-----------------------------------------------------------------------------
# Librerias
#-----------------------------------------------------------------------------
print('Cargando Librerias ...')

# General
import numpy as np
import math #para aproximaciones
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Para filtro
from scipy import signal

# Para dividir datos de test y entrenamiento
from sklearn.model_selection import train_test_split

# Para el fastICA
from sklearn.decomposition import FastICA #implementacion de FastICA

# Para la RNC
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam

# Para matrices de confución
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from tensorflow.math import argmax # para convertir de one hot a un vector
import seaborn as sns # para el mapa de calor


# Para guardar puntos de control
import os
import tensorflow as tf
# from tensorflow import keras

# Mis funciones
import Funciones as f


print('Librerias cargadas')

#-----------------------------------------------------------------------------
# General
#-----------------------------------------------------------------------------
print('Extrayendo información de la base de datos ...')

# Parametros generales
haceremg = True
hacereeg = True

graficar_senales = True
graficar_filtradas = True
graficar_ventanas = True
graficar_ica = True #no usar con ica_antiguo
graficar_clasificador = True
graficar_combinada = True

quitar_imaginacion_motora = False # para quitar las ventanas de IM
balanceo = True # Para balancear la base de datos
hacer_ica = True # de momento lo de ica no hace nada
ica_antiguo = False # antigüa forma de calcular FastICA
cargar_ica = False
hacer_clasificador = False
cargar_clasificador = False


#-----------------------------------------------------------------------------
# Preprocesamiento
#-----------------------------------------------------------------------------

# Direcciones para el guardado de datos
PATH = 'D:\ASUS\Documents\Tareas\Trabajo de grado\Trabajo de grado\
    Software\Clasificadores_guardados'
sujeto = '2'
sesion = '1'

# Direcciones
PATH = "G:/Proyectos/ICCH/Dataset/"
direccion_emg = PATH + 'Subjet_' + sujeto + '/EMG_session' + sesion +'_sub' + sujeto + '_reaching_realMove.mat'
direccion_eeg = PATH + 'Subjet_' + sujeto + '/EEG_session' + sesion +'_sub' + sujeto + '_reaching_realMove.mat'

#direccion_emg = 'Sujeto_2/EMG_session1_sub2_reaching_realMove.mat'
#direccion_eeg = 'Sujeto_2/EEG_session1_sub2_reaching_realMove.mat'

#-----------------------------------------------------------------------------
# Generalidades

# Saca los nombres de todos los canales de EEG y EMG del dataset
nombres = f.NombresCanales(direccion_eeg, direccion_emg)

# Lista con los nombres de todos los canales

# canales_emg = [
#    'EMG_1', 'EMG_2', 'EMG_3', 'EMG_4', 'EMG_5', 'EMG_6', 'EMG_ref'
#    ]

# EEG 10-10
# canales_eeg = [
#    'Fp1', 'AF7', 'AF3', 'AFz', 'F7', 'F5', 'F3', 'F1', 'Fz', 'FT7', 
#    'FC5', 'FC3', 'FC1', 'T7', 'C5', 'C3', 'C1', 'Cz', 'TP7', 'CP5', 
#    'CP3', 'CP1', 'CPz', 'P7', 'P5', 'P3', 'P1', 'Pz', 'PO7', 'PO3', 
#    'POz', 'Fp2', 'AF4', 'AF8', 'F2', 'F4', 'F6', 'F8', 'FC2', 'FC4', 
#    'FC6', 'FT8', 'C2', 'C4', 'C6', 'T8', 'CP2', 'CP4', 'CP6', 'TP8', 
#    'P2', 'P4', 'P6', 'P8', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz'
#    ]

# EEG 10-20
# canales_eeg = [
#    'Fp1', 'F7', 'F3', 'Fz', 'T7', 'C3', 'Cz', 'P7', 'P3', 'Pz', 
#    'Fp2', 'F4', 'F8', 'C4', 'T8', 'P4', 'P8', 'O1', 'Oz', 'O2'
#    ]

# El canal Fpz se utiliza como tierra

# Crear los canales para usar ahora
nombres['Canales EMG'] = [
    'EMG_1', 'EMG_2', 'EMG_3', 'EMG_4', 'EMG_5', 'EMG_6', 'EMG_ref'
    ]

nombres['Canales EEG'] = [
            'Fz', 'FC3', 'FC1', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
            'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz',
            'P2', 'POz']

# Nombres de los canales originales del dataset a los del estandar
canales_emg = f.TraducirNombresCanales(nombres['Canales EMG'])
canales_eeg = f.TraducirNombresCanales(nombres['Canales EEG'])

nombre_clases = ['Forward', 'Backward', 'Left', 'Right', 'Up', 'Down', 'Rest']

# Los datos
annots_EMG = loadmat(direccion_emg)
# Los datos
if hacereeg:
    annots_EEG = loadmat(direccion_eeg)

# Sobre información de las grabaciones

# Incio de la grabación
inicio_grabacion = annots_EMG['mrk'][0][0][5][0][0][0][0][0]
# Final de la grabación
final_grabacion = annots_EMG['mrk'][0][0][5][0][0][0][0][1] 
# Frecuencia de muestreo 2500 Hz
frec_muestreo = annots_EMG['mrk'][0][0][2][0][0]
# Banderas que indican incio y final de la ejecución de la actividad
banderas = annots_EMG['mrk'][0][0][0][0]
# Matriz one-hot de todas las actividades, correspondiente al intervalo dado por las banderas.
One_Hot = annots_EMG['mrk'][0][0][3]
# Calculo de numero de canales
num_clases = len(One_Hot)
num_canales_emg = len(canales_emg) 
num_canales_eeg = len(canales_eeg)

# del direccion_eeg, direccion_emg, 
# Actualizar para que no sea dependiente de la función de traducir nombre.
# del nombres


print('Datos extraidos')

#-----------------------------------------------------------------------------
# Graficar señales originales
#para las graficas
# desfase = 7468
tam_ventana_ms = 1000 #ms
paso_ms = 30 #ms
n = 460
t = np.linspace(
    int(inicio_grabacion/frec_muestreo)+paso_ms*n,
    int(inicio_grabacion/frec_muestreo)+paso_ms*n+tam_ventana_ms,
    int(frec_muestreo*tam_ventana_ms*0.001))

if graficar_senales:
    # El t inicia en la n-ésima ventana hasta el fin de esta con las muestra de 
    # la frecuencia de muestreo correspondientes.
    
    # para EMG
    tamano_figura = (8,10)
    fig, ax = plt.subplots(len(canales_emg),1,sharex=True,figsize=tamano_figura)
    fig.suptitle('Señales de EMG')

    for i in range(len(canales_emg)):
        ax[i].plot(
            t,annots_EMG[canales_emg[i]][
                inicio_grabacion + int(paso_ms*frec_muestreo*n/1000):
                inicio_grabacion + int((paso_ms*n + tam_ventana_ms) 
                                        *frec_muestreo/1000)
                                        ])
        ax[i].margins(x=0)
        ax[i].set_ylabel(nombres['Canales EMG'][i])
    
    ax[i].set_xlabel('tiempo (ms)')
    fig.tight_layout()
    
    # para EEG
    tamano_figura = (8,16)
    fig, ax = plt.subplots(len(canales_eeg),1,sharex=True,figsize=tamano_figura)
    fig.suptitle('Señales de EEG')

    for i in range(len(canales_eeg)):
        ax[i].plot(
            t,annots_EEG[canales_eeg[i]][
                inicio_grabacion + int(paso_ms*frec_muestreo*n/1000):
                inicio_grabacion + int((paso_ms*n + tam_ventana_ms) 
                                        *frec_muestreo/1000)
                                        ])
        ax[i].margins(x=0)
        ax[i].set_ylabel(nombres['Canales EEG'][i])
    
    ax[i].set_xlabel('tiempo (ms)')
    fig.tight_layout()
    #plt.show()

#-----------------------------------------------------------------------------
# Transformada de Fourier de las ventanas
import scipy

# Periodo de muestreo = 1 / Frecuencia de muestreo
ffs = frec_muestreo
tts = 1/ffs # periodo de muestreo
ll = tam_ventana_ms*frec_muestreo/1000 # numero de muestras
tt = np.arange(0,int(ll))*tts # vector de tiempo
ff = ffs*(np.arange(0, int(ll/2)))/ll # vector de frecuencias

# para EMG
senales_emg_tf = [None] * num_canales_emg
for i in range(num_canales_emg):
    senales_emg_tf[i] = scipy.fft.fft(annots_EMG[canales_emg[i]][
        inicio_grabacion + int(paso_ms*frec_muestreo*n/1000):
        inicio_grabacion + int((paso_ms*n + tam_ventana_ms) 
                                *frec_muestreo/1000)
                                ].flatten())
        
# para EEG
senales_eeg_tf = [None] * num_canales_eeg
for i in range(num_canales_eeg):
    senales_eeg_tf[i] = scipy.fft.fft(annots_EEG[canales_eeg[i]][
        inicio_grabacion + int(paso_ms*frec_muestreo*n/1000):
        inicio_grabacion + int((paso_ms*n + tam_ventana_ms) 
                                *frec_muestreo/1000)
                                ].flatten())
    
# Figuras

# EMG
tamano_figura = (8,10)
fig, ax = plt.subplots(len(canales_emg),1,sharex=True,figsize=tamano_figura)
fig.suptitle('Espectro de frecuencias - señales de EMG')

for i in range(num_canales_emg):
    pone = 2*np.absolute(senales_emg_tf[i][0:int(ll/2)]/ll)
    ax[i].plot(ff,pone)
    ax[i].margins(x=0)
    ax[i].set_ylabel(nombres['Canales EMG'][i])

ax[i].set_xlabel('Frecuencia (Hz)')
fig.tight_layout()

#EEG
tamano_figura = (8,16)
fig, ax = plt.subplots(len(canales_eeg),1,sharex=True,figsize=tamano_figura)
fig.suptitle('Espectro de frecuencias - señales de EEG')

for i in range(num_canales_eeg):
    pone = 2*np.absolute(senales_eeg_tf[i][0:int(ll/2)]/ll)
    ax[i].plot(ff,pone)
    ax[i].margins(x=0)
    ax[i].set_ylabel(nombres['Canales EEG'][i])

ax[i].set_xlabel('Frecuencia (Hz)')
fig.tight_layout()

#-----------------------------------------------------------------------------
# Filtro
print('Diseñando filtros ...')

# Parametros de los filtros
# Tipo de filtro
f_tipo = 'butter' #Butterworth
# Tipo de banda de paso
b_tipo = 'bandpass'
# Orden del filtro
f_orden = 5 #Butterworth orden 4
# Frecuencia de corte o paso a 3db
frec_corte_emg = np.array([8, 510]) #Hz
frec_corte_eeg = np.array([4, 30]) #Hz

# Filtros EMG y EEG
if haceremg:
    filtro_emg = f.DisenarFiltro(
        f_tipo, b_tipo, f_orden, frec_corte_emg, frec_muestreo)
    # wn = 2*frec_corte_emg/frec_muestreo
    # filtro_emg = signal.iirfilter(f_orden, wn, btype=b_tipo, 
    #                           analog=False, ftype=f_tipo, output='sos')
if hacereeg:
    filtro_eeg = f.DisenarFiltro(
        f_tipo, b_tipo, f_orden, frec_corte_eeg, frec_muestreo)

# del f_tipo, b_tipo, f_orden, frec_corte

# graficas de respuesta en frecuencia de los filtros
w, h = scipy.signal.sosfreqz(filtro_eeg, fs=frec_muestreo)

fig, ax1 = plt.subplots()
ax1.set_title('Digital filter frequency response')

ax1.plot(w, 20 * np.log10(abs(h)), 'b')
ax1.set_ylabel('Amplitude [dB]', color='b')
ax1.set_xlabel('Frequency [Hz]')

ax2 = ax1.twinx()
angles = np.unwrap(np.angle(h))
ax2.plot(w, angles, 'g')
ax2.set_ylabel('Angle (radians)', color='g')
ax2.grid(True)
ax2.axis('tight')
fig.tight_layout() # plt.show()

print('Filtros diseñados')

#-----------------------------------------------------------------------------
# El dataframe
#-----------------------------------------------------------------------------
# Diseñar las clases One Hot
print('Diseñando las etiquetas de las clases en forma ONE HOT ...')

# Tomar la clase de onehot y asignarla a la clases oh de forma que cada
# indice corresponda con las banderas. 
# Dataframe para las clases one-hot
clases_OH = f.ClasesOneHot(
    nombre_clases, num_clases, final_grabacion, banderas, One_Hot)

print('Clases One Hot diseñadas')

#-----------------------------------------------------------------------------
# Aplicar el filtro a los datos y guradarlo en el data frame
print('Aplicando filtros a las señales ...')

# Señales en un dataframe gigante
# Señales de los canales seleccionados:
if haceremg:
    senales_EMG_filt = f.AplicarFiltro(canales_emg, filtro_emg, annots_EMG)
if hacereeg:
    senales_EEG_filt = f.AplicarFiltro(canales_eeg, filtro_eeg, annots_EEG)
    # wn = 2*30/frec_muestreo
    # filtro = signal.iirfilter(f_orden, wn, btype='low', 
    #                           analog=False, ftype=f_tipo, output='sos')
    
    # w_filt = scipy.signal.sosfilt(filtro,w)

print('filtros aplicados')

#-----------------------------------------------------------------------------
# Graficar señales filtradas (mismo tiempo que el anterior)
if graficar_filtradas:
    # n = 124
    # # El t inicia en la n-ésima ventana hasta el fin de esta con las muestra de 
    # # la frecuencia de muestreo correspondientes.
    # t = np.linspace(int(inicio_grabacion/frec_muestreo)+paso_ms*n,
    #                 int(inicio_grabacion/frec_muestreo)+paso_ms*n+tam_ventana_ms,
    #                 int(frec_muestreo*tam_ventana_ms*0.001))

    # Para EMG
    tamano_figura = (8,10)
    figfilt, axfilt = plt.subplots(len(canales_emg),1,sharex=True,
                                   figsize=tamano_figura)
    figfilt.suptitle('Señales de EMG filtradas')

    for i in range(len(canales_emg)):
        axfilt[i].plot(t,senales_EMG_filt[canales_emg[i]][
            inicio_grabacion + int(paso_ms*frec_muestreo*n/1000):
            inicio_grabacion + int((paso_ms*n+tam_ventana_ms)
                                   *frec_muestreo/1000)
            ])
        axfilt[i].margins(x=0)
        axfilt[i].set_ylabel(nombres['Canales EMG'][i])
    
    axfilt[i].set_xlabel('tiempo (ms)')
    figfilt.tight_layout()
    
    # Para EEG
    tamano_figura = (8,16)
    figfilt, axfilt = plt.subplots(len(canales_eeg),1,sharex=True,
                                   figsize=tamano_figura)
    figfilt.suptitle('Señales de EEG filtradas')

    for i in range(len(canales_eeg)):
        axfilt[i].plot(t,senales_EEG_filt[canales_eeg[i]][
            inicio_grabacion + int(paso_ms*frec_muestreo*n/1000):
            inicio_grabacion + int((paso_ms*n+tam_ventana_ms)
                                   *frec_muestreo/1000)
            ])
        axfilt[i].margins(x=0)
        axfilt[i].set_ylabel(nombres['Canales EEG'][i])
    
    axfilt[i].set_xlabel('tiempo (ms)')
    figfilt.tight_layout()

# plt.show()
#-----------------------------------------------------------------------------
# Transformada de Fourier de las ventanas
#import scipy

# Periodo de muestreo = 1 / Frecuencia de muestreo
ffs = frec_muestreo
tts = 1/ffs # periodo de muestreo
ll = tam_ventana_ms*frec_muestreo/1000 # numero de muestras
tt = np.arange(0,int(ll))*tts # vector de tiempo
ff = ffs*(np.arange(0, int(ll/2)))/ll # vector de frecuencias
# ff = ffs*(np.arange(0,ll-1))

# para EMG
# senales_emg_tf = [None] * num_canales_emg
senales_emg_filt_tf = [None] * num_canales_emg
i=0
for i in range(num_canales_emg):
    # senales_emg_tf[i] = scipy.fft.fft(senales_EMG_filt[canales_emg[i]][
    #     inicio_grabacion + int(paso_ms*frec_muestreo*n/1000):
    #     inicio_grabacion + int((paso_ms*n+tam_ventana_ms)
    #                            *frec_muestreo/1000)
    #         ].to_numpy())
        
    senales_emg_filt_tf[i] = scipy.fft.fft(senales_EMG_filt[canales_emg[i]][
            inicio_grabacion + int(paso_ms*frec_muestreo*n/1000):
            inicio_grabacion + int((paso_ms*n+tam_ventana_ms)
                                   *frec_muestreo/1000)
            ])

del i
# para EEG
senales_eeg_filt_tf = [None] * num_canales_eeg
for i in range(num_canales_eeg):
    senales_eeg_filt_tf[i] = scipy.fft.fft(senales_EEG_filt[canales_eeg[i]][
        inicio_grabacion + int(paso_ms*frec_muestreo*n/1000):
        inicio_grabacion + int((paso_ms*n + tam_ventana_ms) 
                                *frec_muestreo/1000)
            ])
del i   
# Figuras

# EMG
tamano_figura = (8,10)
fig, ax = plt.subplots(len(canales_emg),1,sharex=True,figsize=tamano_figura)
fig.suptitle('Espectro de frecuencias - señales filtradas de EMG')

for i in range(num_canales_emg):
    pone = 2*np.absolute(senales_emg_filt_tf[i][0:int(ll/2)]/ll)
    ax[i].plot(ff,pone)
    ax[i].margins(x=0)
    ax[i].set_ylabel(nombres['Canales EMG'][i])

ax[i].set_xlabel('Frecuencia (Hz)')
fig.tight_layout()

del i
#EEG
tamano_figura = (8,16)
fig, ax = plt.subplots(len(canales_eeg),1,sharex=True,figsize=tamano_figura)
fig.suptitle('Espectro de frecuencias - señales filtradas de EEG')

for i in range(num_canales_eeg):
    pone = 2*np.absolute(senales_eeg_filt_tf[i][0:int(ll/2)]/ll)
    ax[i].plot(ff,pone)
    ax[i].margins(x=0)
    ax[i].set_ylabel(nombres['Canales EEG'][i])

ax[i].set_xlabel('Frecuencia (Hz)')
fig.tight_layout()

del i
#
# # prueba chirp
# # ll = 2500
# x = np.linspace(0, tam_ventana_ms, int(ll))
# w = scipy.signal.chirp(x, f0=1, f1=1250, t1=2, method='linear')*100
# w = 2*np.sin(2*np.pi*5*x) + np.sin(2*np.pi*600*x) + 1.5*np.sin(2*np.pi*20*x) + 0.5*np.sin(2*np.pi*155*x) + 2*np.sin(2*np.pi*1200*x) 
# w = annots_EMG[canales_emg[0]][
#     inicio_grabacion + int(paso_ms*frec_muestreo*n/1000):
#     inicio_grabacion + int((paso_ms*n + tam_ventana_ms) 
#                             *frec_muestreo/1000)].flatten()

# ff = ffs*(np.arange(0, int(ll/2)))/ll # vector de frecuencias

# wn = 2*frec_corte_emg/frec_muestreo
# filtro = signal.iirfilter(f_orden, wn, btype=b_tipo, 
#                           analog=False, ftype=f_tipo, output='sos')
# w_filt = scipy.signal.sosfilt(filtro_emg,w)

# # plt.plot(w)
# # plt.plot(w_filt)

# w_tf = scipy.fft.fft(w)
# w_filt_tf = scipy.fft.fft(w_filt)

# tamano_figura = (8,4)
# fig, ax = plt.subplots(2,1,sharex=True,figsize=tamano_figura)
# fig.suptitle('Espectro de frecuencias - señales de EMG')

# pone = 2*np.absolute(w_tf[0:int(ll/2)]/ll)
# ax[0].plot(ff,pone)
# ax[0].margins(x=0)
# ax[0].set_ylabel('normal')

# ptwo = 2*np.absolute(w_filt_tf[0:int(ll/2)]/ll)
# ax[1].plot(ff,ptwo)
# ax[1].margins(x=0)
# ax[1].set_ylabel('filtrada')

# ax[1].set_xlabel('Frecuencia (Hz)')
# fig.tight_layout()


#-----------------------------------------------------------------------------
# Sub muestreo
# Ecuación para el sub muestreo: y(n)=x(Mn)
print('Realizando submuestreo de las señales ...')
# Factor de submuestreo EMG
m_emg = 1 #pasa de 2500 a 1250 Hz
# Calcular la frecuencia de sub muestreo
frec_submuestreo_emg = int(frec_muestreo / m_emg)
# Factor de submuestreo EEG
m_eeg = 1 #pasa de 2500 a 250 Hz
# Calcular la frecuencia de sub muestreo
frec_submuestreo_eeg = int(frec_muestreo / m_eeg)

if haceremg:
    # Variable donde guardar el submuestreo
    # senales_EMG_subm = f.HacerSubmuestreo(
    #     num_canales_emg, inicio_grabacion, final_grabacion, m_emg, 
    #     senales_EMG_filt, canales_emg)
    senales_EMG_subm, clases_subm = f.SubmuestreoClases(
        senales_EMG_filt, canales_emg, clases_OH, nombre_clases, inicio_grabacion, 
        final_grabacion, m_emg, filtro_emg)

if hacereeg:
    # Variable donde guardar el submuestreo
    # senales_EEG_subm = f.HacerSubmuestreo(
    #     num_canales_eeg, inicio_grabacion, final_grabacion, m_eeg, 
    #     senales_EEG_filt, canales_eeg)
    senales_EEG_subm, clases_subm = f.SubmuestreoClases(
        senales_EEG_filt, canales_eeg, clases_OH, nombre_clases, inicio_grabacion, 
        final_grabacion, m_eeg, filtro_eeg)

#del nombres
# Contadores
#del j

print('Submuestreo realizado')

#-----------------------------------------------------------------------------
# Enventanado
print('Realizando enventanado ...')
# Variables para determinar el numero de ventanas
tam_ventana_ms = 300 #ms
# paso_ms = tam_ventana_ms
paso_ms = 60 #ms
# Paso de ventanas para la frecencia de muestreo original
paso_ventana_general = int(paso_ms * 0.001 * frec_muestreo)
# Variable para calcular el numero de ventanas totales
num_ventanas = int(((final_grabacion-inicio_grabacion)
    / (frec_muestreo*tam_ventana_ms)) * 1000)

# Para determinar el tamaño de las ventanas
tam_ventana_emg = int(tam_ventana_ms * 0.001 * frec_submuestreo_emg)
# Para el tamaño del paso de la ventana en muestras segundo
paso_ventana_emg = int(paso_ms * 0.001 * frec_submuestreo_emg)
# Las ventanas en este caso en un arreglo de np

# Para determinar el tamaño de las ventanas
tam_ventana_eeg = int(tam_ventana_ms * 0.001 * frec_submuestreo_eeg)
# Para el tamaño del paso de la ventana en muestras segundo
paso_ventana_eeg = int(paso_ms * 0.001 * frec_submuestreo_eeg)

# Filtro para la ventana
ventana_emg = signal.windows.hamming(tam_ventana_emg, sym=True)
ventana_eeg = signal.windows.hamming(tam_ventana_eeg, sym=True)

if haceremg:
    # Para EMG junto con las clases de las ventana en formato OH
    # ventanas_EMG, clases_ventanas_OH = f.HacerEnventanado(
    #     num_ventanas, num_canales_emg, num_clases, tam_ventana_emg, 
    #     paso_ventana_emg, paso_ventana_general, inicio_grabacion, 
    #     senales_EMG_subm, clases_OH, sacar_clases = True)
    ventanas_EMG = np.zeros((num_ventanas, num_canales_emg, tam_ventana_emg))
    # para el numero del canal
    k = 0
    for v in range(num_ventanas):
        for k, canal in enumerate(canales_emg):
            ventanas_EMG[v,k,:] = np.multiply(
                senales_EMG_subm[canal][paso_ventana_emg * v:paso_ventana_emg * v + tam_ventana_emg
                    ], ventana_emg)

if hacereeg & haceremg:
    # ventanas_EEG = f.HacerEnventanado(
    #     num_ventanas, num_canales_eeg, num_clases, tam_ventana_eeg, 
    #     paso_ventana_eeg, paso_ventana_general, inicio_grabacion, 
    #     senales_EEG_subm, clases_OH, sacar_clases = False)
    ventanas_EEG = np.zeros((num_ventanas, num_canales_eeg, tam_ventana_eeg))
    # para el numero del canal
    k = 0
    for v in range(num_ventanas):
        for k, canal in enumerate(canales_eeg):
            ventanas_EEG[v,k,:] = np.multiply(
                senales_EEG_subm[canal][paso_ventana_eeg * v:paso_ventana_eeg * v + tam_ventana_eeg
                    ], ventana_eeg)

if hacereeg & ~haceremg:
    ventanas_EEG, clases_ventanas_OH = f.HacerEnventanado(
        num_ventanas, num_canales_eeg, num_clases, tam_ventana_eeg, 
        paso_ventana_eeg, paso_ventana_general, inicio_grabacion, 
        senales_EEG_subm, clases_OH, sacar_clases = True)

print('Enventanado realizado')

#-----------------------------------------------------------------------------
# Descarte de datos de imaginación motora
print('Descartando ventanas de imaginación motora ...')

if quitar_imaginacion_motora:
    # vector one-hot con la clase de reposo
    clase_reposo = np.asarray(clases_ventanas_OH[0], dtype = int)
    if haceremg:
        # Numero de ventanas para poner en máximo por cada estado de reposo 
        # El 3 corresponde a los segundo en los que se está en reposo
        reclamador =  int(frec_submuestreo_emg*4/paso_ventana_emg)
        (ventanas_EMG, clases_ventanas_OH_EMG) = f.QuitarImaginacionMotora(
            ventanas_EMG, clases_ventanas_OH, clase_reposo, banderas, 
            reclamador)

    if hacereeg:
        # Numero de ventanas para poner en máximo por cada estado de reposo 
        # El 3 corresponde a los segundo en los que se está en reposo
        reclamador =  int(frec_submuestreo_eeg*4/paso_ventana_eeg)
        (ventanas_EEG, clases_ventanas_OH_EEG) = f.QuitarImaginacionMotora(
            ventanas_EEG, clases_ventanas_OH, clase_reposo, banderas, 
            reclamador)

# if not quitar_imaginacion_motora:
#     if haceremg:
#         clases_ventanas_OH_EMG = clases_ventanas_OH
#     if hacereeg:
#         clases_ventanas_OH_EEG = clases_ventanas_OH

print('Ventanas de imaginación motora descartadas')

#-----------------------------------------------------------------------------
# Graficar señales enventanadas submuestreadas
if graficar_ventanas:
    # tam_ventana_ms = 300 #ms
    # paso_ms = 60 #ms
    # n = 124
    # # El t inicia en la n-ésima ventana hasta el fin de esta con las 
    # # muestra de la frecuencia de muestreo correspondientes.
    tb_emg = np.linspace(
        int(inicio_grabacion/frec_muestreo) + paso_ms*n, 
        int(inicio_grabacion/frec_muestreo) + paso_ms*n + tam_ventana_ms, 
        int(frec_submuestreo_emg * tam_ventana_ms * 0.001))
    # Para EMG
    tamano_figura = (8, 10)
    figven, axven = plt.subplots(
        len(canales_emg), 1, sharex=True, figsize=tamano_figura)
    figven.suptitle('Señales de EMG submuestreo')

    for i in range(len(canales_emg)):
        #axven[i].plot(t,ventanas_EMG[:,i,n])
        axven[i].plot(tb_emg, ventanas_EMG[n, i, :])
        axven[i].margins(x=0)
        axven[i].set_ylabel(nombres['Canales EMG'][i])
    
    axven[i].set_xlabel('tiempo (ms)')
    figven.tight_layout()
    
    # Para EEG
    tb_eeg = np.linspace(
        int(inicio_grabacion/frec_muestreo) + paso_ms*n, 
        int(inicio_grabacion/frec_muestreo) + paso_ms*n + tam_ventana_ms, 
        int(frec_submuestreo_eeg * tam_ventana_ms * 0.001))
    
    tamano_figura = (8, 16)
    figven, axven = plt.subplots(
        len(canales_eeg), 1, sharex=True, figsize=tamano_figura)
    figven.suptitle('Señales de EEG submuestreo')

    for i in range(len(canales_eeg)):
        #axven[i].plot(t,ventanas_EMG[:,i,n])
        axven[i].plot(tb_eeg, ventanas_EEG[n, i, :])
        axven[i].margins(x=0)
        axven[i].set_ylabel(nombres['Canales EEG'][i])
    
    axven[i].set_xlabel('tiempo (ms)')
    figven.tight_layout()
    # plt.show()

#-----------------------------------------------------------------------------
# respuesta en frecuencia
# Periodo de muestreo = 1 / Frecuencia de muestreo
ffs = frec_muestreo
tts = 1/ffs # periodo de muestreo
ll = tam_ventana_ms*frec_muestreo/1000 # numero de muestras
tt = np.arange(0,int(ll))*tts # vector de tiempo
ff = ffs*(np.arange(0, int(ll/2)))/ll # vector de frecuencias
# ff = ffs*(np.arange(0,ll-1))

# para EMG
# senales_emg_tf = [None] * num_canales_emg
senales_emg_sub_tf = [None] * num_canales_emg
for i, canal in enumerate(canales_emg):
    # senales_emg_sub_tf[i] = scipy.fft.fft(ventanas_EMG[n, i, :])
    senales_emg_sub_tf[i] = scipy.fft.fft(senales_EMG_subm[canal][
        inicio_grabacion + int(paso_ms*frec_submuestreo_emg*n/1000):
        inicio_grabacion + int((paso_ms*n + tam_ventana_ms) 
                                *frec_submuestreo_emg/1000)
                                ])
        
# para EEG
senales_eeg_sub_tf = [None] * num_canales_eeg
for i, canal in enumerate(canales_eeg):
    # senales_eeg_sub_tf[i] = scipy.fft.fft(ventanas_EEG[n, i, :])
    senales_eeg_sub_tf[i] = scipy.fft.fft(senales_EEG_subm[canal][
        inicio_grabacion + int(paso_ms*frec_submuestreo_eeg*n/1000):
        inicio_grabacion + int((paso_ms*n + tam_ventana_ms) 
                                *frec_submuestreo_eeg/1000)
                                ])
    
# Figuras

# EMG
lls = tam_ventana_ms*frec_submuestreo_emg/1000 # numero de muestras
ffs = frec_submuestreo_emg*(np.arange(0, int(lls/2)))/lls # vector de frecuencias
tamano_figura = (8,10)
fig, ax = plt.subplots(len(canales_emg),1,sharex=True,figsize=tamano_figura)
fig.suptitle('Espectro de frecuencias - señales submuestreadas de EMG')

for i in range(num_canales_emg):
    pone = 2*np.absolute(senales_emg_sub_tf[i][0:int(lls/2)]/lls)
    ax[i].plot(ffs,pone)
    ax[i].margins(x=0)
    ax[i].set_ylabel(nombres['Canales EMG'][i])

ax[i].set_xlabel('Frecuencia (Hz)')
fig.tight_layout()

#EEG
lls = tam_ventana_ms*frec_submuestreo_eeg/1000 # numero de muestras
ffs = frec_submuestreo_eeg*(np.arange(0, int(lls/2)))/lls # vector de frecuencias
tamano_figura = (8,16)
fig, ax = plt.subplots(len(canales_eeg),1,sharex=True,figsize=tamano_figura)
fig.suptitle('Espectro de frecuencias - señales submuestreadas de EEG')

for i in range(num_canales_eeg):
    pone = 2*np.absolute(senales_eeg_sub_tf[i][0:int(lls/2)]/lls)
    ax[i].plot(ffs,pone)
    ax[i].margins(x=0)
    ax[i].set_ylabel(nombres['Canales EEG'][i])

ax[i].set_xlabel('Frecuencia (Hz)')
fig.tight_layout()



# Periodo de muestreo = 1 / Frecuencia de muestreo
ffs = frec_muestreo
tts = 1/ffs # periodo de muestreo
ll = tam_ventana_ms*frec_muestreo/1000 # numero de muestras
tt = np.arange(0,int(ll))*tts # vector de tiempo
ff = ffs*(np.arange(0, int(ll/2)))/ll # vector de frecuencias
# ff = ffs*(np.arange(0,ll-1))
canal = 8

x = np.linspace(0, tam_ventana_ms, int(ll))
w = scipy.signal.chirp(x, f0=1, f1=1250, t1=2, method='linear')*100
w = 2*np.sin(2*np.pi*5*x) + np.sin(2*np.pi*600*x) + 1.5*np.sin(2*np.pi*20*x) + 0.5*np.sin(2*np.pi*155*x) + 2*np.sin(2*np.pi*1200*x) 
w = annots_EEG[canales_eeg[canal]][
    inicio_grabacion + int(paso_ms*frec_muestreo*n/1000):
    inicio_grabacion + int((paso_ms*n + tam_ventana_ms) 
                            *frec_muestreo/1000)].flatten()
# w = annots_EEG[canales_eeg[canal]].flatten()

ff = ffs*(np.arange(0, int(ll/2)))/ll # vector de frecuencias

# ws = ventanas_EEG[n, canal, :]
ws = senales_EEG_subm[canales_eeg[canal]][
    inicio_grabacion + int(paso_ms*frec_submuestreo_eeg*n/1000):
    inicio_grabacion + int((paso_ms*n + tam_ventana_ms) 
                            *frec_submuestreo_eeg/1000)
                            ]
# muestras = int(tam_ventana_ms*frec_submuestreo_eeg/1000)
# ws = signal.resample(w, muestras, axis=0, window = 'hamming', domain='time')

lls = tam_ventana_ms*frec_submuestreo_eeg/1000 # numero de muestras
ffs = frec_submuestreo_eeg*(np.arange(0, int(lls/2)))/lls # vector de frecuencias

wn = 2*frec_corte_eeg/frec_muestreo
filtro = signal.iirfilter(f_orden, wn, btype=b_tipo, 
                          analog=False, ftype=f_tipo, output='sos')
# w_filt = scipy.signal.sosfilt(filtro_eeg, w)
w_filt = senales_EEG_filt[canales_eeg[canal]][
    inicio_grabacion + int(paso_ms*frec_muestreo*n/1000):
    inicio_grabacion + int((paso_ms*n + tam_ventana_ms) 
                            *frec_muestreo/1000)
                            ]

# we = np.multiply(ws, ventana_eeg)
we = ventanas_EEG[n, canal]

# plt.plot(w)
# plt.plot(w_filt)

w_tf = scipy.fft.fft(w)
w_filt_tf = scipy.fft.fft(w_filt)
ws_tf = scipy.fft.fft(ws)
we_tf = scipy.fft.fft(we)


tamano_figura = (10,7)
fig, ax = plt.subplots(4,1,sharex=False,figsize=tamano_figura)
fig.suptitle('Comparativa espectro de frecuencias canal ' + nombres['Canales EEG'][canal] + ' - EEG')

# inf = 0
inf = 0 # 20 Hz
sup = 37 # 120 Hz
# sup = 93 # 310 Hz

pone = 2*np.absolute(w_tf[0:int(ll/2)]/ll)
ax[0].plot(ff[inf:sup],pone[inf:sup])
ax[0].margins(x=0)
ax[0].set_ylabel('sin filtrado')

ptwo = 2*np.absolute(w_filt_tf[0:int(ll/2)]/ll)
ax[1].plot(ff[inf:sup],ptwo[inf:sup])
ax[1].margins(x=0)
ax[1].set_ylabel('filtrada')

pthree = 2*np.absolute(ws_tf[0:int(lls/2)]/lls)
ax[2].plot(ffs[inf:sup],pthree[inf:sup])
ax[2].margins(x=0)
ax[2].set_ylabel('submuestreada')

pfour = 2*np.absolute(we_tf[0:int(lls/2)]/lls)
ax[3].plot(ffs[inf:sup],pthree[inf:sup])
ax[3].margins(x=0)
ax[3].set_ylabel('enventanada')

ax[3].set_xlabel('Frecuencia (Hz)')
fig.tight_layout()

plt.plot()

#-----------------------------------------------------------------------------
# Borrar variables
# Cosas generales
# del inicio_grabacion, final_grabacion, num_clases
# del clases_OH
# del clases_ventanas_OH

# Para submuestreo
# del m_emg, frec_submuestreo_emg
# del m_eeg, frec_submuestreo_eeg
# del senales_EMG_subm, senales_EEG_subm
# Para el enventanado
# del v # contador
# del paso_ms
# del num_ventanas
# del tam_ventana_emg, paso_ventana_emg
# del tam_ventana_eeg, paso_ventana_eeg


"""
#-----------------------------------------------------------------------------
# Dividir datos de entrenamiento y test
print('Dividiendo dataset entre entrenamiento y prueba ...')

# Valor con el porcentaje de prueba
porcen_prueba = 0.2
porcen_validacion = 0.2

# El sufijo _un es que los datos estan desbalanceados
# EMG
if haceremg:
    # Divición entrenamiento y prueba
    EMG_train_un, EMG_test_un, EMG_class_train_un, EMG_class_test_un = train_test_split(
        ventanas_EMG, clases_ventanas_OH_EMG, test_size=porcen_prueba, 
        random_state=1, shuffle=False)
    # Divición entrenamiento y validación
    EMG_train_un, EMG_validation_un, EMG_class_train_un, EMG_class_validation_un = train_test_split(
        EMG_train_un, EMG_class_train_un, test_size=porcen_validacion, 
        random_state=1, shuffle=True)
# EEG
if hacereeg:
    # Divición entrenamiento y prueba
    EEG_train_un, EEG_test_un, EEG_class_train_un, EEG_class_test_un = train_test_split(
        ventanas_EEG, clases_ventanas_OH_EEG, test_size=porcen_prueba, 
        random_state=1, shuffle=False)
    # Divición entrenamiento y validación
    EEG_train_un, EEG_validation_un, EEG_class_train_un, EEG_class_validation_un = train_test_split(
        EEG_train_un, EEG_class_train_un, test_size=porcen_validacion, 
        random_state=1, shuffle=True)

# del ventanas_EMG


print('Divición de dataset completado')

#-----------------------------------------------------------------------------
# Balanceo de base de datos mediate submuestreo aleatoreo
print('Balanceando base de datos ...')
# aplicado unicamente a datos de entrenamiento

clases = np.identity(7, dtype = 'int8')

# La inicialización del balance se hace para conservar las variables
# anteriores y poder compararlas
if balanceo: 
    if haceremg:
        # inicialización
        EMG_train, EMG_class_train = f.Balanceo(
            EMG_train_un, EMG_class_train_un, clase_reposo)
        EMG_validation, EMG_class_validation = f.Balanceo(
            EMG_validation_un, EMG_class_validation_un, clase_reposo)
        EMG_test, EMG_class_test = f.Balanceo(
            EMG_test_un, EMG_class_test_un, clase_reposo)
        # demás clases
        for i in range(num_clases-1):
            EMG_train, EMG_class_train = f.Balanceo(
                EMG_train, EMG_class_train, clases[i])
            EMG_validation, EMG_class_validation = f.Balanceo(
                EMG_validation, EMG_class_validation, clases[i])
            EMG_test, EMG_class_test = f.Balanceo(
                EMG_test, EMG_class_test, clases[i])
    if hacereeg:
        # inicialización
        EEG_train, EEG_class_train = f.Balanceo(
            EEG_train_un, EEG_class_train_un, clase_reposo)
        EEG_validation, EEG_class_validation = f.Balanceo(
            EEG_validation_un, EEG_class_validation_un, clase_reposo)
        EEG_test, EEG_class_test = f.Balanceo(
            EEG_test_un, EEG_class_test_un, clase_reposo)
        # demás clase
        for i in range(num_clases-1):
            EEG_train, EEG_class_train = f.Balanceo(
                EEG_train, EEG_class_train, clases[i])
            EEG_validation, EEG_class_validation = f.Balanceo(
                EEG_validation, EEG_class_validation, clases[i])
            EEG_test, EEG_class_test = f.Balanceo(
                EEG_test, EEG_class_test, clases[i])

    # corespondencia de los datos de prueba
    if haceremg and hacereeg:
        # inicializar
        EMG_test, EEG_test, EMG_class_test = f.BalanceDoble(
            EMG_test_un, EEG_test_un, EMG_class_test_un, clase_reposo)
        # demás clases
        for i in range(num_clases-1):
            EMG_test, EEG_test, EMG_class_test = f.BalanceDoble(
                EMG_test, EEG_test, EMG_class_test, clases[i])
        
        EEG_class_test = EMG_class_test
        
if haceremg:
    num_ventanas_entrenamiento = len(EMG_class_train)
    num_ventanas_validacion = len(EMG_class_validation)
    num_ventanas_prueba = len(EMG_class_test)
if hacereeg & ~haceremg:
    num_ventanas_entrenamiento = len(EEG_class_train)
    num_ventanas_validacion = len(EEG_class_validation)
    num_ventanas_prueba = len(EEG_class_test)

print('Base de datos balanceada')

#-----------------------------------------------------------------------------
# Extraccion de caracteristicas
#-----------------------------------------------------------------------------
print('Calculando de transformación FastICA ...')
# Calculo de FastICA para EMG

# El numero de CI a calcular corresponde a la mitad de los canales usados
num_ci_emg = int(num_canales_emg / 2)
num_ci_emg = 5
# Calcular el ultimo valor de entrenamiento
lim_entre_emg = paso_ventana_emg*num_ventanas_prueba + tam_ventana_emg

# Para asegurar que hayan por lo menos 4 ci ya que de lo contrario no
# se puede aplicar las maxpool de la CNN.
if num_ci_emg < 4:
    num_ci_emg = 4

# Calculo de FastICA para EMG
num_ci_eeg = int(num_canales_eeg / 2)
num_ci_eeg = 16
# Calcular el ultimo valor de entrenamiento
lim_entre_eeg = paso_ventana_eeg*num_ventanas_prueba + tam_ventana_eeg

# Direcciones Para guardar la matriz de transfomación de FastICA
ica_dir = 'S' + sujeto + '/' + sesion +'/FastICA/'

# EEG
ica_path_eeg = ica_dir + "EEG_FastICA.obj"
# EMG
ica_path_emg = ica_dir + "EMG_FastICA.obj"

# Unicamente la matriz de whitening
# EMG
ica_w_path_emg = ica_dir + "EMG_FastICA_whiten.pkl"
# EEG
ica_w_path_eeg = ica_dir + "EEG_FastICA_whiten.pkl"

try:
    os.mkdir(ica_dir)
except OSError:
    print('Error en crear %s' %ica_dir)

if hacer_ica:
    # Aplicar FastICA a las ventanas
    if not ica_antiguo:
        if haceremg:
            # para el vector de todos los datos de entrenamiento
            ica_emg_total = FastICA(
                n_components=num_ci_emg, algorithm='parallel', 
                whiten='arbitrary-variance', fun='exp', max_iter=500)
            ica_emg_total.fit(senales_EMG_subm[:, :lim_entre_emg].T)
            # obtener la matriz de blanqueo
            whiten_emg = ica_emg_total.whitening_
            
            #Guardar matriz de blanqueo
            f.GuardarPkl(whiten_emg, ica_w_path_emg)
            
            # aplicar transformaciones
            EMG_train_ica = f.TransformarICA(
                EMG_train, whiten_emg, num_ventanas_entrenamiento, 
                num_ci_emg, tam_ventana_emg)
            EMG_validation_ica = f.TransformarICA(
                EMG_validation, whiten_emg, num_ventanas_validacion, 
                num_ci_emg, tam_ventana_emg)
            EMG_test_ica = f.TransformarICA(
                EMG_test, whiten_emg, num_ventanas_prueba, 
                num_ci_emg, tam_ventana_emg)
            
        if hacereeg:
            # para el vector de todos los datos de entrenamiento
            ica_eeg_total = FastICA(
                n_components=num_ci_eeg, algorithm='parallel', 
                whiten='arbitrary-variance', fun='exp', max_iter=500)
            ica_eeg_total.fit(senales_EEG_subm[:, :lim_entre_eeg].T)
            # obtener la matriz de blanqueo
            whiten_eeg = ica_eeg_total.whitening_
            
            #Guardar matriz de blanqueo
            f.GuardarPkl(whiten_eeg, ica_w_path_eeg)
            
            # aplicar transformaciones
            EEG_train_ica = f.TransformarICA(
                EEG_train, whiten_eeg, num_ventanas_entrenamiento, 
                num_ci_eeg, tam_ventana_eeg)
            EEG_validation_ica = f.TransformarICA(
                EEG_validation, whiten_eeg, num_ventanas_validacion, 
                num_ci_eeg, tam_ventana_eeg)
            EEG_test_ica = f.TransformarICA(
                EEG_test, whiten_eeg, num_ventanas_prueba, 
                num_ci_eeg, tam_ventana_eeg)
    
    if ica_antiguo and not cargar_ica:
        if haceremg:
            # Calculo de Fast ICA con componentes iguales a la mitad de los 
            # Canales disponbles
            ica_emg = FastICA(
                n_components=num_ci_emg, algorithm='parallel', 
                whiten='arbitrary-variance', fun='exp', max_iter=500)
            ica_emg.fit(senales_EMG_subm[:, :lim_entre_emg].T)
            #f.GuardarPkl(ica_emg, ica_path_emg)
            
            # Guardar
            
            f.GuardarPkl(ica_emg, ica_path_emg)

        if hacereeg:
            # Calculo de Fast ICA con componentes iguales a la mitad de los 
            # Canales disponbles
            ica_eeg = FastICA(
                n_components=num_ci_eeg, algorithm='parallel', 
                whiten='arbitrary-variance', fun='exp', max_iter=500)
            ica_eeg.fit(senales_EEG_subm[:, :lim_entre_eeg].T)
            
            # Guardar
            
            f.GuardarPkl(ica_eeg, ica_path_eeg)

        print('Calculo de transformación a CI completado')
     
if cargar_ica:
    if haceremg:
        ica_emg = f.AbrirPkl(ica_path_emg)
    if hacereeg:
        ica_eeg = f.AbrirPkl(ica_path_eeg)
#-----------------------------------------------------------------------------
# Aplicar FastICA a las ventanas de entrenamiento y prueba.
if ica_antiguo:
    print('Aplicando ICA a las ventanas ...')

    if haceremg:
        # Para entrenamiento
        EMG_train_ica = f.AplicarICA(
            num_ventanas_entrenamiento, num_ci_emg, tam_ventana_emg, ica_emg, 
            EMG_train)
        # Para validacion
        EMG_validation_ica = f.AplicarICA(
            num_ventanas_validacion, num_ci_emg, tam_ventana_emg, ica_emg, 
            EMG_validation)
        # Para prueba
        EMG_test_ica = f.AplicarICA(
            num_ventanas_prueba, num_ci_emg, tam_ventana_emg, ica_emg, 
            EMG_test)

    if hacereeg: 
        # Para entrenamiento
        EEG_train_ica = f.AplicarICA(
            num_ventanas_entrenamiento, num_ci_eeg, tam_ventana_eeg, ica_eeg, 
            EEG_train)
        # Para validacion
        EEG_validation_ica = f.AplicarICA(
            num_ventanas_validacion, num_ci_eeg, tam_ventana_eeg, ica_eeg, 
            EEG_validation)
        # Para prueba
        EEG_test_ica = f.AplicarICA(
            num_ventanas_prueba, num_ci_eeg, tam_ventana_eeg, ica_eeg, 
            EEG_test)

print('ICA aplicado a las ventanas')

#-----------------------------------------------------------------------------
# Graficar señales enventanadas submuestreadas

if graficar_ica:
    # tam_ventana_ms = 300 #ms
    # paso_ms = 60 #ms
    # n = 124
    
    ventana_ica_emg = np.dot(whiten_emg,ventanas_EMG[n, :, :])
    if 'ica_emg' in locals():
        pass
    else:
        ica_emg = FastICA(
            algorithm='parallel', whiten=False, fun='exp', max_iter=500)
    
    ventana_ica_emg = ica_emg.fit_transform(ventana_ica_emg.T).T
    #ventana_ica_emg = ica_emg_total.transform(ventanas_EMG[n, :, :].T).T
    
    # El t inicia en la n-ésima ventana hasta el fin de esta con las 
    # muestra de la frecuencia de muestreo correspondientes.
    # t = np.linspace(
    #     int(inicio_grabacion/frec_muestreo) + paso_ms*n, 
    #     int(inicio_grabacion/frec_muestreo) + paso_ms*n + tam_ventana_ms,
    #     int(frec_submuestreo_emg * tam_ventana_ms * 0.001))

    tamano_figura = (8, 10)
    # Para EMG
    figica, axica = plt.subplots(
        int(num_ci_emg), 1, sharex=True, figsize=tamano_figura)
    figica.suptitle('Descomposición en componentes independientes (IC) mediante FastICA de señales de EMG')
        
    for i in range(int(num_ci_emg)):
        axica[i].plot(tb_emg, EMG_train_ica[n, i, :])
        #axica[i].plot(t, ventana_ica_emg[i, :])
        axica[i].margins(x=0)
        axica[i].set_ylabel('IC ' + str(i+1))
    
    axica[i].set_xlabel('tiempo (ms)')
    figica.tight_layout()
    #plt.show()
    
    # Para EEG
    ventana_ica_eeg = np.dot(whiten_eeg,ventanas_EEG[n, :, :])
    if 'ica_eeg' in locals():
        pass
    else:
        ica_eeg = FastICA(
            algorithm='parallel', whiten=False, fun='exp', max_iter=500)
    
    ventana_ica_eeg = ica_eeg.fit_transform(ventana_ica_eeg.T).T
    #ventana_ica_emg = ica_emg_total.transform(ventanas_EMG[n, :, :].T).T
    
    # El t inicia en la n-ésima ventana hasta el fin de esta con las 
    # muestra de la frecuencia de muestreo correspondientes.
    # t = np.linspace(
    #     int(inicio_grabacion/frec_muestreo) + paso_ms*n, 
    #     int(inicio_grabacion/frec_muestreo) + paso_ms*n + tam_ventana_ms,
    #     int(frec_submuestreo_eeg * tam_ventana_ms * 0.001))

    tamano_figura = (8, 16)
    # Para EMG
    figica, axica = plt.subplots(
        int(num_ci_eeg), 1, sharex=True, figsize=tamano_figura)
    figica.suptitle('Descomposición en componentes independientes (IC) mediante FastICA de señales de EEG')
        
    for i in range(int(num_ci_eeg)):
        axica[i].plot(tb_eeg, EEG_train_ica[n, i, :])
        #axica[i].plot(t, ventana_ica_eeg[i, :])
        axica[i].margins(x=0)
        axica[i].set_ylabel('IC ' + str(i+1))
    
    axica[i].set_xlabel('tiempo (ms)')
    figica.tight_layout()
    plt.show()

#-----------------------------------------------------------------------------
# Eliminar variables
# if haceremg:
#     del num_canales_emg
# if hacereeg:
#     del num_canales_eeg

#-----------------------------------------------------------------------------
# Clasificador
#-----------------------------------------------------------------------------
print('Entrenando o cargando clasificadores')
#-----------------------------------------------------------------------------

clasificador_path = 'S' + sujeto + '/' + sesion +'/clasificadores/'

# Crear punto de control del entrenamiento
# EMG
checkpoint_path_emg = clasificador_path + "EMG_1/cp.ckpt"
checkpoint_dir_emg = os.path.dirname(checkpoint_path_emg)
# EEG
checkpoint_path_eeg = clasificador_path + "EEG_1/cp.ckpt"
checkpoint_dir_eeg = os.path.dirname(checkpoint_path_eeg)

# Sobre entrenamiento
epocas = 1024
lotes = 16

if hacer_clasificador & haceremg:
    # Modelo
    modelo_emg = f.ClasificadorEMG(num_ci_emg, tam_ventana_emg, num_clases)
    modelo_emg.summary()
        
    # Crea un callback que guarda los pesos del modelo
    cp_callback_emg = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path_emg, save_weights_only=True, verbose=1)
    
    # Entrenamiento del modelo 
    cnn_emg = modelo_emg.fit(
        EMG_train_ica, EMG_class_train, shuffle=True, epochs=epocas, 
        batch_size=lotes,  
        validation_data = (EMG_validation_ica, EMG_class_validation), 
        callbacks=[cp_callback_emg])
    
    eva_emg = modelo_emg.evaluate(
        EMG_test_ica, EMG_class_test, verbose=1, return_dict=True)
    print("La precición del modelo: {:5.2f}%".format(
        100 * eva_emg['categorical_accuracy']))

if hacer_clasificador & hacereeg:

    modelo_eeg = f.ClasificadorEEG(num_ci_eeg, tam_ventana_eeg, num_clases)
    modelo_eeg.summary()
 
    # Crea un callback que guarda los pesos del modelo
    cp_callback_eeg = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path_eeg, save_weights_only=True, verbose=1)
    
    # Entrenamiento del modelo
    cnn_eeg = modelo_eeg.fit(
        EEG_train_ica, EEG_class_train, shuffle=True, epochs=epocas, 
        batch_size=lotes,
        validation_data = (EEG_validation_ica, EEG_class_validation), 
        callbacks=[cp_callback_eeg])

    # evaluación del modelo
    eva_eeg = modelo_eeg.evaluate(
        EEG_test_ica, EEG_class_test, verbose=1, return_dict=True)
    print("La precición del modelo: {:5.2f}%".format(
        100 * eva_eeg['categorical_accuracy']))

#-----------------------------------------------------------------------------
if cargar_clasificador & haceremg:
    # Esto es con los puntos de control por lo que deven tener la misma
    # extructura
    # Modelo
    modelo_emg = f.ClasificadorEMG(num_ci_emg, tam_ventana_emg, num_clases)
    # Loads the weights
    modelo_emg.load_weights(checkpoint_path_emg)
    
    eva_emg = modelo_emg.evaluate(EMG_test_ica, EMG_class_test, verbose=1, 
                               return_dict=True)
    print("La precición del modelo: {:5.2f}%".format(
        100 * eva_emg['categorical_accuracy']))

if cargar_clasificador & hacereeg:
    # Esto es con los puntos de control por lo que deven tener la misma
    # extructura
    # Modelo
    modelo_eeg = f.ClasificadorEEG(num_ci_eeg, tam_ventana_eeg, num_clases)
    # Loads the weights
    modelo_eeg.load_weights(checkpoint_path_eeg)
    
    eva_eeg = modelo_eeg.evaluate(
        EEG_test_ica, EEG_class_test, verbose=1, return_dict=True)
    print("La precición del modelo: {:5.2f}%".format(
        100 * eva_eeg['categorical_accuracy']))

print('Clasificador entrenado o cargado')

#-----------------------------------------------------------------------------
# Matrices de confución

# A los datos de validación
prediccion_val_emg = modelo_emg.predict(EMG_validation_ica) 
prediccion_val_eeg = modelo_eeg.predict(EEG_validation_ica) 
    
confusion_val_emg = confusion_matrix(
    argmax(EMG_class_validation, axis=1), argmax(prediccion_val_emg, axis=1))
confusion_val_eeg = confusion_matrix(
    argmax(EEG_class_validation, axis = 1),argmax(prediccion_val_eeg, axis = 1))

# Aplicar a los datos de prueba
prediccion_emg = modelo_emg.predict(EMG_test_ica) 
prediccion_eeg = modelo_eeg.predict(EEG_test_ica) 
    
confusion_emg = confusion_matrix(
    argmax(EMG_class_test, axis=1), argmax(prediccion_emg, axis=1))
confusion_eeg = confusion_matrix(
    argmax(EEG_class_test, axis = 1),argmax(prediccion_eeg, axis = 1))

#-----------------------------------------------------------------------------
# Graficas
#-----------------------------------------------------------------------------
if graficar_clasificador:
    # Imprimir la matriz de confución de los modelos por separado
    cm_emg = pd.DataFrame(
        confusion_emg, index=nombre_clases, columns=nombre_clases)
    cm_emg.index.name = 'Verdadero'
    cm_emg.columns.name = 'Predicho'
    
    cm_eeg = pd.DataFrame(
        confusion_eeg, index=nombre_clases, columns=nombre_clases)
    cm_eeg.index.name = 'Verdadero'
    cm_eeg.columns.name = 'Predicho'
      
    # Para EMG
    plt.figure(figsize = (10,8))
    axcm_emg = sns.heatmap(
        cm_emg,cmap="Blues", linecolor='black', linewidth=1, annot=True, 
        fmt='', xticklabels=nombre_clases, yticklabels=nombre_clases, 
        cbar_kws={"orientation": "vertical"})
    axcm_emg.set_title('Matriz de confusión de CNN para EMG - prueba')
    axcm_emg.set_ylabel('Verdadero', fontsize = 13)
    axcm_emg.set_xlabel('Predicho', fontsize = 13)
    
    # Para EEG
    plt.figure(figsize = (10,8))
    axcm_eeg = sns.heatmap(
        cm_eeg,cmap="Reds", linecolor='black', linewidth=1, annot=True, 
        fmt='', xticklabels=nombre_clases, yticklabels=nombre_clases, 
        cbar_kws={"orientation": "vertical"})
    axcm_eeg.set_title('Matriz de confusión de CNN para EEG - prueba')
    axcm_eeg.set_ylabel('Verdadero', fontsize = 13)
    axcm_eeg.set_xlabel('Predicho', fontsize = 13)
    
    # para validación
    cm_emg_val = pd.DataFrame(
        confusion_val_emg, index=nombre_clases, columns=nombre_clases)
    cm_emg_val.index.name = 'Verdadero'
    cm_emg_val.columns.name = 'Predicho'
    
    cm_eeg_val = pd.DataFrame(
        confusion_val_eeg, index=nombre_clases, columns=nombre_clases)
    cm_eeg_val.index.name = 'Verdadero'
    cm_eeg_val.columns.name = 'Predicho'
      
    # Para EMG
    plt.figure(figsize = (10,8))
    axcm_emg_val = sns.heatmap(
        cm_emg_val,cmap="Blues", linecolor='black', linewidth=1, annot=True, 
        fmt='', xticklabels=nombre_clases, yticklabels=nombre_clases, 
        cbar_kws={"orientation": "vertical"})
    axcm_emg_val.set_title('Matriz de confusión de CNN para EMG - validación')
    axcm_emg_val.set_ylabel('Verdadero', fontsize = 13)
    axcm_emg_val.set_xlabel('Predicho', fontsize = 13)
    
    # Para EEG
    plt.figure(figsize = (10,8))
    axcm_eeg_val = sns.heatmap(
        cm_eeg_val,cmap="Reds", linecolor='black', linewidth=1, annot=True, 
        fmt='', xticklabels=nombre_clases, yticklabels=nombre_clases, 
        cbar_kws={"orientation": "vertical"})
    axcm_eeg_val.set_title('Matriz de confusión de CNN para EEG - validación')
    axcm_eeg_val.set_ylabel('Verdadero', fontsize = 13)
    axcm_eeg_val.set_xlabel('Predicho', fontsize = 13)
    
    if hacer_clasificador:
        
        def grafica_clasifi(axcla, cnn, fontsize=12, senales='EEG', tipo='loss'):
            ''' Para hacer las graficas del entrenamiento rapido '''
            axcla.plot(cnn.history[tipo])
            axcla.plot(cnn.history['val_' + tipo])
            axcla.margins(x=0)
            if tipo == 'loss':
                axcla.set_title('Perdida del modelo de ' + senales)
                axcla.set_ylabel('Perdida')
            elif tipo == 'accuracy':
                axcla.set_title('Precisión del modelo de ' + senales)
                axcla.set_ylabel('Precisión')
            axcla.set_xlabel('Epocas')
            axcla.legend(['Entrenamiento', 'Validación'])

        tamano_figura = (10, 8)
        figcla, ((axcla1,axcla2), (axcla3,axcla4))= plt.subplots(
            nrows=2, ncols=2, figsize=tamano_figura)
        figcla.suptitle(
            'Información sobre el entrenamiento de los clasificadores', fontsize=20)
    
        if haceremg:
            grafica_clasifi(axcla1, cnn_emg, fontsize=12, senales='EMG', tipo='accuracy')
            grafica_clasifi(axcla2, cnn_emg, fontsize=12, senales='EMG', tipo='loss')
        if hacereeg:
            grafica_clasifi(axcla3, cnn_eeg, fontsize=12, senales='EEG', tipo='accuracy')
            grafica_clasifi(axcla4, cnn_eeg, fontsize=12, senales='EEG', tipo='loss')
        plt.tight_layout()
        plt.plot()


#-----------------------------------------------------------------------------
# Unión de los clasificadores
#-----------------------------------------------------------------------------
# combinación de predicciones

# calculo de pesos de acuerdo a la precisión
precision_clase_EMG = np.zeros(len(nombre_clases))
for i in range (len(nombre_clases)):
    if sum(confusion_val_emg[:,i]) == 0:
        precision_clase_EMG[i] = 0
    else:
        precision_clase_EMG[i] = confusion_val_emg[i,i]/sum(confusion_val_emg[:,i])

precision_clase_EEG = np.zeros(len(nombre_clases))
for i in range (len(nombre_clases)):
    if sum(confusion_val_eeg[:,i]) == 0:
        precision_clase_EEG[i] = 0
    else:
        precision_clase_EEG[i] = confusion_val_eeg[i,i]/sum(confusion_val_eeg[:,i])

# calculo del vector de deción eq. 5.45 kuncheva
# u[j] = sum from i=1 to L (w[i,j] * d[i,j]) 

# matriz de pesos
w = [precision_clase_EMG, precision_clase_EEG]

# vector de decición
u = prediccion_emg*w[0] + prediccion_eeg*w[1]

confusion_combinada = confusion_matrix(
    argmax(EEG_class_test, axis = 1),argmax(u, axis = 1))


# antigüo
#prediccion_combinada = (prediccion_eeg + prediccion_emg) / 2 # eq (14.7) bishop
#confusion_combinada = confusion_matrix(
#        argmax(prediccion_combinada, axis=1), argmax(prediccion_emg, axis=1))

#-----------------------------------------------------------------------------
# Graficas
#-----------------------------------------------------------------------------
if graficar_combinada:
    # Imprimir la matriz de confución de los modelos por separado
    cm_combinada = pd.DataFrame(
        confusion_combinada, index=nombre_clases, columns=nombre_clases)

    cm_combinada.index.name = 'Verdadero'
    cm_combinada.columns.name = 'Predicho'
    
    plt.figure(figsize = (10,8))
    
    sns.set(font_scale=1.7)
    axcm_combinada = sns.heatmap(
        cm_combinada,cmap="Purples", linecolor='black', linewidth=1, annot=True, 
        fmt='', xticklabels=nombre_clases, yticklabels=nombre_clases, 
        cbar_kws={"orientation": "vertical"}, annot_kws={"fontsize":21})
    axcm_combinada.set_title('Matriz de confusión de clasificadores combinados', 
                             fontsize = 31)
    axcm_combinada.set_ylabel('Verdadero', fontsize = 21)
    axcm_combinada.set_xlabel('Predicho', fontsize = 21)
    
    # Guardar imagenes
    
#-----------------------------------------------------------------------------
# Para presentar todas las graficas
#-----------------------------------------------------------------------------
if (graficar_senales | graficar_filtradas | graficar_ica 
    | graficar_ventanas | graficar_clasificador | graficar_combinada):
    plt.show()
    
# Revisar la presición de los modelos
print("La precición del modelo de EMG: {:5.2f}%".format(
    100 * eva_emg['categorical_accuracy']))
print("La precición del modelo de EEG: {:5.2f}%".format(
    100 * eva_eeg['categorical_accuracy']))
"""