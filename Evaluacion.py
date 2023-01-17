#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 17:01:26 2022
@author: Daniel Gomez
"""

# Librerias
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ubicación del archivo
directo = 'Parametros/'
# Las metricas
metricas = pd.read_csv(directo + 'Rendimiento.csv')
caracteristicas = pd.read_csv(directo + 'Configuracion.csv')

# sobre las Ids
Ids_info = ['32 canales', '32 canales', '21 canales', 
            'ICA total, 4-10', 'ICA total 5-16', 'ICA total 6-21',
            '21 canales', 'ICA vent 6-21', 'ICA vent 5-16',
            'ICA vent 6-21']
# Rendimiento
# Caculo incertidumbre en las medidas
# rendimiento = dict.fromkeys(['EEG', 'EMG'])
# medida = dict.fromkeys(['Promedio', 'Desviacion tipica'])

# Caculo incertidumbre en las medidas
rendimiento_loss = pd.DataFrame(columns=[
    'Id', 'Tipo', 'Mediana', 'Promedio', 'Desviación típica', 'Calcula ICA', 
    'Numero CI'])

for senal in ['EEG', 'EMG']:
    for Id in caracteristicas['Id'].unique():
        # Calculo de la mediana
        # Calculo de la media aritmetica o promedio
        mediana = metricas.loc[
            ((metricas['Tipo de señales'] == senal)
             & (metricas['Id'] == Id)), 'loss'].median()
        # Calculo de la media aritmetica o promedio
        promedio = metricas.loc[
            ((metricas['Tipo de señales'] == senal)
             & (metricas['Id'] == Id)), 'loss'].mean()
        # ya que se tratan de menos de 30 muestras(n):
        # std = sqrt(sum(dispercion_medida^2)/n-1)
        desviacion = metricas.loc[
            ((metricas['Tipo de señales'] == senal)
             & (metricas['Id'] == Id)), 'loss'].std() / math.sqrt(
            metricas.loc[
                ((metricas['Tipo de señales'] == senal)
                 & (metricas['Id'] == Id)), 'loss'].size)
        # Concatenar en rendimiento
        rendimiento_loss = pd.concat([
            rendimiento_loss,
            pd.DataFrame([[
                Id, senal, mediana, promedio, desviacion, caracteristicas.loc[
                    ((caracteristicas['Tipo de señales'] == senal)
                     & (caracteristicas['Id'] == Id)), 'calcular ica'].unique()[0],
                caracteristicas.loc[
                    ((caracteristicas['Tipo de señales'] == senal)
                     & (caracteristicas['Id'] == Id)), 'numero ci'].unique()[0]]],
                columns=[
                    'Id', 'Tipo', 'Mediana', 'Promedio', 
                    'Desviación típica', 'Calcula ICA', 'Numero CI'])],
            ignore_index=True)

# Caculo incertidumbre en las medidas en exactitud
rendimiento = pd.DataFrame(columns=[
    'Id', 'Tipo', 'Mediana', 'Promedio', 'Desviación típica', 'Calcula ICA', 
    'Numero CI'])

for senal in ['EEG', 'EMG']:
    for Id in caracteristicas['Id'].unique():
        # Calculo de la mediana
        # Calculo de la media aritmetica o promedio
        mediana = metricas.loc[
            ((metricas['Tipo de señales'] == senal)
             & (metricas['Id'] == Id)), 'Exactitud'].median()
        # Calculo de la media aritmetica o promedio
        promedio = metricas.loc[
            ((metricas['Tipo de señales'] == senal)
             & (metricas['Id'] == Id)), 'Exactitud'].mean()
        # ya que se tratan de menos de 30 muestras(n):
        # std = sqrt(sum(dispercion_medida^2)/n-1)
        desviacion = metricas.loc[
            ((metricas['Tipo de señales'] == senal)
             & (metricas['Id'] == Id)), 'Exactitud'].std() / math.sqrt(
            metricas.loc[
                ((metricas['Tipo de señales'] == senal)
                 & (metricas['Id'] == Id)), 'Exactitud'].size)
        # Concatenar en rendimiento
        rendimiento = pd.concat([
            rendimiento,
            pd.DataFrame([[
                Id, senal, mediana, promedio, desviacion, caracteristicas.loc[
                    ((caracteristicas['Tipo de señales'] == senal)
                     & (caracteristicas['Id'] == Id)), 'calcular ica'].unique()[0],
                caracteristicas.loc[
                    ((caracteristicas['Tipo de señales'] == senal)
                     & (caracteristicas['Id'] == Id)), 'numero ci'].unique()[0]]],
                columns=[
                    'Id', 'Tipo', 'Mediana', 'Promedio', 
                    'Desviación típica', 'Calcula ICA', 'Numero CI'])],
            ignore_index=True)

# pruebas = ['Kernel', 'CI', 'Ventana']
# id_pruebas = {
#     'EEG': {'Kernel': [0, 1, 2],
#             'CI': [8, 9, 10],
#             'Ventana': [11, 1, 12]},  # Rehacer las de ventana
#     'EMG': {'Kernel': [0, 1, 2],
#             'CI': [3, 4, 5],
#             'Ventana': [6, 1, 7]}}

# configuracion = {
#     'EEG': {'Kernel': ['(3,3)', '(5,1)', '(5,3)',  # prueba 1
#                        '(5,1)', '(5,1)', '(5,1)',  # prueba 2
#                        '(5,1)', '(5,1)', '(5,1)'],  # prueba 3
#             'CI': ['No', 'No', 'No',  # prueba 1
#                    '10', '16', '20',  # prueba 2
#                    'No', 'No', 'No'],  # prueba 3
#             'Ventana': ['300ms', '300ms', '300ms',  # prueba 1
#                         '300ms', '300ms', '300ms',  # prueba 2
#                         '260ms', '300ms', '500ms']},  # prueba 3
#     'EMG': {'Kernel': ['(3,3)', '(5,1)', '(5,3)',  # prueba 1
#                        '(5,1)', '(5,1)', '(5,1)',  # prueba 2
#                        '(5,1)', '(5,1)', '(5,1)'],  # prueba 3
#             'CI': ['No', 'No', 'No',  # prueba 1
#                    '4', '5', '6',  # prueba 2
#                    'No', 'No', 'No'],  # prueba 3
#             'Ventana': ['300ms', '300ms', '300ms',  # prueba 1
#                         '300ms', '300ms', '300ms',  # prueba 2
#                         '260ms', '300ms', '500ms']},  # prueba 3
#     'Combinación': None}

# metricas['Kernel'] = None
# metricas['CI'] = None
# metricas['Ventana'] = None
# metricas['Prueba'] = None

# # numero de pruebas
# num_pruebas = 13
# for senal in ['EEG', 'EMG']:
#     i = 0
#     # print(senal)
#     for prueba in pruebas:
#         # print(prueba)
#         for Id in id_pruebas[senal][prueba]:
#             # print(Id)
#             metricas.loc[
#                 ((metricas['Tipo de señales'] == senal)
#                     & (metricas['Id'] == Id)), 'Prueba'] = prueba
#             metricas.loc[
#                 ((metricas['Tipo de señales'] == senal)
#                     & (metricas['Id'] == Id)),
#                 'Kernel'] = configuracion[senal]['Kernel'][i]
#             metricas.loc[
#                 ((metricas['Tipo de señales'] == senal)
#                     & (metricas['Id'] == Id)),
#                 'CI'] = configuracion[senal]['CI'][i]
#             metricas.loc[
#                 ((metricas['Tipo de señales'] == senal)
#                     & (metricas['Id'] == Id)),
#                 'Ventana'] = configuracion[senal]['Ventana'][i]
#             # print(configuracion[senal][prueba][i])
#             i += 1

# # Rendimiento
# # Caculo incertidumbre en las medidas
# # rendimiento = dict.fromkeys(['EEG', 'EMG'])
# grupo = dict.fromkeys(pruebas)
# medida = dict.fromkeys(['Promedio', 'Desviacion tipica'])

# # Caculo incertidumbre en las medidas
# rendimiento = pd.DataFrame(columns=[
#     'Id', 'Tipo', 'Mediana', 'Promedio', 'Desviación típica', 'Kernel', 'CI', 'Ventana',
#     'Prueba'])

# for senal in ['EEG', 'EMG']:
#     for prueba in pruebas:
#         for Id in id_pruebas[senal][prueba]:
#             # Calculo de la mediana
#             # Calculo de la media aritmetica o promedio
#             mediana = metricas.loc[
#                 ((metricas['Tipo de señales'] == senal)
#                  & (metricas['Id'] == Id)), 'Exactitud'].median()
#             # Calculo de la media aritmetica o promedio
#             promedio = metricas.loc[
#                 ((metricas['Tipo de señales'] == senal)
#                  & (metricas['Id'] == Id)), 'Exactitud'].mean()
#             # ya que se tratan de menos de 30 muestras(n):
#             # std = sqrt(sum(dispercion_medida^2)/n-1)
#             desviacion = metricas.loc[
#                              ((metricas['Tipo de señales'] == senal)
#                               & (metricas['Id'] == Id)), 'Exactitud'].std() / math.sqrt(
#                 metricas.loc[
#                     ((metricas['Tipo de señales'] == senal)
#                      & (metricas['Id'] == Id)), 'Exactitud'].size)
#             # Concatenar en rendimiento
#             rendimiento = pd.concat([
#                 rendimiento,
#                 pd.DataFrame([[
#                     Id, senal, mediana, promedio, desviacion, metricas.loc[
#                         ((metricas['Tipo de señales'] == senal)
#                          & (metricas['Id'] == Id)), 'Kernel'].unique()[0],
#                     metricas.loc[
#                         ((metricas['Tipo de señales'] == senal)
#                          & (metricas['Id'] == Id)), 'CI'].unique()[0],
#                     metricas.loc[
#                         ((metricas['Tipo de señales'] == senal)
#                          & (metricas['Id'] == Id)), 'Ventana'].unique()[0],
#                     prueba]],
#                     columns=[
#                         'Id', 'Tipo', 'Mediana', 'Promedio', 
#                         'Desviación típica', 'Kernel', 'CI', 'Ventana', 
#                         'Prueba'])],
#                 ignore_index=True)

# # Guardar el Dataframe en formato csv
# rendimiento.to_csv('Parametros/Evaluacion.csv', index=False)


# # Graficas
# def diagrama(datos, x=str, y='Exactitud', titulo='Diagrama de cajas'):
#     """ Función para crear el diagrama de cajas.

#     Parameters
#     ----------
#     datos: DATAFRAME, Contiene los datos para calcular las gráficas
#     x: STR, nombre y datos a tomar del Dataframe para el eje x.
#     y: STR, nombre y datos a tomar del Dataframe para el eje y.
#     titulo: STR, título de la grafica.

#     Returns
#     -------

#     """
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111)
#     sns.boxplot(data=datos, x=x, y=y, ax=ax)
#     ax.set_title(
#         titulo,
#         fontsize=16)

# for senal in ['EEG', 'EMG']:
#     prueba: str
#     for prueba in pruebas:
#         diagrama(
#             metricas.loc[
#                 ((metricas['Tipo de señales'] == senal)
#                  & ((metricas['Id'] == id_pruebas[senal][prueba][0])
#                     | (metricas['Id'] == id_pruebas[senal][prueba][1])
#                     | (metricas['Id'] == id_pruebas[senal][prueba][2]))
#                  )],
#             x=prueba,
#             titulo='Diagrama de Cajas, prueba ' + prueba + ' - ' + senal)


# # Caculo incertidumbre en las medidas
# final = pd.DataFrame(columns=[
#     'Tipo', 'Mediana', 'Promedio', 'Desviación típica'])
# # para la segunda fase:
# for senales in ['EEG', 'EMG', 'Combinación']:
#     # mediana
#     mediana = metricas.loc[
#        ((metricas['Tipo de señales'] == senales)
#         & ((metricas['Id'] == 1)
#             |(metricas['Id'] == 13)
#            | (metricas['Id'] == 14)
#            | (metricas['Id'] == 15))), 'Exactitud'].median()
#     # promedio
#     promedio = metricas.loc[
#        ((metricas['Tipo de señales'] == senales)
#         & ((metricas['Id'] == 1)
#             |(metricas['Id'] == 13)
#            | (metricas['Id'] == 14)
#            | (metricas['Id'] == 15))), 'Exactitud'].mean()
#     # desviaciòn
#     desviacion = metricas.loc[
#        ((metricas['Tipo de señales'] == senales)
#         & ((metricas['Id'] == 1)
#             |(metricas['Id'] == 13)
#            | (metricas['Id'] == 14)
#            | (metricas['Id'] == 15))), 'Exactitud'].std() / math.sqrt(metricas.loc[
#               ((metricas['Tipo de señales'] == senales)
#                & ((metricas['Id'] == 1)
#                    |(metricas['Id'] == 13)
#                   | (metricas['Id'] == 14)
#                   | (metricas['Id'] == 15))), 'Exactitud'].size)
#     final = pd.concat([
#         final,
#         pd.DataFrame([[
#             senales, mediana, promedio, desviacion]],
#             columns=[
#                 'Tipo', 'Mediana', 'Promedio', 'Desviación típica'])],
#         ignore_index=True)
  
# diagrama(
#    metricas.loc[
#       ((metricas['Id'] == 1)
#           |(metricas['Id'] == 13)
#           | (metricas['Id'] == 14)
#           | (metricas['Id'] == 15))],
#     x='Tipo de señales',
#     titulo='Diagrama de Cajas, interfaz planteada ')

# data = metricas.loc[
#    ((metricas['Id'] == 1)
#        |(metricas['Id'] == 13)
#        | (metricas['Id'] == 14)
#        | (metricas['Id'] == 15))]

# plt.show()
# # Para calcular la incertidumbre de la medida del rendimiento
# # cálculo de la media aritmetica / promedio
# medida['Promedio'] = np.mean(datos)
# # ya que se tratan de menos de 30 muestras(n):
# # std = sqrt(sum(dispercion_medida^2)/n-1)
# medida['Desviacion tipica'] = np.std(datos, ddof=1)/np.sqrt(6)

