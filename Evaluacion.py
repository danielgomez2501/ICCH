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
directo = 'Parametros/Rendimiento.csv'
# Las metricas
metricas = pd.read_csv(directo)

pruebas = ['Kernel', 'CI', 'Ventana']
id_pruebas = {
    'EEG': {'Kernel': [0, 1, 2],
            'CI': [8, 9, 10],
            'Ventana': [11, 1]},  # Rehacer las de ventana
    'EMG': {'Kernel': [0, 1, 2],
            'CI': [3, 4, 5],
            'Ventana': [6, 1, 7]}}

configuracion = {
    'EEG': {'Kernel': ['(3,3)', '(5,1)', '(5,3)',  # prueba 1
                       '(5,1)', '(5,1)', '(5,1)',  # prueba 2
                       '(5,1)', '(5,1)', '(5,1)'],  # prueba 3
            'CI': ['No', 'No', 'No',  # prueba 1
                   '10', '16', '20',  # prueba 2
                   'No', 'No', 'No'],  # prueba 3
            'Ventana': ['300ms', '300ms', '300ms',  # prueba 1
                        '300ms', '300ms', '300ms',  # prueba 2
                        '260ms', '300ms', '500ms']},  # prueba 3
    'EMG': {'Kernel': ['(3,3)', '(5,1)', '(5,3)',  # prueba 1
                       '(5,1)', '(5,1)', '(5,1)',  # prueba 2
                       '(5,1)', '(5,1)', '(5,1)'],  # prueba 3
            'CI': ['No', 'No', 'No',  # prueba 1
                   '4', '5', '6',  # prueba 2
                   'No', 'No', 'No'],  # prueba 3
            'Ventana': ['300ms', '300ms', '300ms',  # prueba 1
                        '300ms', '300ms', '300ms',  # prueba 2
                        '260ms', '300ms', '500ms']},  # prueba 3
    'Combinada': None}

metricas['Kernel'] = None
metricas['CI'] = None
metricas['Ventana'] = None
metricas['Prueba'] = None

# numero de pruebas
num_pruebas = 13
for senal in ['EEG', 'EMG']:
    i = 0
    # print(senal)
    for prueba in pruebas:
        # print(prueba)
        for Id in id_pruebas[senal][prueba]:
            # print(Id)
            metricas.loc[(
                (metricas['Tipo de señales'] == senal) 
                & (metricas['Id'] == Id)), 'Prueba'] = prueba
            metricas.loc[(
                (metricas['Tipo de señales'] == senal) 
                & (metricas['Id'] == Id)), 
                'Kernel'] = configuracion[senal]['Kernel'][i]
            metricas.loc[(
                (metricas['Tipo de señales'] == senal) 
                & (metricas['Id'] == Id)), 
                'CI'] = configuracion[senal]['CI'][i]
            metricas.loc[(
                (metricas['Tipo de señales'] == senal) 
                & (metricas['Id'] == Id)), 
                'Ventana'] = configuracion[senal]['Ventana'][i]
            # print(configuracion[senal][prueba][i])
            i += 1
            
# Rendimiento
# Caculo incertidumbre en las medidas
# rendimiento = dict.fromkeys(['EEG', 'EMG'])
grupo = dict.fromkeys(pruebas)
medida = dict.fromkeys(['Promedio', 'Desviacion tipica'])

# Caculo incertidumbre en las medidas
rendimiento = pd.DataFrame(columns=[
    'Id', 'Tipo', 'Promedio', 'Desviación típica', 'Kernel', 'CI', 'Ventana', 
    'Prueba'])

for senal in ['EEG', 'EMG']:
    for prueba in pruebas:
        for Id in id_pruebas[senal][prueba]:
            # print (Id)
            promedio = metricas.loc[
                ((metricas['Tipo de señales'] == senal) 
                 & (metricas['Id'] == Id)), 'Exactitud'].mean()
            
            desviacion = metricas.loc[
                ((metricas['Tipo de señales'] == senal) 
                 & (metricas['Id'] == Id)), 'Exactitud'].std() / math.sqrt(
                     metricas.loc[
                         ((metricas['Tipo de señales'] == senal) 
                          & (metricas['Id'] == Id)), 'Exactitud'].size)
            # Concatenar en rendimiento
            rendimiento = pd.concat([rendimiento, 
                    pd.DataFrame([[
                        Id, senal, promedio, desviacion, metricas.loc[
                        ((metricas['Tipo de señales'] == senal) 
                         & (metricas['Id'] == Id)), 'Kernel'].unique()[0], 
                        metricas.loc[
                        ((metricas['Tipo de señales'] == senal) 
                        & (metricas['Id'] == Id)), 'CI'].unique()[0],
                        metricas.loc[
                        ((metricas['Tipo de señales'] == senal) 
                        & (metricas['Id'] == Id)), 'Ventana'].unique()[0],
                        prueba]], 
                    columns=[
                        'Id', 'Tipo', 'Promedio', 'Desviación típica', 'Kernel', 
                        'CI', 'Ventana', 'Prueba'])], 
                ignore_index=True)
        
             
# Graficas
def diagrama(datos, x=str, y='Exactitud', titulo='Diagrama de cajas'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    sns.boxplot(data=datos, x=x, y=y, ax=ax)
    ax.set_title(
        titulo,
        fontsize=16)
    plt.show()
    
    
# # agregar columnas que indican los parametros de la interfaz
# senales = ['Combinada', 'EEG', 'EMG']
# n_ids = metricas['Id'].max()
# kernels = ['(3,3)', '(5,1)', '(5,3)']  # Los disponibles
# kernel = {'EEG': '(5,3)', 'EMG': '(5,1)',
#           'Combinada': 'EEG: (5,3), EMG: (5,1)'}  # Los elegidos
# ci = {'EEG': ['10', '16', '20'],
#       'EMG': ['4', '5', '6']}
# ventanas = ['260 ms', '500 ms']  # 300ms se saca de otras pruebas
# pruebas = ['Kernel', 'CI', 'Ventana', 'Com']

# # nombres que darle a las diferentes pruebas
# metricas['Kernel'] = None
# metricas['CI'] = None
# metricas['Ventana'] = None
# metricas['Com'] = 'Variar'
# metricas['Prueba'] = None

# # Datos para primera fase hasta id = 002
# for i in range(3):
#     metricas.loc[
#         ((metricas['Id'] == i) & (metricas['Tipo de señales'] != 'Combinada')),
#         'Kernel'] = kernels[i]
#     metricas.loc[
#         ((metricas['Id'] == i) & (metricas['Tipo de señales'] == 'Combinada')),
#         'Kernel'] = 'EEG: ' + kernels[i] + ', EMG: ' + kernels[i]
#     metricas.loc[metricas['Id'] == i, 'CI'] = 'No'
#     metricas.loc[metricas['Id'] == i, 'Ventana'] = '300 ms'
#     metricas.loc[metricas['Id'] == i, 'Prueba'] = 'Kernel'

# # Datos para segunda fase hata id = 005
# for i in range(3):
#     for tipo in senales:
#         metricas.loc[
#             ((metricas['Id'] == i + 3) & (metricas['Tipo de señales'] == tipo)),
#             'Kernel'] = kernel[tipo]
#         if tipo != 'Combinada':
#             metricas.loc[
#                 ((metricas['Id'] == i + 3) & (metricas['Tipo de señales'] == tipo)),
#                 'CI'] = ci[tipo][i]
#         else:
#             metricas.loc[
#                 ((metricas['Id'] == i + 3) & (metricas['Tipo de señales'] == 'Combinada')),
#                 'CI'] = 'EEG: ' + ci['EEG'][i] + ', EMG: ' + ci['EMG'][i]
#     metricas.loc[metricas['Id'] == i + 3, 'Ventana'] = '300 ms'
#     metricas.loc[metricas['Id'] == i + 3, 'Prueba'] = 'CI'

# # Datos para segunda fase hata id = 007
# for i in range(2):
#     for tipo in senales:
#         metricas.loc[
#             ((metricas['Id'] == i + 6) & (metricas['Tipo de señales'] == tipo)),
#             'Kernel'] = kernel[tipo]

#         metricas.loc[
#             ((metricas['Id'] == i + 6) & (metricas['Tipo de señales'] == tipo)),
#             'CI'] = 'No'

#     metricas.loc[metricas['Id'] == i + 6, 'Ventana'] = ventanas[i]
#     metricas.loc[metricas['Id'] == i + 6, 'Prueba'] = 'Ventana'

# for tipo in senales:
#     metricas.loc[
#         ((metricas['Id'] == 8) & (metricas['Tipo de señales'] == tipo)),
#         'Kernel'] = '(5,1)'

# metricas.loc[
#     ((metricas['Id'] == 8) & (metricas['Tipo de señales'] == 'EEG')),
#     'CI'] = '20'
# metricas.loc[
#     ((metricas['Id'] == 8) & (metricas['Tipo de señales'] == 'EMG')),
#     'CI'] = 'No'
# metricas.loc[
#     ((metricas['Id'] == 8) & (metricas['Tipo de señales'] == 'Combinada')),
#     'CI'] = 'EEG: 20, EMG: No'

# metricas.loc[metricas['Id'] == 8, 'Ventana'] = '300 ms'
# metricas.loc[metricas['Id'] == 8, 'Prueba'] = 'Com'


# Graficas
def diagrama(datos, x=str, y='Exactitud', titulo='Diagrama de cajas'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    sns.boxplot(data=datos, x=x, y=y, ax=ax)
    ax.set_title(
        titulo,
        fontsize=16)
    plt.show()


for prueba in pruebas:
    datos = metricas.loc[metricas['Prueba'] == prueba]
    for tipo in senales:
        pass
        # diagrama(
        #     datos.loc[datos['Tipo de señales'] == tipo], x=prueba,
        #     titulo='Diagrama de cajas de exactitud de clasificador - ' + tipo)

# Caculo incertidumbre en las medidas
rendimiento = dict.fromkeys(senales)
grupo = dict.fromkeys(pruebas)
configuracion = dict()
medida = dict.fromkeys(['Promedio', 'Desviacion tipica'])

for tipo in senales:
    datos = metricas.loc[metricas['Tipo de señales'] == tipo]
    print('El tipo: ' + tipo)
    for prueba in pruebas:
        print('La prueba: ' + prueba)
        parametros = datos.loc[datos['Prueba'] == prueba][prueba].unique()
        configuracion = dict.fromkeys(parametros)
        for conf in parametros:
            print('La configuraciòn: ' + conf)
            # calculo de la media aritmetica / promedio
            medida['Promedio'] = datos.loc[
                ((datos['Prueba'] == prueba) & (datos[prueba] == conf))
            ]['Exactitud'].mean()
            # ya que se tratan de menos de 30 muestras(n):
            # std = sqrt(sum(dispercion_medida^2)/n-1)
            medida['Desviacion tipica'] = datos.loc[
                ((datos['Prueba'] == prueba) 
                 & (datos[prueba] == conf))][
                     'Exactitud'].std() / math.sqrt(datos.loc[
                         ((datos['Prueba'] == prueba) 
                          & (datos[prueba] == conf))]['Exactitud'].size)
            # agregar a dicionario de configuraciòn
            configuracion[conf] = medida
        grupo[prueba] = configuracion
    rendimiento[tipo] = grupo

# Caculo incertidumbre en las medidas
rendimiento = pd.DataFrame(columns=[
    'Id', 'Tipo', 'Promedio', 'Desviación típica', 'Configuración', 'Prueba'])

for tipo in senales:
    datos = metricas.loc[metricas['Tipo de señales'] == tipo]
    print('El tipo: ' + tipo)
    for prueba in pruebas:
        print('La prueba: ' + prueba)
        parametros = datos.loc[datos['Prueba'] == prueba][prueba].unique()
        configuracion = dict.fromkeys(parametros)
        for conf in parametros:
            print('La configuraciòn: ' + conf)
            # calculo de la media aritmetica / promedio
            promedio = datos.loc[
                ((datos['Prueba'] == prueba) & (datos[prueba] == conf))
            ]['Exactitud'].mean()
            # ya que se tratan de menos de 30 muestras(n):
            # std = sqrt(sum(dispercion_medida^2)/n-1)
            desviacion = datos.loc[
                             ((datos['Prueba'] == prueba) & (datos[prueba] == conf))]['Exactitud'
                         ].std() / math.sqrt(datos.loc[
                                                 ((datos['Prueba'] == prueba) & (datos[prueba] == conf))][
                                                 'Exactitud'].size)
            # revisar la Id
            Id = datos.loc[
                ((datos['Prueba'] == prueba) & (datos[prueba] == conf))
            ]['Id'].unique()[0]
            # agregar la configuración al dataframe
            rendimiento = pd.concat([rendimiento, pd.DataFrame.from_dict({
                'Id': [Id], 'Tipo': [tipo], 'Promedio': [promedio],
                'Desviación típica': [desviacion], 'Configuración': [conf],
                'Prueba': [prueba]}, orient='columns')], ignore_index=True)
            # rendimiento.append({
            #     'Id': Id, 'Tipo': tipo, 'Promedio': promedio,
            #     'Desviación típica': desviacion, 'Cònfiguración': conf,
            #     'Prueba': prueba}, ignore_index=True)

# # Para calcular la incertidumbre de la medida del rendimiento
# # calculo de la media aritmetica / promedio
# medida['Promedio'] = np.mean(datos)
# # ya que se tratan de menos de 30 muestras(n):
# # std = sqrt(sum(dispercion_medida^2)/n-1)
# medida['Desviacion tipica'] = np.std(datos, ddof=1)/np.sqrt(6)
