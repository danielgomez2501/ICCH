#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisis estadistico de las pruebas

Created on Sat Mar  9 22:47:18 2024

@author: alego
"""

# Rescatado de Evaluyación.py anteriores
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

# Sacar resultados del archivo csv deacuerdo con la id
# ids que comprenden cada experimento
ids_info = {'uno': list(range(0, 5)),
            'dos': list(range(5, 10)),
            'tres': list(range(10, 15)),
            'quattro': list(range(15, 20))
            }
# expe = 'tres'
medida = ['Exactitud', 'loss'] # 'loss', 'Exactitud'
sujetos = [2, 7, 8, 15, 21, 22, 23] # sujetos de entrenamiento

# resultados = pd.DataFrame(
#     columns=['Identificadores', 'Sujeto', 'Exactidud', 'Exactitud_std', 
#              'Loss', 'Loss_std'])
# rendimiento = {expe: dict.fromkeys(sujetos)}

for expe in ids_info.keys():
    ids = ids_info[expe]
    for sujeto in sujetos:
        rendimiento= {
            'Identificadores': [str(ids)],
            'Sujeto': [str(sujeto)]}
        
        for metrica in medida:
            media = metricas.loc[
                ((metricas['Id'].isin(ids_info[expe])) 
                 & (metricas['Sujeto'] == sujeto)), 
                metrica].mean()
            mediana = metricas.loc[
                ((metricas['Id'].isin(ids_info[expe])) 
                 & (metricas['Sujeto'] == sujeto)), 
                metrica].median()
            std = metricas.loc[
                ((metricas['Id'].isin(ids_info[expe])) 
                 & (metricas['Sujeto'] == sujeto)), 
                metrica].std()
            # pone las m
            if metrica == "loss":
                rendimiento['Loss'] = [media]
                rendimiento['Loss_mediana'] = [mediana]
                rendimiento['Loss_std'] = [std]
            else:
                rendimiento[metrica] = [media]
                rendimiento[metrica + '_mediana'] = [mediana]
                rendimiento[metrica + '_std'] = [std]
        # revisa si el diccionario existe
        if not 'resultados' in locals():
            resultados = pd.DataFrame(rendimiento)
        else:
            rendimiento = pd.DataFrame(rendimiento)
            # concatena en el dataframe
            resultados = pd.concat(
                [resultados, rendimiento], ignore_index=True)
        
                
    # analisis de todos:
    rendimiento= {
        'Identificadores': [str(ids)],
        'Sujeto': ['General']}
    
    for metrica in medida:
        media = metricas.loc[
            (metricas['Id'].isin(ids)), 
            metrica].mean()
        mediana = metricas.loc[
            (metricas['Id'].isin(ids)), 
            metrica].median()
        std = metricas.loc[
            (metricas['Id'].isin(ids)), 
            metrica].std()
        
        # pone las m
        if metrica == "loss":
            rendimiento['Loss'] = [media]
            rendimiento['Loss_mediana'] = [mediana]
            rendimiento['Loss_std'] = [std]
        else:
            rendimiento[metrica] = [media]
            rendimiento[metrica + '_mediana'] = [mediana]
            rendimiento[metrica + '_std'] = [std]
    # Concatena en el dataframe
    rendimiento = pd.DataFrame(rendimiento)
    resultados = pd.concat([resultados, rendimiento], ignore_index=True)
# rendimiento utilizará:
#   Experimento
#       Sujetos
#           exactitud: mean, std
#           loss: mean, std
   

# es nesesario sacar los datos teniendo en cuenta el sujeto e id, que indican
# el experimento realizado.

# calcular media, moda y std.

# crear una variable o una tabla donde guardarlas