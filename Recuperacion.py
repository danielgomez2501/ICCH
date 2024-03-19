#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recuperación de los datos eliminados de Rendimiento.csv

Created on Tue Mar 19 13:47:40 2024

@author: alego
"""
import os
import pandas as pd
import Funciones as f

# Las llaves
keys = ['Sujeto', 'Id', 'Tipo de señales', 'Exactitud', 'loss', 'categorical_accuracy']
archivo = 'General/metricas_Combinada.pkl'

rango = [0, 40] # sin contar el 40
sujetos = [2, 7, 8, 15, 21, 22, 23]
directo = 'Parametros/'

for sujeto in sujetos:
    dir_suj = directo + 'Sujeto_' + str(sujeto) +'/'
    for Id in range(rango[0], rango[-1]):
        dir_id = dir_suj + format(Id, '03') +'/'
        direccion = dir_id + archivo # revisar que el formato esté correcto
        resul = f.AbrirPkl(direccion) # revisar que sea un diccionario
        
        # crear el diccionario
        datos ={'Sujeto': sujeto, 
                'Id': [format(Id, '03')], 
                'Tipo de señales': ['Combinada'], 
                'Exactitud': resul['categorical_accuracy']*100, 
                'loss': resul['loss'], 
                'categorical_accuracy': resul['categorical_accuracy']}
        
        # revisa si el diccionario existe
        if not 'resultados' in locals():
            resultados = pd.DataFrame(datos) # revisar que esté correcto
        else:
            rendimiento = pd.DataFrame(datos)
            # concatena en el dataframe
            resultados = pd.concat(
                [resultados, rendimiento], ignore_index=True)
            
resultados.to_csv(directo + 'Recuperacion.csv',  index=False)