#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 17:01:26 2022

@author: Daniel Gomez
"""

# Librerias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ubicación del archivo
directo = 'Parametros/Rendimiento.csv'
# Las metricas
metricas = pd.read_csv(directo)

# nombres que darle a las diferentes pruebas
metricas['Info'] = ''
metricas.loc[metricas['Id'] == 0, 'Info'] = 'Kernel (3,3)'
metricas.loc[metricas['Id'] == 1, 'Info'] = 'Kernel (5,1)'
metricas.loc[metricas['Id'] == 2, 'Info'] = 'Kernel (5,3)'

combinada = metricas[metricas['Tipo de señales'] == 'Combinada']
eeg = metricas[metricas['Tipo de señales'] == 'EEG']
emg = metricas[metricas['Tipo de señales'] == 'EMG']

metricas['Id'].max()
figco = plt.figure(figsize = (10,8))
axco = figco.add_subplot(111)
sns.boxplot(data=combinada, x="Info", y="Exactitud", ax = axco)
axco.set_title(
    'Diagramas de caja para la exactitud de clasificadores combinados', 
    fontsize = 16)

figee = plt.figure(figsize = (10,8))
axee = figee.add_subplot(111)
sns.boxplot(data=eeg, x="Info", y="Exactitud", ax = axee)
axee.set_title(
    'Diagramas de caja para la exactitud de clasificadores de EEG', 
    fontsize = 16)

figem = plt.figure(figsize = (10,8))
axem = figem.add_subplot(111)
sns.boxplot(data=emg, x="Info", y="Exactitud", ax = axem)
axem.set_title(
    'Diagramas de caja para la exactitud de clasificadores de EMG', 
    fontsize = 16)

plt.show()
# plt.boxplot(combinada[combinada['Id']==0]['Exactitud'])
