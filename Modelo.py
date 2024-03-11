# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 12:54:34 2022

@author: Daniel
"""

# -----------------------------------------------------------------------------
# Librerías
# -----------------------------------------------------------------------------
print('Cargando Librerías ...')

# General
import numpy as np
# Para generar multiples hilos
import threading
# Para generar multiples procesos
# from multiprocessing import process
# dividir la base de datos
from sklearn.model_selection import train_test_split
# uso de K-folds
# from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
# Para matrices de confusión
from sklearn.metrics import confusion_matrix
# para cargar modelos
from tensorflow.keras.models import load_model
# from tensorflow.math import argmax  # para convertir de one hot a un vector
# Mis funciones
import Funciones as f
# Interactuar con el sistema operativo
from os.path import exists

# extración de caracteristicas
from mne.decoding import CSP
# seleccion de caracteristicas
from niapy.task import Task
from niapy.algorithms.basic import ParticleSwarmOptimization
from sklearn.svm import SVC
# pandas
import pandas as pd

# -----------------------------------------------------------------------------
# Procesamiento
# -----------------------------------------------------------------------------

class Modelo(object):
    """Clase Modelo

    Contiene todos los parámetros de la HBCI, con los métodos
    "Entrenamiento" y "Carga" se puede efectuar el entrenamiento o la
    carga de los parámetros de la interfaz.
    """

    def __init__(self):
        # super(ClassName, self).__init__()
        # Para la ejecución en hilos
        self.lock = threading.Lock()
        # Hacer seguimiento al progreso
        self.progreso = {'EMG': 0.0, 'EEG': 0.0, 'General': 0.0}
        # Argumentos
        # Parámetros generales, valores predeterminados
        self.directorio = 'Dataset'
        self.sujeto = 2
        self.nombres = dict()
        self.nombre_clases = list
        self.f_tipo = 'butter'
        self.b_tipo = 'bandpass'
        self.frec_corte = {'EMG': np.array([8, 520]), 'EEG': np.array([6, 24])}
        self.f_orden = 5
        self.m = {'EMG': 2, 'EEG': 2}
        self.tam_registro_s = 11 # en s
        self.tam_ventana_ms = 300  # en ms
        self.paso_ms = 60  # en ms
        self.descarte_ms = {
            'EMG': {'Activo': 100, 'Reposo': 3000},
            'EEG': {'Activo': 100, 'Reposo': 3000}}
        self.reclamador_ms = {
            'EMG': {'Activo': 3500, 'Reposo': 1000},
            'EEG': {'Activo': 3500, 'Reposo': 1000}}
        self.porcen_prueba = 0.2
        self.porcen_validacion = 0.1
        self.calcular_csp = {'EMG': False, 'EEG': False}
        self.calcular_ica = {'EMG': False, 'EEG': False} # eliminar
        self.num_ci = {'EMG': 4, 'EEG': 7}
        self.carac_seleccionadas = False # prebiamente elegidas
        self.caracteristicas = {'EMG': None, 'EEG': None}
        self.caracteristicascanal = {'EMG': None, 'EEG': None}
        self.parcial = {'EMG': None, 'EEG': None}
        self.epocas = 1024
        self.lotes = 128
        self.divisiones = 5 # Divisiones de k folds
        self.balancear = True
        self.agrupar_ventanas = False # pos procesamiento
        # Calculados a partir de los parámetros generales
        self.num_clases = int  # 7 clases
        self.canales = dict.fromkeys(['EMG', 'EEG'])  # nombres para los canales de EEG y EMG
        self.num_canales = dict.fromkeys(['EMG', 'EEG'])
        # Se calculan a medida que se va ejecutando el código
        # Diccionario para frecuencias de sub muestreo
        self.frec_submuestreo = dict.fromkeys(['EMG', 'EEG'])
        # Diccionario para el enventanado
        self.tam_ventana = dict.fromkeys(['EMG', 'EEG'])
        self.paso_ventana = dict.fromkeys(['EMG', 'EEG'])
        self.num_ventanas = {'EMG': dict.fromkeys([
                                'Entrenamiento', 'Validacion', 'Prueba']), 
                             'EEG': dict.fromkeys([
                                 'Entrenamiento', 'Validacion', 'Prueba'])}
        # Variables a guardar
        self.filtro = dict.fromkeys(['EMG', 'EEG'])
        self.csp = dict.fromkeys(['EEG', 'EMG'])
        self.ica_total = dict.fromkeys(['EMG', 'EEG'])
        self.whiten = dict.fromkeys(['EMG', 'EEG'])
        # se cambia para las versiones finales
        # self.modelo = dict.fromkeys(['EMG', 'EEG']) 
        self.modelo = None
        self.confusion = {'EMG': dict.fromkeys(['Validacion', 'Prueba']),
                          'EEG': dict.fromkeys(['Validacion', 'Prueba']),
                          'Combinada': dict.fromkeys(['Validacion', 'Prueba'])}
        self.metricas = dict.fromkeys(['EMG', 'EEG', 'Combinada'])
        self.exactitud = dict.fromkeys(['EMG', 'EEG', 'Combinada'])
        # Para revisar las predicciones y la combinación
        self.prediccion = dict.fromkeys(['EMG', 'EEG', 'Combinada'])
        self.class_test = None
        self.test = dict.fromkeys(['EMG', 'EEG'])
        # Calculados luego del entrenamiento
        self.w = None
        # Para la carga de datos
        self.direccion = str
        self.ubi = str  # el formato es '###'
        # Para dividir los registros
        self.registros_id = dict.fromkeys(['train', 'val', 'test'])
        self.registros_id['train'] = []
        self.registros_id['val'] = []
        self.registros_id['test'] = []

        # Datos y canales a utilizar predeterminados
        # 'EMG_ref'
        self.nombres['EMG'] = [
            'EMG_1', 'EMG_2', 'EMG_3', 'EMG_4', 'EMG_5', 'EMG_6'
        ]
        # 10-20
        self.nombres['EEG'] = [
            'FP1', 'F7', 'F3', 'Fz', 'T7', 'C3', 'Cz', 'P7', 'P3', 'Pz',
            'FP2', 'F4', 'F8', 'C4', 'T8', 'P4', 'P8', 'O1', 'Oz', 'O2'
        ]
        # Sobre corteza motora
        # nombres['EEG'] = [
        #     'FC5', 'FC3', 'FC1', 'Fz', 'FC2', 'FC4', 'FC6', 'C5', 'C3','C1',
        #     'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
        #     'Cz'
        #     ]
        self.nombre_clases = [
            'Click izq.', 'Click der.', 'Izquierda', 'Derecha',
            'Arriba', 'Abajo', 'Reposo'
        ]
        # self.caracteristicas['EMG'] = dict()
            # [
            # 'potencia de banda', 'cruce por cero', 'desviacion estandar',
            # 'varianza', 'entropia', 'media', 'rms', 'energia', 
            # 'longitud de onda', 'integrada', 'ssc'
            # ]
        # self.caracteristicas['EEG'] = dict()
            # [
            # 'potencia de banda', 'cruce por cero', 'desviacion estandar',
            # 'varianza', 'entropia', 'media', 'rms', 'energia', 
            # 'longitud de onda', 'integrada', 'ssc'
            # ]
        
        # la configuración general del clasificador
        self.configuracion = {
            'directorio': self.directorio, 'sujeto': self.sujeto,
            'nombres': self.nombres, 'nombre clases': self.nombre_clases,
            'filtro': self.f_tipo, 'banda de filtro': self.b_tipo,
            'frecuencia corte': self.frec_corte,
            'orden del filtro': self.f_orden, 'm': self.m,
            'tamaño ventana': self.tam_ventana_ms, 'paso': self.paso_ms,
            'descarte': self.descarte_ms, 'reclamador': self.reclamador_ms,
            'porcentaje prueba': self.porcen_prueba,
            'porcentaje validacion': self.porcen_validacion,
            'calcular ica': self.calcular_ica, 'numero ci': self.num_ci,
            'epocas': self.epocas, 'lotes': self.lotes}

    def ActualizarProgreso(self, tipo, progreso):
        """Metodo ActualizarProgreso:

        Realiza la actualización del progreso de la carga o
        entrenamiento de la interfaz de forma que no haya errores
        cuando sean modificadas por los hilos.

        Parameters
        ----------
        tipo: STR, el tipo de señales para entrenamiento, puede ser
            'EMG' o 'EEG'
        progreso: FLOAT, porcentaje de progreso

        Returns
        -------
        """
        self.lock.acquire()
        self.progreso[tipo] = progreso
        if tipo == 'General':
            pass
        else:
            self.progreso['General'] = (
                self.progreso['EMG'] + self.progreso['EEG']) / 2
        self.lock.release()

    def ObtenerParametros(self, sujeto):
        """Para obtener los párametros de la interfaz a entrenar
        """
        # Definiciones temporales de los datos
        # cambiar a la hora de integrarlo en la interfaz
        # directorio = 'G:\Proyectos\ICCH\Dataset'
        directorio = '/home/alego/Proyectos/Dataset'
        # Datos y canales a utilizar
        nombres = dict()
        # 'EMG_ref'
        nombres['EMG'] = [
            'EMG_1', 'EMG_2', 'EMG_3', 'EMG_4', 'EMG_5', 'EMG_6'
        ]
        
        # nombres['EMG'] = ['EMG_1', 'EMG_2', 'EMG_4', 'EMG_6', 'EMG_ref']
        
        
        # # Sobre corteza motora
        # # Corteza motora de acuerdo a [1] - 32 canales
        # nombres['EEG'] = [
        #     'FP1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC3', 'FC1', 'FC2',
        #     'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3',
        #     'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8',
        #     'O1', 'O2']
        # Corteza motora de acuerdo a [4] - 22 canales
        # nombres['EEG'] = [
        #     'Fz', 'FC3', 'FC1', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
        #     'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz',
        #     'P2', 'POz']
        
        # # Corteza motora reducción de [6] - 20 canales
        nombres['EEG'] = [
            'FC1', 'FC3', 'FC5', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 
            'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'
            ]
        
        # Todos los disponibles
        # nombres['EEG'] = [
        #     'FP1', 'AF7', 'AF3', 'AFz', 'F7', 'F5', 'F3', 'F1', 'Fz', 'FT7', 
        #     'FC5', 'FC3', 'FC1', 'T7', 'C5', 'C3', 'C1', 'Cz', 'TP7', 'CP5',
        #     'CP3', 'CP1', 'CPz', 'P7', 'P5', 'P3', 'P1', 'Pz', 'PO7', 'PO3',
        #     'POz', 'FP2', 'AF4', 'AF8', 'F2', 'F4', 'F6', 'F8', 'FC2', 'FC4',
        #     'FC6', 'FT8', 'C2', 'C4', 'C6', 'T8', 'CP2', 'CP4', 'CP6', 'TP8',
        #     'P2', 'P4', 'P6', 'P8', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz'
        #     ]
        
        nombre_clases = [
            'Click izq.', 'Click der.', 'Izquierda', 'Derecha', 'Arriba',
            'Abajo', 'Reposo'
        ]
        
        # caracteristicas['EG'] = [
        #     'potencia de banda', 'cruce por cero', 'desviacion estandar',
        #     'varianza', 'entropia', 'media', 'rms', 'energia', 
        #     'longitud de onda', 'integrada', 'ssc'
        #     ]
        
        caracteristicas = dict()
        
        caracteristicas['EMG'] =  [
            'potencia de banda', 'desviacion estandar', 'media', 'rms']
        caracteristicas['EEG'] =  [
            'potencia de banda', 'desviacion estandar', 'media', 'rms']
        
        # caracteristicas['EMG'] =  ['potencia de banda']
        # caracteristicas['EEG'] =  ['potencia de banda']
        
        
        self.Parametros(
            directorio, sujeto, nombres, nombre_clases, caracteristicas,
            f_tipo='butter', b_tipo='bandpass', frec_corte={
                'EMG': np.array([8, 520]), 'EEG': np.array([4, 30])},
            f_orden=5, m={'EMG': 2, 'EEG': 2}, tam_ventana_ms=1000, paso_ms=300,
            descarte_ms = {
                'EMG': {'Activo': 300, 'Reposo': 3000},
                'EEG': {'Activo': 300, 'Reposo': 3000}}, reclamador_ms={
                'EMG': {'Activo': 3100, 'Reposo': 860},
                'EEG': {'Activo': 3100, 'Reposo': 860}},
            porcen_prueba=0.2, porcen_validacion=0.1,
            calcular_csp={'EMG': True, 'EEG': True},
            calcular_ica={'EMG': False, 'EEG': False},
            num_ci={'EMG': 4, 'EEG': 7}, determinar_ci=False, epocas=128,
            lotes=128)

    def Parametros(
            self, directorio, sujeto, nombres, nombre_clases, caracteristicas, 
            f_tipo='butter', b_tipo='bandpass', frec_corte=None, f_orden=5, 
            m=None, tam_ventana_ms=1000, paso_ms=500, descarte_ms=None, 
            reclamador_ms=None, porcen_prueba=0.2, porcen_validacion=0.1, 
            calcular_csp=None, calcular_ica=None, num_ci=None, 
            determinar_ci=False, epocas=128, lotes=128):
        """Metodo Parametros:

        Se definen los parámetros predeterminados de la
        interfaz.

        Parameters
        ----------
        directorio: STR, dirección del directorio donde se encuentra la
            base de datos.
        sujeto: INT, Número del sujeto, los disponibles son 2, 7, 11, 13, 21, 25
        nombres: DICT, Nombres de los canales:
                'EMG': LIST, nombres de acuerdo al dataset.
                'EEG': LIST, nombres de acuerdo al estándar 10-10.
        nombre_clases: LIST, contiene el nombre de las clases
        caracteristicas: LIST, contiene una lista con las caracteristicas
        f_tipo: STR, tipo de filtro, predeterminado 'butter'.
        b_tipo: STR, tipo de banda de paso, predeterminado 'bandpass'.
        frec_corte: DICT, valores de las bandas de paso en Hz:
                'EMG': np.ARRAY, frecuencias, pred: np.array([8, 520]).
                'EEG': np.ARRAY, frecuencias, pred: np.array([6, 24]).
        f_orden: INT, orden del filtro a calcular, pred: 5.
        m: DICT, factor de sub muestreo:
                'EMG': INT, factor de sub muestreo para emg, pred: 2.
                'EEG': INT. factor de sub muestreo para eeg, pred: 25.
        tam_ventana_ms: INT, tamaño de ventana en ms, pred: 260.
        paso_ms: INT, paso entre ventanas en ms, pred: 60.
        descarte_ms: DICT,  indica el tiempo a saltar para luego
            empezar a tomar los datos después de una bandera, en ms:
                'EMG': DICT, el tiempo de salto para emg:
                    'Activo': INT, salto para actividad de movimiento,
                        pred: 300.
                    'Reposo': INT, salto para estado de reposo,
                        pred: 3000.
                'EEG': DICT, el tiempo de salto para eeg:
                    'Activo': INT, salto para actividad de movimiento,
                        pred: 3500.
                    'Reposo': INT, salto para estado de reposo,
                        pred: 1000.
        reclamador_ms: DICT, indica la duración de la franja de tiempo
            en la cual tomar ventanas despues de una bandera, en ms:
                'EMG': DICT, duración de franja de tiempo para emg:
                    'Activo': INT, duración para movimiento,
                        pred: 3500.
                    'Reposo': INT, duración para estado de reposo,
                        prec: 3500.
                'EMG': DICT, duración de franja de tiempo para emg:
                    'Activo': INT, duración para movimiento,
                        pred: 1000.
                    'Reposo': INT, duración para estado de reposo,
                        pred: 1000.
        porcen_prueba: FLOAT, porcentaje de datos de prueba, pred: 0.2.
        porcen_validacion: FLOAT, porcentaje de datos de validación,
            pred: 0.2.
        calcular_ica:  DICT, índica se realiza el ICA:
                'EMG': BOOL, indica si aplicar ICA a EMG, rec: False
                'EEG': BOOL, indica si aplicar ICA a EEG, rec: False
        num_ci: DICT, indica los componentes independientes a calcular:
                'EMG': INT, componentes independientes, mínimo 4
                'EEG': INT, componentes independientes, mínimo 4
        determinar_ci: BOOL, permite el cálculo automático del numero de
            ci, siendo igual a la mitad del número de canales y mayor
            que 4.
        epocas: INT, cantidad de épocas (epoch) de entrenamiento,
            pred: 1024
        lotes: INT, cantidad de lotes (batch) de entrenamiento,
            pred: 16

        Returns
        -------
        """
        # ajuste de valores predeterminados
        if frec_corte is None:
            frec_corte = {'EMG': np.array([8, 520]),
                          'EEG': np.array([6, 24])}
        if m is None:
            m = {'EMG': 2, 'EEG': 10}
        if descarte_ms is None:
            descarte_ms = {
                'EMG': {'Activo': 100, 'Reposo': 3000},
                'EEG': {'Activo': 100, 'Reposo': 3000}}
        if reclamador_ms is None:
            reclamador_ms = {
                'EMG': {'Activo': 3500, 'Reposo': 1000},
                'EEG': {'Activo': 3500, 'Reposo': 1000}}
        if calcular_csp is None:
            calcular_csp = {'EMG': True, 'EEG': True}
        if calcular_ica is None:
            calcular_ica = {'EMG': False, 'EEG': False}
        if num_ci is None:
            num_ci = {'EMG': 4, 'EEG': 7}

        # Parámetros generales
        self.directorio = directorio
        self.sujeto = sujeto
        self.nombres = nombres
        self.nombre_clases = nombre_clases
        self.f_tipo = f_tipo
        self.b_tipo = b_tipo
        self.frec_corte = frec_corte
        self.f_orden = f_orden
        self.m = m
        self.tam_ventana_ms = tam_ventana_ms
        self.paso_ms = paso_ms
        self.descarte_ms = descarte_ms
        self.reclamador_ms = reclamador_ms
        self.porcen_prueba = porcen_prueba
        self.porcen_validacion = porcen_validacion
        self.calcular_csp = calcular_csp
        self.calcular_ica = calcular_ica
        self.num_ci = num_ci
        self.caracteristicas = caracteristicas
        self.epocas = epocas
        self.lotes = lotes

        # se calculan de acuerdo a los parámetros dados
        # calcular el número de clases
        self.num_clases = len(nombre_clases)
        # traduce los nombres de canales del estándar 10-10 a los del dataset
        self.canales['EMG'] = f.TraducirNombresCanales(nombres['EMG'])
        self.canales['EEG'] = f.TraducirNombresCanales(nombres['EEG'])
        self.num_canales['EMG'] = len(self.canales['EMG'])
        self.num_canales['EEG'] = len(self.canales['EEG'])
        # para los componentes independientes
        if determinar_ci:
            # Cantidad de componentes independientes a calcular
            # El número de CI corresponde a la mitad de los canales usados
            self.num_ci['EMG'] = int(self.num_canales['EMG'] / 2)
            self.num_ci['EEG'] = int(self.num_canales['EEG'] / 2)
        # Para asegurar que haya por lo menos 4 ci, ya que de lo contrario no
        # se puede aplicar las maxpool de la CNN.
        if self.num_ci['EMG'] < 4:
            self.num_ci['EMG'] = 4
        if self.num_ci['EEG'] < 6:
            self.num_ci['EEG'] = 6

    def ParametrosTipo(
            self, tipo, directorio, sujeto, nombres, nombre_clases, 
            caracteristicas, f_tipo='butter', b_tipo='bandpass', 
            frec_corte=None, f_orden=5, m=None, tam_ventana_ms=300, paso_ms=60, 
            descarte_ms=None, reclamador_ms=None, porcen_prueba=0.2, 
            porcen_validacion=0.1, calcular_csp=None, calcular_ica=None, 
            num_ci=None, determinar_ci=False, epocas=1024, lotes=16):
        """Método ParametrosTipo:

        Se definen los parámetros predeterminados de la
        interfaz, para un tipo de señales en concreto.

        Parameters
        ----------
        tipo: STR, tipo de señales a modificar los parámetros.
        directorio: STR, dirección del directorio donde se encuentra la
            base de datos.
        sujeto: INT, Número del sujeto, los disponibles son 2, 7, 11, 13, 21, 25
        nombres: DICT, Nombres de los canales:
                'EMG': LIST, nombres de acuerdo al dataset.
                'EEG': LIST, nombres de acuerdo al estándar 10-10.
        nombre_clases: LIST, contiene el nombre de las clases
        f_tipo: STR, tipo de filtro, predeterminado 'butter'.
        b_tipo: STR, tipo de banda de paso, predeterminado 'bandpass'.
        frec_corte: DICT, valores de las bandas de paso en Hz:
                'EMG': np.ARRAY, frecuencias, rec: np.array([8, 520]).
                'EEG': np.ARRAY, frecuencias, rec: np.array([6, 24]).
        f_orden: INT, orden del filtro a calcular, pred: 5.
        m: DICT, factor de sub muestreo:
                'EMG': INT, factor de sub muestreo para emg, pred: 2.
                'EEG': INT. factor de sub muestreo para eeg, pred: 25.
        tam_ventana_ms: INT, tamaño de ventana en ms, pred: 260.
        paso_ms: INT, paso entre ventanas en ms, pred: 60.
        descarte_ms: DICT, indica el tiempo a saltar para luego
            empesar a tomar los datos después de una bandera, en ms:
                'EMG': DICT, el tiempo de salto para emg:
                    'Activo': INT, salto para actividad de movimiento,
                        rec: 300.
                    'Reposo': INT, salto para estado de reposo,
                        rec: 3000.
                'EEG': DICT, el tiempo de salto para eeg:
                    'Activo': INT, salto para actividad de movimiento,
                        rec: 3500.
                    'Reposo': INT, salto para estado de reposo,
                        rec: 1000.
        reclamador_ms: DICT, indica la duración de la franja de tiempo
            en la cual tomar ventanas después de una bandera, en ms:
                'EMG': DICT, duración de franja de tiempo para emg:
                    'Activo': INT, duración para movimiento,
                        rec: 3500.
                    'Reposo': INT, duración para estado de reposo,
                        rec: 3500.
                'EMG': DICT, duración de franja de tiempo para emg:
                    'Activo': INT, duración para movimiento,
                        rec: 1000.
                    'Reposo': INT, duración para estado de reposo,
                        rec: 1000.
        porcen_prueba: FLOAT, porcentaje de datos de prueba, pred: 0.2.
        porcen_validacion: FLOAT, porcentaje de datos de validación,
            pred: 0.2.
        calcular_ica:  DICT, índica se realiza el ICA:
                'EMG': BOOL, indica si aplicar ICA a EMG, rec: False
                'EEG': BOOL, indica si aplicar ICA a EEG, rec: False
        num_ci: DICT, indica los componentes independientes a calcular:
                'EMG': INT, componentes independientes, mínimo 4
                'EEG': INT, componentes independientes, mínimo 4
        determinar_ci: BOOL, permite el cálculo automático del número de
            ci, siendo igual a la mitad del número de canales y mayor
            que 4.
        epocas: INT, cantidad de épocas (epoch) de entrenamiento,
            pred: 1024
        lotes: INT, cantidad de lotes (batch) de entrenamiento,
            pred: 16

        Returns
        -------
        """
        # ajuste de valores predeterminados
        if tipo == 'EEG':
            if frec_corte is None:
                frec_corte['EEG'] = np.array([6, 24])
            if m is None:
                m['EEG'] = 10
            if descarte_ms is None:
                descarte_ms['EEG'] = {'Activo': 300, 'Reposo': 3000}
            if reclamador_ms is None:
                reclamador_ms['EEG'] = {'Activo': 3500, 'Reposo': 1000}
            if calcular_csp is None:
                calcular_csp['EEG'] = True
            if calcular_ica is None:
                calcular_ica['EEG'] = False
            if num_ci is None:
                num_ci['EEG'] = 6

        elif tipo == 'EMG':
            if frec_corte is None:
                frec_corte['EMG'] = np.array([8, 520])
            if m is None:
                m['EMG'] = 2
            if descarte_ms is None:
                descarte_ms['EMG'] = {'Activo': 300, 'Reposo': 3000}
            if reclamador_ms is None:
                reclamador_ms['EMG'] = {'Activo': 3500, 'Reposo': 1000}
            if calcular_csp is None:
                calcular_csp['EMG'] = True
            if calcular_ica is None:
                calcular_ica['EMG'] = False
            if num_ci is None:
                num_ci['EMG'] = 6

        # Parámetros generales
        self.directorio = directorio
        self.sujeto = sujeto
        self.nombres[tipo] = nombres[tipo]
        self.nombre_clases = nombre_clases
        self.f_tipo = f_tipo
        self.b_tipo = b_tipo
        self.frec_corte[tipo] = frec_corte[tipo]
        self.f_orden = f_orden
        self.m[tipo] = m[tipo]
        self.tam_ventana_ms = tam_ventana_ms
        self.paso_ms = paso_ms
        self.descarte_ms = descarte_ms
        self.reclamador_ms[tipo] = reclamador_ms[tipo]
        self.porcen_prueba = porcen_prueba
        self.porcen_validacion = porcen_validacion
        self.calcular_ica[tipo] = calcular_ica[tipo]
        self.num_ci[tipo] = num_ci[tipo]
        self.epocas = epocas
        self.lotes = lotes

        # se calculan de acuerdo a los párametros dados
        # calcular el número de clases
        self.num_clases = len(nombre_clases)
        # traduce los nombres de canales del estándar 10-10 a los del dataset
        self.canales[tipo] = f.TraducirNombresCanales(nombres[tipo])
        self.num_canales[tipo] = len(self.canales[tipo])
        # para los componentes independientes
        if determinar_ci:
            # Cantidad de componentes independientes a calcular
            # El número de CI corresponde a la mitad de los canales usados
            self.num_ci[tipo] = int(self.num_canales[tipo] / 2)
        # Para asegurar que haya por lo menos 4 ci, ya que de lo contrario no
        # se puede aplicar las maxpool de la CNN.
        if self.num_ci[tipo] < 6:
            self.num_ci[tipo] = 6

    def GuardarParametros(self):
        """Método GuardarParametros

        Método para guardar los parámetros con los que se diseña el
        sistema de reconocimiento de señales.

        Returns
        -------
        None.

        """
        # diccionario con la configuración del entrenamiento
        self.configuracion = {
            'directorio': self.directorio, 'sujeto': self.sujeto,
            'nombres': self.nombres, 'nombre clases': self.nombre_clases,
            'filtro': self.f_tipo, 'banda de filtro': self.b_tipo,
            'frecuencia corte': self.frec_corte,
            'orden del filtro': self.f_orden, 'm': self.m,
            'tamaño ventana': self.tam_ventana_ms, 'paso': self.paso_ms,
            'descarte': self.descarte_ms, 'reclamador': self.reclamador_ms,
            'porcentaje prueba': self.porcen_prueba,
            'porcentaje validacion': self.porcen_validacion,
            'calcular csp': self.calcular_csp,
            'calcular ica': self.calcular_ica, 'numero ci': self.num_ci,
            'epocas': self.epocas, 'lotes': self.lotes}
        # se guarda el diccionario con la información de entrenamiento
        f.GuardarPkl(self.configuracion, self.direccion + '/configuracion.pkl')

    def CargarParametros(self, tipo = None):
        self.configuracion = f.AbrirPkl(self.direccion + '/configuracion.pkl')
        # Cargar los parámetros
        if tipo is None:
            self.Parametros(
                self.configuracion['directorio'], self.configuracion['sujeto'],
                self.configuracion['nombres'], self.configuracion['nombre clases'],
                self.configuracion['filtro'], self.configuracion['banda de filtro'],
                self.configuracion['frecuencia corte'],
                self.configuracion['orden del filtro'], self.configuracion['m'],
                self.configuracion['tamaño ventana'], self.configuracion['paso'],
                self.configuracion['descarte'], self.configuracion['reclamador'],
                self.configuracion['porcentaje prueba'],
                self.configuracion['porcentaje validacion'], self.configuracion['calcular csp'],
                self.configuracion['calcular ica'], self.configuracion['numero ci'],
                False, self.configuracion['epocas'], self.configuracion['lotes'])
        else:
            self.ParametrosTipo(
                tipo, self.configuracion['directorio'], self.configuracion['sujeto'],
                self.configuracion['nombres'], self.configuracion['nombre clases'],
                self.configuracion['filtro'], self.configuracion['banda de filtro'],
                self.configuracion['frecuencia corte'],
                self.configuracion['orden del filtro'], self.configuracion['m'],
                self.configuracion['tamaño ventana'], self.configuracion['paso'],
                self.configuracion['descarte'], self.configuracion['reclamador'],
                self.configuracion['porcentaje prueba'],
                self.configuracion['porcentaje validacion'], self.configuracion['calcular csp'],
                self.configuracion['calcular ica'], self.configuracion['numero ci'],
                False, self.configuracion['epocas'], self.configuracion['lotes'])

    def Seleccion(self, tipo, sel_canales=True, sel_cara=True):
        """
        """
        epocas = 32
        # Determinar si existe una carpeta donde se evalue el rendimiento
        directo = 'Parametros/Sujeto_' + str(self.sujeto) + '/Canales/'
        # revisar si existe la carpeta
        if not exists(directo):
            f.CrearDirectorio(directo)
        
        # -----------------------------------------------------------------------------
        # lista con los canales disponibles en la base de datos
        if not sel_canales:
            lista_canales = [
                f.NombreCanal(nombre, invertir=True) for nombre in self.canales[tipo]]
        elif tipo == 'EMG':
            lista_canales = [
                'EMG_1', 'EMG_2', 'EMG_3', 'EMG_4', 'EMG_5', 'EMG_6'
                ]
        elif tipo == 'EEG':
            # lista_canales = [
            #     'FP1', 'AF7', 'AF3', 'AFz', 'F7', 'F5', 'F3', 'F1', 'Fz', 'FT7', 
            #     'FC5', 'FC3', 'FC1', 'T7', 'C5', 'C3', 'C1', 'Cz', 'TP7', 'CP5',
            #     'CP3', 'CP1', 'CPz', 'P7', 'P5', 'P3', 'P1', 'Pz', 'PO7', 'PO3',
            #     'POz', 'FP2', 'AF4', 'AF8', 'F2', 'F4', 'F6', 'F8', 'FC2', 'FC4',
            #     'FC6', 'FT8', 'C2', 'C4', 'C6', 'T8', 'CP2', 'CP4', 'CP6', 'TP8',
            #     'P2', 'P4', 'P6', 'P8', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz'
            #     ]
            # sobre corteza motora -  20 canales elegidos de bibliografia.
            lista_canales = [
                'FC1', 'FC3', 'FC5', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 
                'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'
                ]
        
        # lista con las caracteristicas temporales a extraer
        # lista_caracteristicas = [
        #     'potencia de banda', 'cruce por cero', 'desviacion estandar',
        #     'varianza', 'media', 'rms', 'energia', 
        #     'longitud de onda', 'integrada', 'ssc'
        #     ]
        
        # lista_caracteristicas = [
        #     'potencia de banda', 'desviacion estandar',
        #     'media', 'rms']
        
        # lista_caracteristicas = [
        #     'potencia de banda']
        
        # if sel_cara:
        #     lista_caracteristicas = ['potencia de banda']
        
        # if not sel_cara:
        lista_caracteristicas = self.caracteristicas[tipo]
        
        # por cada canar hacer el entrenamiento mediante kfolds
        # es necesario entonce sacar la información de los registros
        # por lo cual el procesamiento de las señales se realiza casi que igual a lo de 
        # cargar datos o el de entrenamiento
        
        # Traducir el nombre de los canales a los utlizados en la base de datos
        canales = f.TraducirNombresCanales(lista_canales)
        num_canales = len(canales)
        
        # variable donde guardar la información de los rendimientos obtenidos
        rendimiento =  dict.fromkeys(canales)
        
        for canal in canales:
            # procesamiento de señales
            rendimiento[canal] = []
            
        # -----------------------------------------------------------------------------
        # Extraer datos
        print('Extrayendo la información de la base de datos para ' + tipo)
        datos = f.ExtraerDatos(self.directorio, self.sujeto, tipo)

        # Actualiza la variable para hacer seguimiento al progreso
        print('Información extraida')
        self.ActualizarProgreso(tipo, 0.15)
        # -----------------------------------------------------------------------------
        # Filtro
        print('Diseñando el filtro para ' + tipo)
        self.filtro[tipo] = f.DisenarFiltro(
            self.f_tipo, self.b_tipo, self.f_orden, self.frec_corte[tipo],
            datos['Frecuencia muestreo'])

        # Actualiza la variable para hacer seguimiento al progreso
        print('Diseñado')
        self.ActualizarProgreso(tipo, 0.21)
        # -----------------------------------------------------------------------------
        # Función para sub muestreo
        print('Apicando filtro y submuestreo para ' + tipo)
        
        # donde se guardan los datos
        senales = dict.fromkeys(canales)
        for canal in canales:
            senales[canal] = []
        clases_OH = []
        for sesion in range(1,4):
            senales_subm, clases = f.Submuestreo(
                self.directorio, tipo, datos, self.sujeto, sesion,
                canales, self.nombre_clases, self.filtro[tipo],
                self.m[tipo])
            clases_OH.append(clases)
            del clases
            for canal in canales:
                senales[canal].append(senales_subm[canal])
            del senales_subm
            
        # Calcular a partir de frecuencias de sub muestreo
        self.frec_submuestreo[tipo] = int(
            datos['Frecuencia muestreo'] / self.m[tipo])
        self.tam_ventana[tipo] = int(
            self.tam_ventana_ms * 0.001 * self.frec_submuestreo[tipo])
        self.paso_ventana[tipo] = int(
            self.paso_ms * 0.001 * self.frec_submuestreo[tipo])

        # Actualiza la variable para hacer seguimiento al progreso
        print('Aplicados')
        self.ActualizarProgreso(tipo, 0.44)
        # -----------------------------------------------------------------------------
        # Registros
        print('Dividiendo registros para ' + tipo)
        # Cada registro es de 13 segundos, de la siguiente manera: 
        # 4 segundos para reposo, 
        # 3 segundo donde se presenta una pista visual
        # 4 segundo para ejecutar el movimiento
        tam_registro = self.tam_registro_s*self.frec_submuestreo[tipo]
        
        # donde se guardan los datos
        registros_train = dict.fromkeys(canales)
        for canal in canales:
            registros_train[canal] = []
        del canal
        # las clases de los registros
        clases_regis_train =[]

        for sesion in range(3):
            # Traducir las banderas a valores en submuestreo
            # Revisar que esta traducción sea correcta
            banderas = (datos['Banderas'][sesion][1::2]
                        - datos['Inicio grabacion'][sesion])/self.m[tipo]
            banderas = banderas.astype(int)
            clases = datos['One Hot'][sesion][:,::2]
            num_registros = len(datos['Banderas'][sesion][::2])
            regis = dict.fromkeys(canales)
            for canal in canales:
                regis[canal] = np.empty([num_registros, tam_registro])
            del canal
            
            # para iteractuar entre los registros
            i = 0
            for bandera in banderas:
                for canal in canales:
                    regis[canal][i,:] = senales[canal][sesion][bandera-tam_registro:bandera]
                # regis[i,:,:] = senales[sesion][:,bandera-tam_registro:bandera]
                i += 1
            del canal, i
            
            # Concatenar los registros
            # se juntan los de entrenamiento y de validación
            for canal in canales:
                registros_train[canal].append(regis[canal][np.concatenate((
                    self.registros_id['train'][sesion], 
                    self.registros_id['val'][sesion]))])
            del regis, canal
            # para las clases
            clases_regis_train.append(
                clases[:,np.concatenate((
                    self.registros_id['train'][sesion], 
                    self.registros_id['val'][sesion]))])
        del clases, bandera, banderas, num_registros, senales
        
        # Actualiza la variable para hacer seguimiento al progreso
        print('Divididos')
        self.ActualizarProgreso(tipo, 0.50)
        # -----------------------------------------------------------------------------
        # Descarte de datos ambiguos
        print('Diseñando ventanas para ' + tipo)
        # Valores para descarte:
        # traducción de tiempos de descarte y reclamador a número de muestras
        descarte = dict.fromkeys(['Activo', 'Reposo'])
        descarte['Activo'] = int(
            self.descarte_ms[tipo]['Activo'] * self.frec_submuestreo[tipo] / 1000)
        descarte['Reposo'] = int(
            self.descarte_ms[tipo]['Reposo'] * self.frec_submuestreo[tipo] / 1000)
        reclamador = dict.fromkeys(['Activo', 'Reposo'])
        reclamador['Activo'] = int(
            self.reclamador_ms[tipo]['Activo'] * self.frec_submuestreo[tipo] / 1000)
        reclamador['Reposo'] = int(
            self.reclamador_ms[tipo]['Reposo'] * self.frec_submuestreo[tipo] / 1000)

        # calculo de las ventanas
        # Se realiza un salto de 7 segundos para las ventanas de actividad
        x, y = f.Ventanas(
            registros_train, clases_regis_train, num_canales,
            self.num_clases, reclamador, descarte,
            self.tam_ventana[tipo], self.paso_ventana[tipo],
            7*self.frec_submuestreo[tipo])
        del registros_train, clases_regis_train
        

        # Actualiza la variable para hacer seguimiento al progreso
        print('Diseñadas')
        self.ActualizarProgreso(tipo, 0.55)
        # -----------------------------------------------------------------------------
        # Balancear ventanas
        self.balancear = False
        if self.balancear:
            print('Balanceando ventanas de ' + tipo)
            # La inicialización del balance se hace para conservar las 
            # variables anteriores y poder compararlas
            clases = np.identity(self.num_clases, dtype='int8')
            # # inicialización
            # x, y = f.Balanceo(
            #     x, y, clases[-1])
            # En el caso de que se requiera realizar en todas las clases
            for i in range(self.num_clases):
                x, y = f.Balanceo(
                    x, y, clases[i])
                
            print('Se balancean los datos para ' + tipo)
        
        # -----------------------------------------------------------------------------
        # Aquí inicia la seleción de canales
        # 
        if sel_canales:
            print('Iniciando selección de canales')
            
            # X_train, X_test, y_train, y_test = train_test_split(
            #     x, y, test_size=0.1, stratify=y)
            
            # print('Ejecutando PSO')
            # # problem = f.SVMFeatureSelection(X_train, y_train)
            # problem = f.CSPMLPChannelSelection(X_train, y_train)
            # task = Task(problem, max_iters=16) #16
            # algorithm = ParticleSwarmOptimization(population_size=32) #32
            # best_features, best_fitness = algorithm.run(task)
    
            # selected_features = best_features > 0.5
            # print('Number of selected features:', selected_features.sum())
            # print(
            #     'Selected features:', 
            #     ', '.join(np.array(lista_canales)[selected_features].tolist()))
            
            # # transformación CSP para la prueba final
            # csp_all = CSP(
            #     n_components=num_canales, reg=None, log=None, 
            #     norm_trace=False, transform_into='average_power')
            # # para calcular el csp la clases deben ser categoricas
            # x_train = csp_all.fit_transform(
            #     X_train, np.argmax(y_train, axis=1))
            # x_test = csp_all.transform(X_test)
            
            # # modelo con todas las caracteristicas
            # model_all = f.ClasificadorUnico(
            #     num_canales, 0, self.num_clases)
            
            # model_all.fit(
            #     x_train, y_train, shuffle=True, epochs=64, batch_size=32, 
            #     verbose=1) # epocas 128
            # ren_todas = model_all.evaluate(
            #     x_test, y_test, verbose=1, return_dict=False)[1]
            # print('All Features Accuracy:', ren_todas)
            
            
            # # model_selected = SVC()
            # # model_all = SVC()
            # # transformación CSP para la prueba final
            # csp = CSP(
            #     n_components=int(sum(selected_features)), reg=None, log=None, 
            #     norm_trace=False, transform_into='average_power')
            # # para calcular el csp la clases deben ser categoricas
            # x_train = csp.fit_transform(
            #     X_train[:, selected_features], np.argmax(y_train, axis=1))
            # x_test = csp.transform(X_test[:, selected_features])
            # del X_train, X_test
            
            # model_selected = f.ClasificadorUnico(
            #     int(sum(selected_features)), 0, self.num_clases)
            
            # model_selected.fit(
            #     x_train, y_train, shuffle=True, epochs=64, batch_size=32, 
            #     verbose=1) # epocas 128
            # ren_sel =  model_selected.evaluate(
            #     x_test, y_test, verbose=1, return_dict=False)[1]
            
            # print('Subset accuracy:', ren_sel)
            
            # rendimiento = pd.DataFrame(
            #     np.array([lista_canales, best_features]).T, 
            #     columns=['Canal', 'Rendimiento'])
            
            # # LA información de los canales tomados
            # f.GuardarPkl(rendimiento, directo + 'rendimiento_' + tipo)
            
            # # de esta forma se puede usar una mascara con numpy, y se
            # # se volveria a lo una lista normal
            # self.canales[tipo] = f.TraducirNombresCanales(
            #     np.array(lista_canales)[selected_features].tolist())
            # self.num_canales[tipo] = len(self.canales[tipo])
            
            
            # Selección recomendada por el profe
            for n_canal, canal in enumerate(canales):
                # división k folds
                print('Se inica evaluación iterativa mediante K-folds')
                # kfolds = KFold(n_splits=10)
                # usar shuffle split ya que con el otro no se puede hacer 
                # menos entrenamientos sin dividir más el dataset
                kfolds = ShuffleSplit(
                    n_splits=self.divisiones, test_size=self.porcen_prueba) # 5 diviciones
                  
                modelo = f.ClasificadorUnico(
                    len(lista_caracteristicas), self.tam_ventana[tipo], 
                    self.num_clases)
                # ciclo de entrenamiento:
                for i, (train_index, test_index) in enumerate(kfolds.split(x)):
                    print(str(i+1) + 'º iteración para el canal ' + canal)
                    # Diviciòn de los k folds
                    
                    # aquí son tomadas las señales de cada canal de forma
                    # que tienen la siguiente forma matricial [n_ventanas, 1, n_muestras]
                    x_train = x[train_index, n_canal].reshape(
                        (len(train_index), 1, x.shape[-1]))
                    x_test = x[test_index, n_canal].reshape(
                        (len(test_index), 1, x.shape[-1]))
                    y_train, y_test = y[train_index], y[test_index]
                    
                    # Calcular CSP
                    csp = CSP(
                        n_components=1, reg=None, log=None, 
                        norm_trace=False, transform_into='csp_space')
                       
                    # para calcular el csp la clases deben ser categoricas
                    x_train = csp.fit_transform(
                        x_train, np.argmax(y_train, axis=1))
                    x_train = f.Caracteristicas(
                        x_train, lista_caracteristicas, csp=csp)
                    
                    x_test = csp.transform(x_test)
                    x_test = f.Caracteristicas(
                        x_test, lista_caracteristicas, csp=csp)
                    
                    # clasificador a utilizar
                    modelo.fit(
                        x_train, y_train, shuffle=True, epochs=epocas, 
                        batch_size=self.lotes) # 32 epocas
                    eva = modelo.evaluate(
                        x_test, y_test, verbose=1, return_dict=True)
                           
                    rendimiento[canal].append(eva)
                    # entrenar y evaluar la clasificaciòn
                    # guardar el rendimiento obtenido
            
            # Evaluaciòn del rendimiento usando pandas
            # Seleccion de canal
            print(rendimiento)
            f.GuardarPkl(rendimiento, directo + 'rendimiento_' + tipo)
            
            # Seleccion de canal
            self.canales[tipo] = f.ElegirCanales(
                rendimiento, directo, tipo, determinar=True, num_canales=self.num_ci[tipo])
            self.num_canales[tipo] = len(self.canales[tipo])
            
            # ajustar el nombre de los canales:
            self.nombres[tipo] = []
            for canal in self.canales[tipo]:
                self.nombres[tipo].append(f.NombreCanal(canal, invertir=True))
        
        # Aquí termina la seleción de canales
        # -----------------------------------------------------------------------------
        # Aquí inicia la selección de caracteristicas mediante PSO
        
        print('Iniciando selección de caracteristicas')
        if sel_cara: 
            X_train, X_test, y_train, y_test = train_test_split(
                x, y, test_size=self.porcen_prueba, stratify=y)
            print('Iniciando selección de caracteristicas')
            # Calculo de CSP
            # Revisar si ya se hizo un entrenamiento para el tipo actual
            if self.csp[tipo] is None:
                csp = CSP(
                    n_components=num_canales, reg=None, log=None, 
                    norm_trace=False, transform_into='csp_space')
                    # norm_trace=False, transform_into='power_average')
                # para calcular el csp la clases deben ser categoricas
                X_train = csp.fit_transform(
                    X_train, np.argmax(y_train, axis=1))
            else:
                csp = self.csp[tipo]
                X_train = csp.transform(X_train)
                
            X_test = csp.transform(X_test)
            
            print('Evaluando: ', lista_caracteristicas)
            # Calculo de caracteristicas
            X_train = f.Caracteristicas(
                X_train, lista_caracteristicas, csp=csp)
            X_test, feature_names = f.Caracteristicas(
                X_test, lista_caracteristicas, generar_lista=True, 
                canales=canales, csp=csp)
            feature_names = np.array(feature_names, dtype='str')
            
            
            print('Ejecutando PSO')
            # problem = f.SVMFeatureSelection(X_train, y_train)
            problem = f.MLPFeatureSelection(X_train, y_train)
            task = Task(problem, max_iters=16) #16
            algorithm = ParticleSwarmOptimization(population_size=16) #32
            best_features, best_fitness = algorithm.run(task)
    
            selected_features = best_features > 0.5
            print('Number of selected features:', selected_features.sum())
            print(
                'Selected features:', 
                ', '.join(feature_names[selected_features].tolist()))
            
            # model_selected = SVC()
            # model_all = SVC()
            model_selected = f.ClasificadorUnico(
                int(sum(selected_features)), 0, self.num_clases)
            model_all = f.ClasificadorUnico(
                len(selected_features), 0, self.num_clases)
            
            model_selected.fit(
                X_train[:, selected_features], y_train, shuffle=True, 
                epochs=int(self.epocas/2), batch_size=self.lotes, verbose=1) 
            # epocas 128
            ren_sel =  model_selected.evaluate(
                X_test[:, selected_features], y_test, verbose=1, 
                return_dict=False)[1]
            
            print('Subset accuracy:', ren_sel)
            
            model_all.fit(
                X_train, y_train, shuffle=True, epochs=int(self.epocas/2), 
                batch_size=self.lotes, verbose=1) # epocas 128
            ren_todas = model_all.evaluate(
                X_test, y_test, verbose=1, return_dict=False)[1]
            print('All Features Accuracy:', ren_todas)
            
            exactitud_carac = {'todas': ren_todas, 'Seleccion': ren_sel}
            f.GuardarPkl(exactitud_carac, directo + 'evaluacion_carac_'+tipo)
            
            self.parcial[tipo] = f.CrearRevision(feature_names.tolist(), best_features)
            # resultados = pd.concat([resultados, parcial])
            f.GuardarPkl(self.parcial[tipo], directo + 'resultados_' + tipo)
            
            self.caracteristicascanal[tipo] = f.SeleccionarCaracteristicas(self.parcial[tipo])
            # self.caracteristicas[tipo] = f.SeleccionarCaracteristicas(parcial)

        """ Aquí termina la selección de caracteristicas.
        """
                
        print('Seleccion completada')
        
        

    def Entrenamiento(self, tipo):
        """Método Entrenamiento:

        Realiza el entrenamiento del sistema para un tipo de señales,
        es decir no realiza la combinación de estas, hecho de esta
        manera para permitir la ejecución en hilo.

        Parameters
        ----------
        tipo: STR, el tipo de señales para entrenamiento, puede ser
            'EMG' o 'EEG'

        Returns
        -------

        """
        # -----------------------------------------------------------------------------
        # Extraer datos
        print('Extrayendo la información de la base de datos para ' + tipo)
        datos = f.ExtraerDatos(self.directorio, self.sujeto, tipo)

        # Actualiza la variable para hacer seguimiento al progreso
        print('Información extraida')
        self.ActualizarProgreso(tipo, 0.15)
        # -----------------------------------------------------------------------------
        # Filtro
        print('Diseñando el filtro para ' + tipo)
        self.filtro[tipo] = f.DisenarFiltro(
            self.f_tipo, self.b_tipo, self.f_orden, self.frec_corte[tipo],
            datos['Frecuencia muestreo'])

        # Actualiza la variable para hacer seguimiento al progreso
        print('Diseñado')
        self.ActualizarProgreso(tipo, 0.21)
        # -----------------------------------------------------------------------------
        # Función para sub muestreo
        print('Apicando filtro y submuestreo para ' + tipo)
        
        # donde se guardan los datos
        senales = dict.fromkeys(self.canales[tipo])
        for canal in self.canales[tipo]:
            senales[canal] = []
        clases_OH = []
        for sesion in range(1,4):
            senales_subm, clases = f.Submuestreo(
                self.directorio, tipo, datos, self.sujeto, sesion,
                self.canales[tipo], self.nombre_clases, self.filtro[tipo],
                self.m[tipo])
            clases_OH.append(clases)
            del clases
            for canal in self.canales[tipo]:
                senales[canal].append(senales_subm[canal])
            del senales_subm
            
        # Calcular a partir de frecuencias de sub muestreo
        self.frec_submuestreo[tipo] = int(
            datos['Frecuencia muestreo'] / self.m[tipo])
        self.tam_ventana[tipo] = int(
            self.tam_ventana_ms * 0.001 * self.frec_submuestreo[tipo])
        self.paso_ventana[tipo] = int(
            self.paso_ms * 0.001 * self.frec_submuestreo[tipo])

        # Actualiza la variable para hacer seguimiento al progreso
        print('Aplicados')
        self.ActualizarProgreso(tipo, 0.44)
        # -----------------------------------------------------------------------------
        # Registros
        print('Dividiendo registros para ' + tipo)
        # Cada registro es de 13 segundos, de la siguiente manera: 
        # 4 segundos para reposo, 
        # 3 segundo donde se presenta una pista visual
        # 4 segundo para ejecutar el movimiento
        tam_registro = self.tam_registro_s*self.frec_submuestreo[tipo]
        
        # donde se guardan los datos
        registros_train = dict.fromkeys(self.canales[tipo])
        registros_val = dict.fromkeys(self.canales[tipo])
        registros_test = dict.fromkeys(self.canales[tipo])
        for canal in self.canales[tipo]:
            registros_train[canal] = []
            registros_val[canal] = []
            registros_test[canal] = []
        del canal
        # las clases de los registros
        clases_regis_train =[]
        clases_regis_val = []
        clases_regis_test = []

        for sesion in range(3):
            # Traducir las banderas a valores en submuestreo
            # Revisar que esta traducción sea correcta
            banderas = (datos['Banderas'][sesion][1::2]
                        - datos['Inicio grabacion'][sesion])/self.m[tipo]
            banderas = banderas.astype(int)
            clases = datos['One Hot'][sesion][:,::2]
            num_registros = len(datos['Banderas'][sesion][::2])
            regis = dict.fromkeys(self.canales[tipo])
            for canal in self.canales[tipo]:
                regis[canal] = np.empty([num_registros, tam_registro])
            del canal
            
            # para iteractuar entre los registros
            i = 0
            for bandera in banderas:
                for canal in self.canales[tipo]:
                    regis[canal][i,:] = senales[canal][sesion][bandera-tam_registro:bandera]
                # regis[i,:,:] = senales[sesion][:,bandera-tam_registro:bandera]
                i += 1
            del canal, i
            
            # Concatenar los registros
            for canal in self.canales[tipo]:
                registros_train[canal].append(regis[canal][self.registros_id['train'][sesion]])
                registros_val[canal].append(regis[canal][self.registros_id['val'][sesion]])
                registros_test[canal].append(regis[canal][self.registros_id['test'][sesion]])
            del regis, canal
            # para las clases
            clases_regis_train.append(
                clases[:,self.registros_id['train'][sesion]])
            clases_regis_val.append(
                clases[:,self.registros_id['val'][sesion]])
            clases_regis_test.append(
                clases[:,self.registros_id['test'][sesion]])
        del clases, bandera, banderas, num_registros, senales
        
        # Actualiza la variable para hacer seguimiento al progreso
        print('Divididos')
        self.ActualizarProgreso(tipo, 0.50)
        # -----------------------------------------------------------------------------
        # Calcular ICA
        # se clacula la matriz de transformación aquí para ahorrar memoria
        if self.calcular_ica[tipo]:
            print ('Calculando la transformada ICA para ' + tipo)
            # if tipo == 'Isa':
            #senales = registros_train
            senales = np.concatenate(
                [registros_train[0][:],registros_train[1][:],registros_train[2][:]],
                axis=0)
            senales = np.concatenate(senales[:], axis=1)
            # Calcular transformación ICA y matriz de blanqueo
            self.ica_total[tipo], self.whiten[tipo] = f.CICA(
                senales, self.num_ci[tipo])
            del senales

            print ('Calculada')
            self.ActualizarProgreso(tipo, 0.53)
        
        # -----------------------------------------------------------------------------
        # Descarte de datos ambiguos
        print('Diseñando ventanas para ' + tipo)
        # Valores para descarte:
        # traducción de tiempos de descarte y reclamador a número de muestras
        descarte = dict.fromkeys(['Activo', 'Reposo'])
        descarte['Activo'] = int(
            self.descarte_ms[tipo]['Activo'] * self.frec_submuestreo[tipo] / 1000)
        descarte['Reposo'] = int(
            self.descarte_ms[tipo]['Reposo'] * self.frec_submuestreo[tipo] / 1000)
        reclamador = dict.fromkeys(['Activo', 'Reposo'])
        reclamador['Activo'] = int(
            self.reclamador_ms[tipo]['Activo'] * self.frec_submuestreo[tipo] / 1000)
        reclamador['Reposo'] = int(
            self.reclamador_ms[tipo]['Reposo'] * self.frec_submuestreo[tipo] / 1000)

        # calculo de las ventanas
        train, class_train = f.Ventanas(
            registros_train, clases_regis_train, self.num_canales[tipo],
            self.num_clases, reclamador, descarte,
            self.tam_ventana[tipo], self.paso_ventana[tipo],
            7*self.frec_submuestreo[tipo])
        del registros_train, clases_regis_train
        validation, class_validation = f.Ventanas(
            registros_val, clases_regis_val, self.num_canales[tipo],
            self.num_clases, reclamador, descarte,
            self.tam_ventana[tipo], self.paso_ventana[tipo],
            7*self.frec_submuestreo[tipo])
        del registros_val, clases_regis_val
        test, class_test = f.Ventanas(
            registros_test, clases_regis_test, self.num_canales[tipo],
            self.num_clases, reclamador, descarte,
            self.tam_ventana[tipo], self.paso_ventana[tipo],
            7*self.frec_submuestreo[tipo])
        del registros_test, clases_regis_test

        # Actualiza la variable para hacer seguimiento al progreso
        print('Diseñadas')
        self.ActualizarProgreso(tipo, 0.55)
        # -----------------------------------------------------------------------------
        # Balancear ventanas
        if self.balancear:
            print('Balanceando ventanas de ' + tipo)
            
            # solo se balancea reposo
            # clase_reposo = np.eye(self.num_clases, dtype='int8')[:,-1]
            # train, class_train = f.Balanceo(train, class_train, clase_reposo)
            # validation, class_validation = f.Balanceo(
            #     validation, class_validation, clase_reposo)
            # test, class_test = f.Balanceo(test, class_test, clase_reposo)
            
            # La inicialización del balance se hace para conservar las 
            # variables anteriores y poder compararlas
            clases = np.identity(self.num_clases, dtype='int8')
            # inicialización
            train, class_train = f.Balanceo(
                train, class_train, clases[-1])
            validation, class_validation = f.Balanceo(
                validation, class_validation, clases[-1])
            test, class_test = f.Balanceo(
                test, class_test, clases[-1])

            # En el caso de que se requiera realizar en todas las clases
            for i in range(self.num_clases - 1):
                train, class_train = f.Balanceo(
                    train, class_train, clases[i])
                validation, class_validation = f.Balanceo(
                    validation, class_validation, clases[i])
                test, class_test = f.Balanceo(
                    test, class_test, clases[i])
            
            print('Se balancean los datos para ' + tipo)

        # para revisar la cantidad de ventanas disponibles
        # self.num_ventanas[tipo] = dict.fromkeys(['Entrenamiento', 'Validacion', 'Prueba'])
        self.num_ventanas[tipo]['Entrenamiento'] = len(class_train)
        self.num_ventanas[tipo]['Validacion'] = len(class_validation)
        self.num_ventanas[tipo]['Prueba'] = len(class_test)

        # Actualiza la variable para hacer seguimiento al progreso
        self.ActualizarProgreso(tipo, 0.69)
        # -----------------------------------------------------------------------------
        # Extracción de características
        # Cálculo de FastICA
        # aplicar ICA
        if self.calcular_ica[tipo]:
            print ('Aplicando transformada ICA para ' + tipo)
            # aplicar transformaciones a las ventanas
            # de acuerdo con las matrices de transformación entrenadas
            # train = f.AplicarICA(
            #     self.num_ventanas[tipo]['Entrenamiento'], self.num_ci[tipo],
            #     self.tam_ventana[tipo], self.ica_total[tipo], train)
            # validation = f.AplicarICA(
            #     self.num_ventanas[tipo]['Validacion'], self.num_ci[tipo],
            #     self.tam_ventana[tipo], self.ica_total[tipo], validation)
            # test = f.AplicarICA(
            #     self.num_ventanas[tipo]['Prueba'], self.num_ci[tipo],
            #     self.tam_ventana[tipo], self.ica_total[tipo], test)
            
            # transformar ICA para cada imagen
            # se aplica withening de acuerdo a los datos entrenados y 
            # luego se calcula los IC de las ventanas
            train = f.TransformarICA(train, self.whiten[tipo], 
                self.num_ventanas[tipo]['Entrenamiento'], self.num_ci[tipo], 
                self.tam_ventana[tipo])
            # transformar ICA para cada imagen
            validation = f.TransformarICA(validation, self.whiten[tipo], 
                self.num_ventanas[tipo]['Validacion'], self.num_ci[tipo], 
                self.tam_ventana[tipo])
            # transformar ICA para cada imagen
            test = f.TransformarICA(test, self.whiten[tipo], 
                self.num_ventanas[tipo]['Prueba'], self.num_ci[tipo], 
                self.tam_ventana[tipo])
            print ('Aplicada')
            self.ActualizarProgreso(tipo, 0.77)
        else:
            self.num_ci[tipo] = self.num_canales[tipo]
            
        # Calculo de CSP
        self.csp[tipo] = CSP(
            n_components=self.num_canales[tipo], reg=None, log=None, 
            norm_trace=False, transform_into='average_power')
            # norm_trace=False, transform_into='csp_space')
        
        # para calcular el csp la clases deven ser categoricas
        train  = self.csp[tipo].fit_transform(
            train, np.argmax(class_train, axis=1))
        validation = self.csp[tipo].transform(validation)
        test = self.csp[tipo].transform(test)
        
        # calcular caracteristicas
        # train = f.Caracteristicas(train, self.caracteristicas[tipo])
        # validation = f.Caracteristicas(validation, self.caracteristicas[tipo])
        # test = f.Caracteristicas(test, self.caracteristicas[tipo])

        # -----------------------------------------------------------------------------
        # Clasificador
        print('Entrenamiento del clasificador para ' + tipo)
        # diseñar, entrenar y revisar el rendimiento de los clasificadores
        self.modelo[tipo], cnn, self.metricas[tipo], self.confusion[tipo], self.prediccion[tipo] = f.Clasificador(
            train, class_train, validation, class_validation,
            test, class_test, self.direccion, tipo, self.num_ci[tipo],
            self.tam_ventana[tipo], self.nombre_clases, self.num_clases,
            self.epocas, self.lotes)
        del train, validation, test, class_train, class_validation
        self.class_test = class_test
        del class_test

        # valor de la precisión general del modelo entrenado
        self.exactitud[tipo] = 100 * self.metricas[tipo]['categorical_accuracy']
        print("La exactitud del modelo: {:5.2f}%".format(
            100 * self.metricas[tipo]['categorical_accuracy']))
        # Función para graficar las matrices de confusión y las gráficas de
        # entrenamiento
        f.Graficas(
            self.direccion, cnn, self.confusion[tipo], self.nombre_clases,
            tipo)

        # Actualiza la variable para hacer seguimiento al progreso
        print('Concluye el entrenamiento del clasificador para ' + tipo)
        self.ActualizarProgreso(tipo, 0.90)
        # Entrenado
        # -----------------------------------------------------------------------------
        # Guardar datos
        print('Guardando información de entrenamiento')
        # actualización de dirección
        path = self.direccion + '/Procesamiento/'
        # Guardar filtros
        f.GuardarPkl(self.filtro[tipo], path + 'filtro_' + tipo + '.pkl')
        # Guardar datos de ICA
        if self.calcular_ica[tipo]:
            f.GuardarPkl(self.whiten[tipo], path + 'whiten_' + tipo + '.pkl')
            f.GuardarPkl(self.ica_total[tipo], path + 'ica_' + tipo + '.pkl')
        # Guardar datos de matrices de confusión
        f.GuardarPkl(
            self.confusion[tipo]['Validacion'],
            path + 'CM_val_' + tipo + '.pkl')
        # Guardar datos de historial de entrenamiento
        # cnn_emg
        # Guardar métricas de entrenamiento
        f.GuardarPkl(
            self.metricas[tipo],
            self.direccion + '/General/metricas_' + tipo + '.pkl')
        # Diccionario donde se guardan las métricas de entrenamiento
        info = {
            'Sujeto': self.sujeto, 'Id': self.ubi,
            'Tipo de señales': tipo, 'Exactitud': self.exactitud[tipo]}
        # calcular precisión por clases mediante matriz de confusión
        presicion_clases, _ = f.PresicionClases(self.confusion[tipo]['Prueba'])
        presicion_clases = dict(zip(self.nombre_clases, presicion_clases))
        # concatenar en un solo diccionario
        info.update(self.metricas[tipo])
        info.update(presicion_clases)
        f.GuardarMetricas(info)
        # Diccionario donde se guarda la configuración de la interfaz !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        config = {
            'Sujeto': self.sujeto, 'Id': self.ubi,
            'Tipo de señales': tipo, 'canales': ', '.join(self.nombres[tipo]),
            'clases': ', '.join(self.nombre_clases), 'filtro': self.f_tipo,
            'banda': self.b_tipo,
            'frecuencia de corte': '-'.join(str(n) for n in self.frec_corte[tipo]),
            'orden filtro': self.f_orden, 'm': self.m[tipo],
            'tamaño ventana ms': self.tam_ventana_ms, 'paso ms': self.paso_ms,
            'porcen_prueba': self.porcen_prueba,
            'porcentaje validacion': self.porcen_validacion,
            'calcular ica': self.calcular_ica[tipo],
            'numero ci': self.num_ci[tipo], 'epocas': self.epocas,
            'lotes': self.lotes}
        f.GuardarConfiguracion(config)
        # revisar si guardar los párametros del clasificador.
        print('Se guardan los datos de entrenamiento para ' + tipo)
        self.ActualizarProgreso(tipo, 0.95)

    def Entrenar(self):
        """Método Entrenar:

        Realiza el entrenamiento del sistema combinando las señales
        antes de llegar al clasificador.

        Parameters
        ----------

        Returns
        -------

        """
        # # -----------------------------------------------------------------------------
        # # Extraer datos
        # print('Extrayendo la información de la base de datos')
        # datos = f.ExtraerDatos(self.directorio, self.sujeto, 'EMG')
        
        # # Actualiza la variable para hacer seguimiento al progreso
        # print('Información extraida')
        # self.ActualizarProgreso('General', 0.15)
        
        # Donde se guardaran las ventanas
        entrenamiento = dict.fromkeys(['EEG', 'EMG'])
        validacion = dict.fromkeys(['EEG', 'EMG'])
        prueba = dict.fromkeys(['EEG', 'EMG'])
        
        # direccion donde se guardan los parametros
        path = self.direccion + '/Procesamiento/'
        
        for tipo in ['EMG', 'EEG']:
            # -----------------------------------------------------------------------------
            # Extraer datos
            print('Extrayendo la información de la base de datos para ' + tipo)
            datos = f.ExtraerDatos(self.directorio, self.sujeto, tipo)

            # Actualiza la variable para hacer seguimiento al progreso
            print('Información extraida')
            self.ActualizarProgreso(tipo, 0.15)
            # -----------------------------------------------------------------------------
            # Filtro
            print('Diseñando el filtro para ' + tipo)
            self.filtro[tipo] = f.DisenarFiltro(
                self.f_tipo, self.b_tipo, self.f_orden, self.frec_corte[tipo],
                datos['Frecuencia muestreo'])

            # Actualiza la variable para hacer seguimiento al progreso
            print('Diseñado')
            self.ActualizarProgreso(tipo, 0.21)
            
            # -----------------------------------------------------------------------------
            # Función para sub muestreo
            print('Apicando filtro y submuestreo para ' + tipo)
            
            # donde se guardan los datos
            senales = dict.fromkeys(self.canales[tipo])
            for canal in self.canales[tipo]:
                senales[canal] = []
            clases_OH = []
            for sesion in range(1,4):
                senales_subm, clases = f.Submuestreo(
                    self.directorio, tipo, datos, self.sujeto, sesion,
                    self.canales[tipo], self.nombre_clases, self.filtro[tipo],
                    self.m[tipo])
                clases_OH.append(clases)
                del clases
                for canal in self.canales[tipo]:
                    senales[canal].append(senales_subm[canal])
                del senales_subm
            
            # Calcular a partir de frecuencias de sub muestreo
            self.frec_submuestreo[tipo] = int(
                datos['Frecuencia muestreo'] / self.m[tipo])
            self.tam_ventana[tipo] = int(
                self.tam_ventana_ms * 0.001 * self.frec_submuestreo[tipo])
            self.paso_ventana[tipo] = int(
                self.paso_ms * 0.001 * self.frec_submuestreo[tipo])

            # Actualiza la variable para hacer seguimiento al progreso
            print('Aplicados')
            self.ActualizarProgreso(tipo, 0.44)
            
            # -----------------------------------------------------------------------------
            # Registros
            print('Dividiendo registros para ' + tipo)
            # Cada registro es de 13 segundos, de la siguiente manera: 
            # 4 segundos para reposo, 
            # 3 segundo donde se presenta una pista visual
            # 4 segundo para ejecutar el movimiento
            tam_registro = self.tam_registro_s*self.frec_submuestreo[tipo]
            
            # donde se guardan los datos
            registros_train = dict.fromkeys(self.canales[tipo])
            registros_val = dict.fromkeys(self.canales[tipo])
            registros_test = dict.fromkeys(self.canales[tipo])
            for canal in self.canales[tipo]:
                registros_train[canal] = []
                registros_val[canal] = []
                registros_test[canal] = []
            del canal
            # las clases de los registros
            clases_regis_train =[]
            clases_regis_val = []
            clases_regis_test = []

            for sesion in range(3):
                # Traducir las banderas a valores en submuestreo
                # Revisar que esta traducción sea correcta
                banderas = (datos['Banderas'][sesion][1::2]
                            - datos['Inicio grabacion'][sesion])/self.m[tipo]
                banderas = banderas.astype(int)
                clases = datos['One Hot'][sesion][:,::2]
                num_registros = len(datos['Banderas'][sesion][::2])
                regis = dict.fromkeys(self.canales[tipo])
                for canal in self.canales[tipo]:
                    regis[canal] = np.empty([num_registros, tam_registro])
                del canal
                
                # para iteractuar entre los registros
                i = 0
                for bandera in banderas:
                    for canal in self.canales[tipo]:
                        regis[canal][i,:] = senales[canal][sesion][
                            bandera-tam_registro:bandera]
                    # regis[i,:,:] = senales[sesion][
                    #   :,bandera-tam_registro:bandera]
                    i += 1
                del canal, i
                
                # Concatenar los registros
                for canal in self.canales[tipo]:
                    registros_train[canal].append(regis[canal][
                        self.registros_id['train'][sesion]])
                    registros_val[canal].append(regis[canal][
                        self.registros_id['val'][sesion]])
                    registros_test[canal].append(regis[canal][
                        self.registros_id['test'][sesion]])
                del regis, canal
                # para las clases
                clases_regis_train.append(
                    clases[:,self.registros_id['train'][sesion]])
                clases_regis_val.append(
                    clases[:,self.registros_id['val'][sesion]])
                clases_regis_test.append(
                    clases[:,self.registros_id['test'][sesion]])
            del clases, bandera, banderas, num_registros, senales
            
            # Actualiza la variable para hacer seguimiento al progreso
            print('Divididos')
            self.ActualizarProgreso(tipo, 0.50)
            
            # -----------------------------------------------------------------------------
            # Calcular ICA
            # se clacula la matriz de transformación aquí para ahorrar memoria
            if self.calcular_ica[tipo]:
                print ('Calculando la transformada ICA para ' + tipo)
                # if tipo == 'Isa':
                #senales = registros_train
                senales = np.concatenate(
                    [registros_train[0][:],registros_train[1][:],registros_train[2][:]],
                    axis=0)
                senales = np.concatenate(senales[:], axis=1)
                # Calcular transformación ICA y matriz de blanqueo
                self.ica_total[tipo], self.whiten[tipo] = f.CICA(
                    senales, self.num_ci[tipo])
                del senales
                print ('Calculada')
                self.ActualizarProgreso(tipo, 0.53)
            
            # -----------------------------------------------------------------------------
            # Descarte de datos ambiguos
            print('Diseñando ventanas para ' + tipo)
            # Valores para descarte:
            # traducción de tiempos de descarte y reclamador a número de muestras
            descarte = dict.fromkeys(['Activo', 'Reposo'])
            descarte['Activo'] = int(
                self.descarte_ms[tipo]['Activo'] * self.frec_submuestreo[tipo] / 1000)
            descarte['Reposo'] = int(
                self.descarte_ms[tipo]['Reposo'] * self.frec_submuestreo[tipo] / 1000)
            reclamador = dict.fromkeys(['Activo', 'Reposo'])
            reclamador['Activo'] = int(
                self.reclamador_ms[tipo]['Activo'] * self.frec_submuestreo[tipo] / 1000)
            reclamador['Reposo'] = int(
                self.reclamador_ms[tipo]['Reposo'] * self.frec_submuestreo[tipo] / 1000)

            # calculo de las ventanas
            train, class_train = f.Ventanas(
                registros_train, clases_regis_train, self.num_canales[tipo],
                self.num_clases, reclamador, descarte,
                self.tam_ventana[tipo], self.paso_ventana[tipo],
                7*self.frec_submuestreo[tipo])
            del registros_train, clases_regis_train
            validation, class_validation = f.Ventanas(
                registros_val, clases_regis_val, self.num_canales[tipo],
                self.num_clases, reclamador, descarte,
                self.tam_ventana[tipo], self.paso_ventana[tipo],
                7*self.frec_submuestreo[tipo])
            del registros_val, clases_regis_val
            test, class_test = f.Ventanas(
                registros_test, clases_regis_test, self.num_canales[tipo],
                self.num_clases, reclamador, descarte,
                self.tam_ventana[tipo], self.paso_ventana[tipo],
                7*self.frec_submuestreo[tipo])
            del registros_test, clases_regis_test

            # Actualiza la variable para hacer seguimiento al progreso
            print('Diseñadas')
            self.ActualizarProgreso(tipo, 0.55)
            
            # -----------------------------------------------------------------------------
            # Extracción de características
            # Cálculo de FastICA
            # aplicar ICA
            if self.calcular_ica[tipo]:
                print ('Aplicando transformada ICA para ' + tipo)
                # aplicar transformaciones a las ventanas
                # de acuerdo con las matrices de transformación entrenadas
                # train = f.AplicarICA(
                #     self.num_ventanas[tipo]['Entrenamiento'], self.num_ci[tipo],
                #     self.tam_ventana[tipo], self.ica_total[tipo], train)
                # validation = f.AplicarICA(
                #     self.num_ventanas[tipo]['Validacion'], self.num_ci[tipo],
                #     self.tam_ventana[tipo], self.ica_total[tipo], validation)
                # test = f.AplicarICA(
                #     self.num_ventanas[tipo]['Prueba'], self.num_ci[tipo],
                #     self.tam_ventana[tipo], self.ica_total[tipo], test)
                
                # transformar ICA para cada imagen
                # se aplica withening de acuerdo a los datos entrenados y 
                # luego se calcula los IC de las ventanas
                train = f.TransformarICA(train, self.whiten[tipo], 
                    self.num_ventanas[tipo]['Entrenamiento'], self.num_ci[tipo], 
                    self.tam_ventana[tipo])
                # transformar ICA para cada imagen
                validation = f.TransformarICA(validation, self.whiten[tipo], 
                    self.num_ventanas[tipo]['Validacion'], self.num_ci[tipo], 
                    self.tam_ventana[tipo])
                # transformar ICA para cada imagen
                test = f.TransformarICA(test, self.whiten[tipo], 
                    self.num_ventanas[tipo]['Prueba'], self.num_ci[tipo], 
                    self.tam_ventana[tipo])
                print ('Aplicada')
                self.ActualizarProgreso(tipo, 0.69)
            else:
                self.num_ci[tipo] = self.num_canales[tipo]
            
            if self.calcular_csp[tipo]:
                print ('Aplicando transformada CSP para ' + tipo)
                # Calculo de CSP
                self.csp[tipo] = CSP(
                    n_components=self.num_canales[tipo], reg=None, log=None,
                    # norm_trace=False, transform_into='average_power') !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    norm_trace=False, transform_into='csp_space')
            
                # para calcular el csp la clases deven ser categoricas
                train = self.csp[tipo].fit_transform(
                    train, np.argmax(class_train, axis=1))
                validation = self.csp[tipo].transform(validation)
                test = self.csp[tipo].transform(test)
                
                # valores me media y std
                media = self.csp[tipo].mean_
                std = self.csp[tipo].std_
            else:
                media, std = f.MediaStd(train)
            
            # entrenamiento[tipo]  = self.csp[tipo].fit_transform(
            #     train, np.argmax(class_train, axis=1)) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # validacion[tipo] = self.csp[tipo].transform(validation)
            # prueba[tipo] = self.csp[tipo].transform(test)
            
            # seleccionando con PSO
            if self.carac_seleccionadas:
                selected_features = np.array(
                    self.parcial[tipo]['Rendimiento'], dtype='float') > 0.5
                entrenamiento[tipo] = entrenamiento[tipo][:, selected_features]
                validacion[tipo] = validacion[tipo][:, selected_features]
                prueba[tipo] = prueba[tipo][:, selected_features]
            
            # Calcular caracteristica en el tiempo
            # Calculo de caracteristicas
            entrenamiento[tipo] = f.ExtraerCaracteristicas(
                train, self.caracteristicascanal[tipo], self.canales[tipo],
                media=media, std=std)
            validacion[tipo] = f.ExtraerCaracteristicas(
                validation, self.caracteristicascanal[tipo], 
                self.canales[tipo], media=media, std=std)
            prueba[tipo] = f.ExtraerCaracteristicas(
                test, self.caracteristicascanal[tipo],  self.canales[tipo],
                media=media, std=std)
            """ 
            Supongo que en este punto hay que poner la parte de la 
            extracción de caracteristicas de acuerdo a lo que se ha 
            seleccionado, ver donde se guarda los datos de CSP, guardar la media por canal
            """
            """
            Final de la extracción de caracteristicas seleccionadas
            """
            
            # -----------------------------------------------------------------------------
            # Guardar datos
            print('Guardando información de entrenamiento')
            
            # Guardar filtros
            f.GuardarPkl(self.filtro[tipo], path + 'filtro_' + tipo + '.pkl')
            # Guardar datos de ICA
            if self.calcular_ica[tipo]:
                f.GuardarPkl(self.whiten[tipo], path + 'whiten_' + tipo + '.pkl')
                f.GuardarPkl(self.ica_total[tipo], path + 'ica_' + tipo + '.pkl')
            # Guardar datos de CSP
            if self.calcular_csp[tipo]:
                f.GuardarPkl(self.csp[tipo], path + 'csp_' + tipo + '.pkl')
            
            f.GuardarPkl(media, path + 'media_' + tipo + '.pkl')
            f.GuardarPkl(std, path + 'std_' + tipo + '.pkl')
            
            # Diccionario donde se guarda la configuración de la interfaz !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            config = {
                'Sujeto': self.sujeto, 'Id': self.ubi,
                'Tipo de señales': tipo, 'canales': ', '.join(self.nombres[tipo]),
                'clases': ', '.join(self.nombre_clases), 'filtro': self.f_tipo,
                'banda': self.b_tipo,
                'frecuencia de corte': '-'.join(str(n) for n in self.frec_corte[tipo]),
                'orden filtro': self.f_orden, 'm': self.m[tipo],
                'tamaño ventana ms': self.tam_ventana_ms, 'paso ms': self.paso_ms,
                'porcen_prueba': self.porcen_prueba,
                'porcentaje validacion': self.porcen_validacion,
                'calcular ica': self.calcular_ica[tipo],
                'numero ci': self.num_ci[tipo], 'epocas': self.epocas,
                'lotes': self.lotes}
            f.GuardarConfiguracion(config)
            # revisar si guardar los párametros del clasificador.
            print('Se guardan los datos de entrenamiento para ' + tipo)
            self.ActualizarProgreso(tipo, 0.77)
            
            # # Donde se guardaran las ventanas para la combinación
            # entrenamiento[tipo] = f.Caracteristicas(train, self.caracteristicas[tipo])
            # validacion[tipo], lista = f.Caracteristicas(
            #     validation, self.caracteristicas[tipo], generar_lista=True,
            #     canales=self.canales[tipo])
            # prueba[tipo] = f.Caracteristicas(test, self.caracteristicas[tipo])
            # clases_entrenamiento[tipo] = class_train
            # clases_validacion[tipo] = class_validation
            # clases_prueba[tipo] = class_test
            
            del train, validation, test
        # -----------------------------------------------------------------------------
        # Balancear ventanas
        self.balancear = False
        if self.balancear:
            print('Balanceando ventanas')

            tipos_clases = np.identity(self.num_clases, dtype='int8')
            for clase in tipos_clases:
                entrenamiento['EEG'], entrenamiento['EMG'], class_train = f.BalanceDoble(
                    entrenamiento['EEG'], entrenamiento['EMG'], class_train, 
                    clase)
                validacion['EEG'], validacion['EMG'], class_validation = f.BalanceDoble(
                    validacion['EEG'], validacion['EMG'], class_validation, 
                    clase)
                # prueba['EEG'], prueba['EMG'], class_test = f.BalanceDoble(
                #     prueba['EEG'], prueba['EMG'], class_test, clase)

            print('Se balancean los datos de entrenamiento y validación')

        # para revisar la cantidad de ventanas disponibles
        # self.num_ventanas[tipo] = dict.fromkeys(['Entrenamiento', 'Validacion', 'Prueba'])
        self.num_ventanas[tipo]['Entrenamiento'] = len(class_train)
        self.num_ventanas[tipo]['Validacion'] = len(class_validation)
        self.num_ventanas[tipo]['Prueba'] = len(class_test)

        # Actualiza la variable para hacer seguimiento al progreso
        self.ActualizarProgreso(tipo, 0.69)
        
        # -----------------------------------------------------------------------------
        # Combinación de las ventanas
        # la estructura de los datos es:
        # ventanas = [id_ventana, canal, muestra]
        train = np.concatenate(
            (entrenamiento['EMG'], entrenamiento['EEG']), axis=1)
        validation = np.concatenate(
            (validacion['EMG'], validacion['EEG']), axis=1)
        test = np.concatenate(
            (prueba['EMG'], prueba['EEG']), axis=1)
        
        # -----------------------------------------------------------------------------
        # Clasificador
        print('Entrenamiento del clasificador combinado')
        
        tipo = 'Combinada'
        # diseñar, entrenar y revisar el rendimiento de los clasificadores
        self.modelo, cnn, self.metricas[tipo], self.confusion[tipo], self.prediccion = f.Clasificador(
            train, class_train, validation, class_validation, test, class_test, 
            self.direccion, tipo, self.num_canales['EEG']+self.num_canales['EMG'],
            self.tam_ventana['EEG'], self.nombre_clases, self.num_clases,
            self.epocas, self.lotes)
        del train, validation, test, class_train, class_validation
        self.class_test = class_test
        del class_test

        # valor de la precisión general del modelo entrenado
        self.exactitud[tipo] = 100 * self.metricas[tipo]['categorical_accuracy']
        print("La exactitud del modelo: {:5.2f}%".format(
            100 * self.metricas[tipo]['categorical_accuracy']))
        # Función para graficar las matrices de confusión y las gráficas de
        # entrenamiento
        f.Graficas(
            self.direccion, cnn, self.confusion[tipo], self.nombre_clases,
            tipo)

        # Actualiza la variable para hacer seguimiento al progreso
        print('Concluye el entrenamiento del clasificador')
        self.ActualizarProgreso(tipo, 0.90)
        # Entrenado
        
        print('Ajustando selección mediante post procesamiento')
        # combinación de ventas de salida
        if self.agrupar_ventanas:
            num_vent_agrupar = int(self.tam_ventana_ms/self.paso_ms)
            self.prediccion = f.DeterminarClase(
                self.prediccion, num_vent_agrupar)
            
            self.confusion['Combinada']['Prueba'] = confusion_matrix(
                np.argmax(self.class_test, axis=1), 
                np.argmax(self.prediccion, axis=1))

            f.GraficaMatrizConfusion(
                self.confusion['Combinada']['Prueba'], self.nombre_clases, 
                self.direccion)
        # -----------------------------------------------------------------------------
        # Guardar datos
        print('Guardando información de entrenamiento')
        # Guardar datos de historial de entrenamiento
        # cnn_emg
        # Guardar métricas de entrenamiento
        f.GuardarPkl(
            self.metricas[tipo],
            self.direccion + '/General/metricas_' + tipo + '.pkl')
        
        # calcular precisión por clases mediante matriz de confusión
        presicion_clases, self.exactitud[tipo]  = f.PresicionClases(self.confusion[tipo]['Prueba'])
        presicion_clases = dict(zip(self.nombre_clases, presicion_clases))
        self.exactitud[tipo] = self.exactitud[tipo] * 100
        # Diccionario donde se guardan las métricas de entrenamiento
        info = {
            'Sujeto': self.sujeto, 'Id': self.ubi,
            'Tipo de señales': tipo, 'Exactitud': self.exactitud[tipo]}
        
        # concatenar en un solo diccionario
        info.update(self.metricas[tipo])
        info.update(presicion_clases)
        f.GuardarMetricas(info)
        

    def Combinacion(self):
        """Método Combinación:

        Realiza el entrenamiento del sistema para un tipo de señales,
        es decir no realiza la combinación de estas, hecho de esta
        manera para permitir la ejecución en hilo.

        Parameters
        ----------

        Returns
        -------
        """
        
        # matriz de pesos
        print('Determinar matriz de pesos')
        self.w = f.CalculoPesos(
            self.confusion['EMG']['Validacion'],
            self.confusion['EEG']['Validacion'])

        # vector de decisión
        print('Calculo de predicción combinada')
        self.prediccion['Combinada'] = self.prediccion['EMG'] * self.w[0] + self.prediccion['EEG'] * self.w[1]
        
        # combinación de ventas de salida
        agrupar_ventanas = True
        if agrupar_ventanas:
            num_vent_agrupar = int(self.tam_ventana_ms/self.paso_ms)
            # self.prediccion['Combinada'] = DeterminarClase(
            #     np.argmax(self.prediccion['Combinada'], axis=1), num_vent_agrupar)
            
            self.prediccion['Combinada'] = f.DeterminarClase(
                self.prediccion['Combinada'], num_vent_agrupar)
            # oh = np.zeros(self.prediccion['Combinada'].shape, dtype='int8')
            # for i in range(self.prediccion['Combinada'].shape[0]):
            #     oh[i, np.argmax(self.prediccion['Combinada'][i])] = 1
            # self.prediccion['Combinada'] = DeterminarClase(
            #     oh, num_vent_agrupar)
            
        self.confusion['Combinada'] = confusion_matrix(
            np.argmax(self.class_test, axis=1),
            np.argmax(self.prediccion['Combinada'], axis=1))

        f.GraficaMatrizConfusion(
            self.confusion['Combinada'], self.nombre_clases, self.direccion)

        # Guardar datos
        f.GuardarPkl(self.w, self.direccion + '/Procesamiento/w.pkl')
        # Calculo de exactitud y precisión por clases
        presicion_clases, self.exactitud['Combinada'] = f.PresicionClases(
            self.confusion['Combinada'])
        # convertir a diccionario los valores de precisión
        presicion_clases = dict(zip(self.nombre_clases, presicion_clases))
        self.exactitud['Combinada'] = self.exactitud['Combinada'] * 100

        info = {
            'Sujeto': self.sujeto, 'Id': self.ubi,
            'Tipo de señales': 'Combinada',
            'Exactitud': (self.exactitud['Combinada'])}
        # concatenar en un solo diccionario
        # se agregan dos espacios vacios para estandarizar Rendimiento.csv
        info.update({'loss':None, 'categorical_accuracy': None})
        info.update(presicion_clases)
        f.GuardarMetricas(info)
        # Actualizar el valor del progreso
        self.ActualizarProgreso('General', 0.99)
        

    def CombinacionCargada(self, crear_directorio=True):
        """Calcular la matriz de confusión de los mejores resultados
        """
        # se deven calcular de nuevo las predicciones para que las
        # ventanas correspondan
        # -----------------------------------------------------------------------------
        # Las ventanas
        test = dict.fromkeys(['EEG', 'EMG'])
        print('Realizando combinación...')
        for tipo in ['EEG', 'EMG']:
            print('Cargando datos de señales de ' + tipo)
            # los datos
            datos = f.ExtraerDatos(self.directorio, self.sujeto, tipo)

            # -----------------------------------------------------------------------------
            # Función para sub muestreo
            # donde se guardan los datos
            senales = dict.fromkeys(self.canales[tipo])
            for canal in self.canales[tipo]:
                senales[canal] = []
            for sesion in range(1,4):
                senales_subm, _ = f.Submuestreo(
                    self.directorio, tipo, datos, self.sujeto, sesion,
                    self.canales[tipo], self.nombre_clases, self.filtro[tipo],
                    self.m[tipo])
                for canal in self.canales[tipo]:
                    senales[canal].append(senales_subm[canal])
                del senales_subm
            
            print('Cargado')
            # Enventanado

            # -----------------------------------------------------------------------------
            # Registros
            print('Cargando registros de ' + tipo)
            # Cada registro es de 13 segundos, de la siguiente manera: 
            # 4 segundos para reposo, 
            # 3 segundo donde se presenta una pista visual
            # 4 segundo para ejecutar el movimiento
            tam_registro = self.tam_registro_s*self.frec_submuestreo[tipo]
            # donde se guardarán los registros
            registros_test = dict.fromkeys(self.canales[tipo])
            for canal in self.canales[tipo]:
                registros_test[canal] = []

            # las clases de los registros
            clases_regis_test = []

            for sesion in range(3):
                # Traducir las banderas a valores en submuestreo
                # Revisar que esta traducción sea correcta
                banderas = (datos['Banderas'][sesion][1::2]
                            - datos['Inicio grabacion'][sesion])/self.m[tipo]
                banderas = banderas.astype(int)
                clases = datos['One Hot'][sesion][:,::2]
                num_registros = len(datos['Banderas'][sesion][::2])
                regis = dict.fromkeys(self.canales[tipo])
                for canal in self.canales[tipo]:
                    regis[canal] = np.empty([num_registros, tam_registro])
                del canal
                
                # para iteracionar entre los registros
                i = 0
                for bandera in banderas:
                    for canal in self.canales[tipo]:
                        regis[canal][i,:] = senales[canal][sesion][bandera-tam_registro:bandera]
                    i += 1
                # concatenar a registros
                # clases
                clases_regis_test.append(
                    clases[:,self.registros_id['test'][sesion]])
                del clases
                # registros
                for canal in self.canales[tipo]:
                    registros_test[canal].append(regis[canal][self.registros_id['test'][sesion]])
                del regis, canal
                
            del bandera, banderas, num_registros, senales, tam_registro, datos
            
            # Actualiza la variable para hacer seguimiento al progreso
            print('Se divide los registros para ' + tipo)
            self.ActualizarProgreso(tipo, 0.50)
            print('Cargados')
            # -----------------------------------------------------------------------------
            # Descarte de datos ambiguos
            print('Cargando sincronizadas para ' + tipo)
            # Valores para descarte:
            # traducción de tiempos de descarte y reclamador a número de muestras
            descarte = dict.fromkeys(['Activo', 'Reposo'])
            descarte['Activo'] = int(
                self.descarte_ms[tipo]['Activo'] * self.frec_submuestreo[tipo] / 1000)
            descarte['Reposo'] = int(
                self.descarte_ms[tipo]['Reposo'] * self.frec_submuestreo[tipo] / 1000)
            reclamador = dict.fromkeys(['Activo', 'Reposo'])
            reclamador['Activo'] = int(
                self.reclamador_ms[tipo]['Activo'] * self.frec_submuestreo[tipo] / 1000)
            reclamador['Reposo'] = int(
                self.reclamador_ms[tipo]['Reposo'] * self.frec_submuestreo[tipo] / 1000)

            prueba, class_test = f.Ventanas(
                registros_test, clases_regis_test, self.num_canales[tipo],
                self.num_clases, reclamador, descarte,
                self.tam_ventana[tipo], self.paso_ventana[tipo],
                7*self.frec_submuestreo[tipo])
            del registros_test, clases_regis_test, descarte, reclamador

            # Actualiza la variable para hacer seguimiento al progreso
            print('Cargadas')
            self.ActualizarProgreso(tipo, 0.55)

            # determinar el número de ventanas
            self.num_ventanas['Prueba'] = len(class_test)
            # -----------------------------------------------------------------------------
            # Extracción de características
            # Cálculo de FastICA
            # aplicar ICA
            if self.calcular_ica[tipo]:
                # aplicar transformaciones a las ventanas
                print('Cargando ICA ' + tipo)
                prueba = f.AplicarICA(
                    self.num_ventanas[tipo]['Prueba'], self.num_ci[tipo],
                    self.tam_ventana[tipo], self.ica_total[tipo], prueba)
                print('Cargado')
            
            # para calcular el csp
            test[tipo] = self.csp[tipo].transform(prueba)
            # se asigna las señales a test[tipo]
            # test[tipo] = prueba
            del prueba

        # Balanceo doble aplicado a todas las clases
        balancear = False
        if balancear:
            tipos_clases = np.identity(self.num_clases, dtype='int8')
            for clase in tipos_clases:
                test['EEG'], test['EMG'], class_test= f.BalanceDoble(
                    test['EEG'], test['EMG'], class_test, clase)
        self.class_test = class_test
        del class_test

        for tipo in ['EEG', 'EMG']:
            print('Realizando la clasificación de ventas de prueba')
            # Determinar predicción
            self.prediccion[tipo] = self.modelo[tipo].predict(test[tipo])
            print('Clasificadas')

        if crear_directorio:
            # Crear un nuevo directorio
            self.direccion, self.ubi = f.Directorios(self.sujeto)
        # Luego de calcular las predicciones conjuntas se combina
        self.Combinacion()

    def CargarDatos(self, tipo):
        """Método CargarDatos:

        Parameters
        ----------
        tipo: STR, el tipo de señales para entrenamiento, puede ser
            'EMG' o 'EEG'

        Returns
        -------
        """
        # -----------------------------------------------------------------------------
        # Extraer datos
        datos = f.ExtraerDatos(self.directorio, self.sujeto, tipo)

        # Actualiza la variable para hacer seguimiento al progreso
        self.progreso[tipo] = 0.15
        # -----------------------------------------------------------------------------
        # Filtro
        self.filtro[tipo] = f.AbrirPkl(
            self.direccion + '/Procesamiento/filtro_' + tipo + '.pkl')

        # Actualiza la variable para hacer seguimiento al progreso
        self.progreso[tipo] = 0.20
        # -----------------------------------------------------------------------------
        # Diseñar las clases One Hot

        # Actualiza la variable para hacer seguimiento al progreso
        self.progreso[tipo] = 0.25
        # -----------------------------------------------------------------------------
        # Function para sub muestreo

        # Calcular a partir de frecuencias de sub muestreo
        self.frec_submuestreo[tipo] = int(
            datos['Frecuencia muestreo'] / self.m[tipo])

        self.tam_ventana[tipo] = int(
            self.tam_ventana_ms * 0.001 * self.frec_submuestreo[tipo])
        self.paso_ventana[tipo] = int(
            self.paso_ms * 0.001 * self.frec_submuestreo[tipo])

        # Actualiza la variable para hacer seguimiento al progreso
        self.ActualizarProgreso(tipo, 0.44)
        # -----------------------------------------------------------------------------
        # Enventanado

        # Actualiza la variable para hacer seguimiento al progreso
        self.ActualizarProgreso(tipo, 0.50)
        # -----------------------------------------------------------------------------
        # División y balanceo del dataset
        # Descarte de datos de imaginación motora (Revisar si hacer)

        # -----------------------------------------------------------------------------
        # Dividir datos de entrenamiento, test y validación; Además se realiza 
        # el balanceo de base de datos mediante sub muestreo aleatorio
        # División y balanceo de dataset datos EMG

        # Actualiza la variable para hacer seguimiento al progreso
        self.ActualizarProgreso(tipo, 0.55)
        # -----------------------------------------------------------------------------
        # Extracción de características
        # Cargar FastICA entrenado
        if self.calcular_ica[tipo]:
            self.ica_total[tipo] = f.AbrirPkl(
                self.direccion + '/Procesamiento/ica_' + tipo + '.pkl')
        else:
            self.num_ci[tipo] = self.num_canales[tipo]

        # Actualiza la variable para hacer seguimiento al progreso
        self.ActualizarProgreso(tipo, 0.77)
        # -----------------------------------------------------------------------------
        # Clasificador
        # Cargar el modelo
        # se cargan lo puntos de control
        # if tipo == 'EMG':
        #     self.modelo[tipo] = f.ClasificadorEMG(
        #         self.num_ci[tipo], self.tam_ventana[tipo], self.num_clases)
        # elif tipo == 'EEG':
        #     self.modelo[tipo] = f.ClasificadorEEG(
        #         self.num_ci[tipo], self.tam_ventana[tipo], self.num_clases)

        # # Loads the weights
        # self.modelo[tipo].load_weights(
        #     self.direccion + '/Clasificador/' + tipo + "/" + tipo + "_cp.ckpt")
        
        # se carga el modelo
        self.modelo[tipo] = load_model(
            self.direccion + '/Clasificador/' + tipo + "/" + tipo + "_modelo.h5")

        # Cargar matriz de confusión
        conf = f.AbrirPkl(
            self.direccion + '/Procesamiento/CM_val_' + tipo + '.pkl')
        self.confusion[tipo]['Validacion'] =  conf
        del conf
        # Cargar metricas
        self.metricas[tipo] = f.AbrirPkl(
            self.direccion + '/General/metricas_' + tipo + '.pkl')
        # valor de la presisión general del modelo entrenado
        self.exactitud[tipo] = 100 * self.metricas[tipo]['categorical_accuracy']
        print("La exactitud del modelo: {:5.2f}%".format(
            100 * self.metricas[tipo]['categorical_accuracy']))

        # Actualiza la variable para hacer seguimiento al progreso
        self.ActualizarProgreso(tipo, 0.90)

    def CargarCombinacion(self):
        """Método Combinación:

        Realiza el entrenamiento del sistema para un tipo de señales,
        es decir no realiza la combinación de estas, hecho de esta
        manera para permitir la ejecución en hilo.

        Parameters
        ----------

        Returns
        -------
        """
        # matriz de pesos
        self.w = f.AbrirPkl(
            self.direccion + '/Procesamiento/w.pkl')

        # Actualizar el valor del progreso
        self.ActualizarProgreso('General', 0.99)
    
    def DeterminarCanales(self, tipo):
        """
        """
        # Determinar si existe una carpeta donde se evalue el rendimiento
        directo = 'Parametros/Sujeto_' + str(self.sujeto) + '/Canales/'
        # revisar si existe la carpeta
        if not exists(directo):
            f.CrearDirectorio(directo)
        
        # -----------------------------------------------------------------------------
        # lista con los canales disponibles en la base de datos
        if tipo == 'EMG':
            lista_canales = [
                'EMG_1', 'EMG_2', 'EMG_3', 'EMG_4', 'EMG_5', 'EMG_6'
                ]
        elif tipo == 'EEG':
            lista_canales = [
                'FP1', 'AF7', 'AF3', 'AFz', 'F7', 'F5', 'F3', 'F1', 'Fz', 'FT7', 
                'FC5', 'FC3', 'FC1', 'T7', 'C5', 'C3', 'C1', 'Cz', 'TP7', 'CP5',
                'CP3', 'CP1', 'CPz', 'P7', 'P5', 'P3', 'P1', 'Pz', 'PO7', 'PO3',
                'POz', 'FP2', 'AF4', 'AF8', 'F2', 'F4', 'F6', 'F8', 'FC2', 'FC4',
                'FC6', 'FT8', 'C2', 'C4', 'C6', 'T8', 'CP2', 'CP4', 'CP6', 'TP8',
                'P2', 'P4', 'P6', 'P8', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz'
                ]
        
        # lista con las caracteristicas temporales a extraer
        lista_caracteristicas = [
            'potencia de banda', 'cruce por cero', 'desviacion estandar',
            'varianza', 'entropia', 'media', 'rms', 'energia', 
            'longitud de onda', 'integrada', 'ssc'
            ]
        # lista_caracteristicas = [
        #     'potencia de banda', 'cruce por cero', 'desviacion estandar',
        #     'varianza', 'media', 'rms', 'energia', 
        #     'longitud de onda', 'integrada', 'ssc'
        #     ]
        
        # por cada canar hacer el entrenamiento mediante kfolds
        # es necesario entonce sacar la información de los registros
        # por lo cual el procesamiento de las señales se realiza casi que igual a lo de 
        # cargar datos o el de entrenamiento
        
        # Traducir el nombre de los canales a los utlizados en la base de datos
        canales = f.TraducirNombresCanales(lista_canales)
        
        # variable donde guardar la información de los rendimientos obtenidos
        rendimiento =  dict.fromkeys(canales)
        
        for canal in canales:
            # procesamiento de señales
            rendimiento[canal] = []
            
            # los datos
            print('Extrayendo la información de la base de datos para ' + tipo)
            datos = f.ExtraerDatos(self.directorio, self.sujeto, tipo)
            
            # Actualiza la variable para hacer seguimiento al progreso
            print('Información extraida')
            self.ActualizarProgreso(tipo, 0.15)
            # -----------------------------------------------------------------------------
            # Filtro
            print('Diseñando el filtro para ' + tipo)
            self.filtro[tipo] = f.DisenarFiltro(
                self.f_tipo, self.b_tipo, self.f_orden, self.frec_corte[tipo],
                datos['Frecuencia muestreo'])

            # Actualiza la variable para hacer seguimiento al progreso
            print('Diseñado')
            self.ActualizarProgreso(tipo, 0.21)
            
            # Función para sub muestreo
            print('Apicando filtro y submuestreo para ' + tipo)
            
            # donde se guardan los datos
            senales = {canal: []}
            clases_OH = []
            for sesion in range(1,4):
                senales_subm, clases = f.Submuestreo(
                    self.directorio, tipo, datos, self.sujeto, sesion,
                    [canal], self.nombre_clases, self.filtro[tipo],
                    self.m[tipo])
                clases_OH.append(clases)
                del clases
                senales[canal].append(senales_subm[canal])
                del senales_subm
                
            # Calcular a partir de frecuencias de sub muestreo
            self.frec_submuestreo[tipo] = int(
                datos['Frecuencia muestreo'] / self.m[tipo])
            self.tam_ventana[tipo] = int(
                self.tam_ventana_ms * 0.001 * self.frec_submuestreo[tipo])
            self.paso_ventana[tipo] = int(
                self.paso_ms * 0.001 * self.frec_submuestreo[tipo])

            # Actualiza la variable para hacer seguimiento al progreso
            print('Aplicados')
            self.ActualizarProgreso(tipo, 0.44)
            # -----------------------------------------------------------------------------
            # Registros
            print('Dividiendo registros para ' + tipo)
            # Cada registro es de 13 segundos, de la siguiente manera: 
            # 4 segundos para reposo, 
            # 3 segundo donde se presenta una pista visual
            # 4 segundo para ejecutar el movimiento
            tam_registro = self.tam_registro_s*self.frec_submuestreo[tipo]
            
            # donde se guardan los datos
            registros_train = {canal: []}
            # las clases de los registros
            clases_regis_train =[]
            
            for sesion in range(3):
                # Traducir las banderas a valores en submuestreo
                # Revisar que esta traducción sea correcta
                banderas = (datos['Banderas'][sesion][1::2]
                            - datos['Inicio grabacion'][sesion])/self.m[tipo]
                banderas = banderas.astype(int)
                clases = datos['One Hot'][sesion][:,::2]
                num_registros = len(datos['Banderas'][sesion][::2])
                regis = {canal: np.empty([num_registros, tam_registro])}
                
                # para iteractuar entre los registros
                i = 0
                for bandera in banderas:
                    regis[canal][i,:] = senales[canal][sesion][bandera-tam_registro:bandera]
                    # regis[i,:,:] = senales[sesion][:,bandera-tam_registro:bandera]
                    i += 1
                del i
                
                # Concatenar los registros
                # revisar si funciona de manera correcta esta extracción de datos
                registros_train[canal].append(
                    regis[canal][
                        np.concatenate(
                            (self.registros_id['train'][sesion], 
                             self.registros_id['val'][sesion]))])
                del regis
                # para las clases
                clases_regis_train.append(
                    clases[:,np.concatenate(
                        (self.registros_id['train'][sesion], 
                         self.registros_id['val'][sesion]))])
            del clases, bandera, banderas, num_registros, senales
            
            # Actualiza la variable para hacer seguimiento al progreso
            print('Divididos')
            self.ActualizarProgreso(tipo, 0.50)
            # -----------------------------------------------------------------------------
            # Descarte de datos ambiguos
            print('Diseñando ventanas para ' + tipo)
            # Valores para descarte:
            # traducción de tiempos de descarte y reclamador a número de muestras
            descarte = dict.fromkeys(['Activo', 'Reposo'])
            descarte['Activo'] = int(
                self.descarte_ms[tipo]['Activo'] * self.frec_submuestreo[tipo] / 1000)
            descarte['Reposo'] = int(
                self.descarte_ms[tipo]['Reposo'] * self.frec_submuestreo[tipo] / 1000)
            reclamador = dict.fromkeys(['Activo', 'Reposo'])
            reclamador['Activo'] = int(
                self.reclamador_ms[tipo]['Activo'] * self.frec_submuestreo[tipo] / 1000)
            reclamador['Reposo'] = int(
                self.reclamador_ms[tipo]['Reposo'] * self.frec_submuestreo[tipo] / 1000)

            # calculo de las ventanas
            x, y = f.Ventanas(
                registros_train, clases_regis_train, 1,
                self.num_clases, reclamador, descarte,
                self.tam_ventana[tipo], self.paso_ventana[tipo],
                7*self.frec_submuestreo[tipo])
            del registros_train, clases_regis_train
            
            # Actualiza la variable para hacer seguimiento al progreso
            print('Diseñadas')
            self.ActualizarProgreso(tipo, 0.55)
            # -----------------------------------------------------------------------------
            # Balancear ventanas
            if self.balancear:
                print('Balanceando ventanas de ' + tipo)
                # La inicialización del balance se hace para conservar las 
                # variables anteriores y poder compararlas
                clases = np.identity(self.num_clases, dtype='int8')
                # # inicialización
                # x, y = f.Balanceo(
                #     x, y, clases[-1])

                # En el caso de que se requiera realizar en todas las clases
                for i in range(self.num_clases - 1):
                    x, y = f.Balanceo(
                        x, y, clases[i])
                
                print('Se balancean los datos para ' + tipo)
            
            """
            # revisar si la creaciòn de ventanas es correcta
            # revisar si las corresponde tanto para las ventanas de las
            # señales de EEG como EMG
            
            from niapy.task import Task
            from niapy.algorithms.basic import ParticleSwarmOptimization
            # Seleción de caracteristicas
            problem = f.SVMFeatureSelection(train, class_train)
            task = Task(problem, max_iters=100)
            algorithm = ParticleSwarmOptimization(population_size=10, seed=1234)
            best_features, best_fitness = algorithm.run(task)
            """
            
            # numero_ventanas = len(y)
            # extracciòn de caracteristicas
            
            # división k folds
            print('Se inica evaluación iterativa mediante K-folds')
            # kfolds = KFold(n_splits=10)
            # usar shcle split ya que con el otro no se puede hacer 
            # menos entrenamientos sin dividir más el dataset
            kfolds = ShuffleSplit(n_splits=10, test_size=self.porcen_prueba)
            
            modelo = f.ClasificadorCanales(1, self.tam_ventana[tipo], self.num_clases)
            # ciclo de entrenamiento:
            for i, (train_index, test_index) in enumerate(kfolds.split(x)):
                print(str(i+1) + 'º iteración para el canal ' + canal)
                # Diviciòn de los k folds
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                # calcular csp y extraer caracteristicas
                # Calculo de CSP
                csp = CSP(
                    n_components=1, reg=None, log=None, 
                    norm_trace=False, transform_into='csp_space')
                
                # para calcular el csp la clases deven ser categoricas
                x_train = csp.fit_transform(
                    x_train, np.argmax(y_train, axis=1))
                x_test = csp.transform(x_test)
                
                x_train = f.Caracteristicas(x_train, lista_caracteristicas)
                x_test = f.Caracteristicas(x_test, lista_caracteristicas)
                
                # clasificador a utilizar
                modelo.fit(
                    x_train, y_train, shuffle=True, epochs=int(self.epocas/2), 
                    batch_size=self.lotes)
                eva = modelo.evaluate(
                    x_test, y_test, verbose=1, return_dict=True)
                   
                rendimiento[canal].append(eva)
                # entrenar y evaluar la clasificaciòn
                # guardar el rendimiento obtenido
            
        # Evaluaciòn del rendimiento usando pandas
        print(rendimiento)
        f.GuardarPkl(rendimiento, directo + 'rendimiento_' + tipo)
        # exactitud_canales = pd.dataframe()
        # loss_canales = pd.dataframe()
            # sacar promedio de entrenamiento por cada k fold
            # y desviaciòn estandar
        # comparar los promedios obtenidos en cada canal
        # se realiza ranking con canales con mejor rendimiento
        
        # Seleccion de canal
        self.canales[tipo] = f.ElegirCanales(
            rendimiento, directo, tipo, determinar=True)

    def DeterminarRegistros(self, guardar=True):
        """Método para determinar los registros a usar
        
        Determina si ya se dividieron los registros en 
        entrenamiento, prueba y validación
        
        Los registros son guardados en un archivo .pkl
        En la carpeta del sujeto

        Returns
        -------
        None.

        """
        
        # Revisa si ya existe "Registros.pkl
        existe = exists('Parametros/Sujeto_' + str(self.sujeto) + '/Registros.pkl')
        
        # en el caso de que exista
        if existe:
            self.registros_id = f.AbrirPkl(
                'Parametros/Sujeto_' + str(self.sujeto) + '/Registros.pkl')
            
            # de no existir
        else:
            # Para dividir los registros
            # sacar los datos de dataset
            # se toma las de EMG ya que son más pequeñas
            datos = f.ExtraerDatos(self.directorio, self.sujeto, 'EMG')
            
            self.registros_id['train'] = []
            self.registros_id['val'] = []
            self.registros_id['test'] = []
            for sesion in range(3):
                regis_id = np.arange(len(datos['Banderas'][sesion][::2]))
                train, test = train_test_split(
                    regis_id, test_size=self.porcen_prueba)
                train, val = train_test_split(
                    train, test_size=self.porcen_validacion)
                self.registros_id['train'].append(train)
                self.registros_id['val'].append(val)
                self.registros_id['test'].append(test)
                del train, val, test, regis_id
            del datos
            
            # se guardan los registros
            if guardar:
                # revisar si existen las carpetas donde guardar los parametros
                f.Directorios(self.sujeto, sin_sesion=True)
                
                f.GuardarPkl(
                    self.registros_id, 'Parametros/Sujeto_' + str(self.sujeto) + '/Registros.pkl')
        
        pass
    
    
    def SeleccionCanales(self, sujetos, tipo, lista_canales):
        """
        Método encargado de la selección de canales

        Parameters
        ----------
        sujetos : LIST
            Identificación de los sujetos en INT.
        lista_canales : LIST
            Lista de canales a calcular en STR.

        Returns
        -------
        None.

        """
        
        # Determinar si existe una carpeta donde se evalue el rendimiento
        directo = 'Parametros/Sujeto_' + str(self.sujeto) + '/Canales/'
        # revisar si existe la carpeta
        if not exists(directo):
            f.CrearDirectorio(directo)
        
        # lista de caracteristicas a calcular
        lista_caracteristicas = [
            'potencia de banda', 'desviacion estandar',
            'media', 'rms']
        
        # Traducir el nombre de los canales a los utlizados en la base de datos
        canales = f.TraducirNombresCanales(lista_canales)
        rendimiento = dict.fromkeys(canales)
        for canal in canales:
            rendimiento[canal] = []
        
        #calcular numero de ventanas y numero de muestras
        
        for n_canal, canal in enumerate(canales):
            print('Se inica evaluación iterativa mediante K-folds')
            print('Evaluando canal ' + canal + ' de ' + tipo)
            kfolds = ShuffleSplit(n_splits=4, test_size=0.10) # 4 diviciones
            modelo = f.ClasificadorUnico(
                len(lista_caracteristicas), 0, self.num_clases)
            
            ventanas, clases = f.CargarVentanas(
                tipo, sujetos, [canal], clases=True)
            
            # ciclo de entrenamiento:
            for i, (train_index, test_index) in enumerate(kfolds.split(ventanas)):
                print(str(i+1) + 'º iteración para el canal ' + canal + ' de ' + tipo)
                # division k-fols
                x_train = ventanas[train_index] # Revisar que la división sea correcta
                x_test = ventanas[test_index]
                y_train, y_test = clases[train_index], clases[test_index]
                
                # calculo CSP
                csp = CSP(
                    n_components=1, reg=None, log=None, 
                    norm_trace=False, transform_into='csp_space')
                x_train = csp.fit_transform(
                    x_train, np.argmax(y_train, axis=1))
                # calculo caracteristicas
                x_train = f.Caracteristicas(
                    x_train, lista_caracteristicas, csp=csp)
                # para entrenamiento
                x_test = csp.transform(x_test)
                x_test = f.Caracteristicas(
                    x_test, lista_caracteristicas, csp=csp)
                del csp
                
                # clasificador a utilizar
                modelo.fit(
                    x_train, y_train, shuffle=True, epochs=int(self.epocas/2), 
                    batch_size=self.lotes) # 32 epocas
                del x_train, y_train
                eva = modelo.evaluate(
                    x_test, y_test, verbose=1, return_dict=True)
                del x_test, y_test
                
                rendimiento[canal].append(eva)
                # entrenar y evaluar la clasificaciòn
                # guardar el rendimiento obtenido
        
        del ventanas, clases, train_index, test_index
        # Evaluaciòn del rendimiento usando pandas
        # Seleccion de canal
        print(rendimiento)
        f.GuardarPkl(rendimiento, directo + 'rendimiento_' + tipo)
        
        # Seleccion de canal
        self.canales[tipo] = f.ElegirCanales(
            rendimiento, directo, tipo, determinar=True)
        self.num_canales[tipo] = len(self.canales[tipo])
        
    
    def SeleccionCaracteristicas(self, sujetos, tipo):
        """
        Método para la selección de caracteristicas

        Parameters
        ----------
        sujetos : LIST
            Lista de sujetos en formato INT
        tipo : STR
            Descripción del tipo de señales a usar: 'EEG' ó 'EMG'

        Returns
        -------
        None.

        """
        # Cargar ventanas
        ventanas, clases = f.CargarVentanas(
            tipo, sujetos, self.canales[tipo], clases=True)
        # divición de ventanas
        x_train, x_test, y_train, y_test = train_test_split(
            ventanas, clases, test_size=self.porcen_prueba, stratify=clases)
        del ventanas, clases
        
        print('Iniciando selección de caracteristicas')
        # Calculo de CSP
        csp = CSP(
            n_components=len(self.canales[tipo]), reg=None, log=None, 
            # norm_trace=False, transform_into='csp_space')
            norm_trace=False, transform_into='average_power')
        # para calcular el csp la clases deben ser categoricas
        x_train = csp.fit_transform(
            x_train, np.argmax(y_train, axis=1))
        # x_train = f.Caracteristicas(x_train, self.caracteristicas[tipo], csp=csp)
        # para prueba
        x_test = csp.transform(x_test)
        # x_test, feature_names = f.Caracteristicas(
        #     x_test, self.caracteristicas[tipo], csp=csp, 
        #     canales=self.canales[tipo], generar_lista=True)
        del csp
        
        print('Ejecutando PSO')
        # problem = f.SVMFeatureSelection(X_train, y_train)
        problem = f.MLPFeatureSelection(x_train, y_train)
        task = Task(problem, max_iters=16) #16
        algorithm = ParticleSwarmOptimization(population_size=16) #16
        best_features, best_fitness = algorithm.run(task)
        
        # # Selección
        # feature_names = np.array(feature_names, dtype='str')
        feature_names = np.array(
            [canal + ': potencia de banda' for canal in self.canales[tipo]], 
            dtype='str')
        selected_features = best_features > 0.5
        print('Number of selected features:', selected_features.sum())
        print(
            'Selected features:', 
            ', '.join(feature_names[selected_features].tolist()))
        
        # Evaluación de selecciones
        print('Evaluando selección')
        model_selected = f.ClasificadorMultiple(
            int(sum(selected_features)), self.num_clases)
        model_selected.fit(
            x_train[:, selected_features], y_train, shuffle=True, 
            epochs=self.epocas, batch_size=self.lotes, verbose=1) # epocas 128
        ren_sel =  model_selected.evaluate(
            x_test[:, selected_features], y_test, verbose=1, 
            return_dict=False)[1]
        print('Subset accuracy:', ren_sel)
        
        model_all = f.ClasificadorMultiple(
            len(selected_features), self.num_clases)
        model_all.fit(
            x_train, y_train, shuffle=True, epochs=self.epocas, 
            batch_size=self.lotes, 
            verbose=1) # epocas 128
        ren_todas = model_all.evaluate(
            x_test, y_test, verbose=1, return_dict=False)[1]
        print('All Features Accuracy:', ren_todas)
        del x_test, x_train, y_test, y_train
        self.parcial[tipo] = f.CrearRevision(feature_names.tolist(), best_features)
        
        # guardar datos
        directo = 'Parametros/Sujeto_' + str(sujetos) + '/Canales/'
        f.GuardarPkl(self.parcial[tipo], directo + 'resultados_' + tipo)
        # creo que ya no son necesarias
        self.caracteristicascanal[tipo] = f.SeleccionarCaracteristicas(
            self.parcial[tipo])
        # self.caracteristicas[tipo] = f.SeleccionarCaracteristicas(
        #     self.parcial[tipo])
    
    
    def Preprocesamiento(self, tipo, sujeto, guardar_clases=True):
        """ Se realiza el bloque de preprocesamiento de señales
        
        Se va a guardar los datos enventanados por cada canal
        de todos los sujetos ingresados aquí, luego se revisará
        como concatenar dichas ventanas para hacer el calculo de 
        los CSP

        Returns
        -------
        None.

        """
        # Crear carpetas donde guardar los datos
        directo = 'Datos/'
        
        if not exists(directo):
            f.CrearDirectorio(directo)
        
        guardar_clases = guardar_clases
        for canal in self.canales[tipo]:
            datos = f.ExtraerDatos(self.directorio, sujeto, tipo)
            
            # Calcular a partir de frecuencias de sub muestreo
            self.frec_submuestreo[tipo] = int(
                datos['Frecuencia muestreo'] / self.m[tipo])
            self.tam_ventana[tipo] = int(
                self.tam_ventana_ms * 0.001 * self.frec_submuestreo[tipo])
            self.paso_ventana[tipo] = int(
                self.paso_ms * 0.001 * self.frec_submuestreo[tipo])
                
            self.filtro[tipo] = f.DisenarFiltro(
                self.f_tipo, self.b_tipo, self.f_orden, self.frec_corte[tipo],
                datos['Frecuencia muestreo'])
            
            senales = []
            #clases_OH = []
            for sesion in range(1,4):
                senales_subm, _ = f.Submuestreo(
                    self.directorio, tipo, datos, sujeto, sesion,
                    [canal], self.nombre_clases, self.filtro[tipo],
                    self.m[tipo])
                # revisar que la concatenación sea buena
                senales.append(senales_subm)
                #clases_OH.append(clases)
            del senales_subm
            
            # donde se guardan los datos
            registros = {canal: []}
            # la parte de los registros
            clases_regis = []
            
            for sesion in range(3):
                tam_registro = self.tam_registro_s*self.frec_submuestreo[tipo]
                num_registros = len(datos['Banderas'][sesion][::2])  
                # Traducir las banderas a valores en submuestreo
                # Revisar que esta traducción sea correcta
                banderas = (datos['Banderas'][sesion][1::2]
                            - datos['Inicio grabacion'][sesion])/self.m[tipo]
                banderas = banderas.astype(int)
                regis = np.empty([num_registros, tam_registro])
            
                # para iteractuar entre los registros
                i = 0
                for bandera in banderas:
                    regis[i,:] = senales[sesion][canal][bandera-tam_registro:bandera]
                    # regis[i,:,:] = senales[sesion][:,bandera-tam_registro:bandera]
                    i += 1
                del i
                # para las clases
                registros[canal].append(regis)
                clases_regis.append(datos['One Hot'][sesion][:,::2])
                    
            # Valores para descarte:
            # traducción de tiempos de descarte y reclamador a número de muestras
            descarte = dict.fromkeys(['Activo', 'Reposo'])
            descarte['Activo'] = int(
                self.descarte_ms[tipo]['Activo'] * self.frec_submuestreo[tipo] / 1000)
            descarte['Reposo'] = int(
                self.descarte_ms[tipo]['Reposo'] * self.frec_submuestreo[tipo] / 1000)
            reclamador = dict.fromkeys(['Activo', 'Reposo'])
            reclamador['Activo'] = int(
                self.reclamador_ms[tipo]['Activo'] * self.frec_submuestreo[tipo] / 1000)
            reclamador['Reposo'] = int(
                self.reclamador_ms[tipo]['Reposo'] * self.frec_submuestreo[tipo] / 1000)
            
            # calculo de las ventanas
            ventanas, clases = f.Ventanas(
                registros, clases_regis, 1,
                self.num_clases, reclamador, descarte,
                self.tam_ventana[tipo], self.paso_ventana[tipo],
                7*self.frec_submuestreo[tipo])
                    
            if guardar_clases:
                f.GuardarPkl(
                    clases, directo + 'clases_sub_' + str(sujeto))
                guardar_clases = False
            # Guardada los canales
            f.GuardarPkl(
                ventanas, directo + tipo + '_' + canal + '_sub_' + str(sujeto))
            

    
    def ExtraccionCaracteristicas(
            self, tipo, sujetos, entrenar=False, clases=True):
        """Se realiza la extracción de caracteristicas, no confundir con la
        selección
        

        Parameters
        ----------
        tipo : TYPE
            DESCRIPTION.
        sujetos : TYPE
            DESCRIPTION.
        entrenar : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        propio = False
        
        if entrenar:
            # # Crear carpetas donde guardar los datos
            # directo = 'Datos/'
            # if not exists(directo):
            #     f.CrearDirectorio(directo)
            ventanas, clases = f.CargarVentanas(
                tipo, sujetos, self.canales[tipo], clases=True)
            
            if not propio:
                # Calculo de CSP
                self.csp[tipo] = CSP(
                    n_components=self.num_canales[tipo], reg=None, log=None,
                    norm_trace=False, transform_into='average_power')
                
                # si algo seria cambiar y añadir algo aquí
                
                cara  = self.csp[tipo].fit_transform(
                    ventanas, np.argmax(clases, axis=1))
                del ventanas
            else:
                # Calculo de CSP
                self.csp[tipo] = CSP(
                    n_components=self.num_canales[tipo], reg=None, log=None,
                    norm_trace=False, transform_into='csp_space')
                ventanas  = self.csp[tipo].fit_transform(
                    ventanas, np.argmax(clases, axis=1))
                cara = f.Caracteristicas(
                    ventanas, self.caracteristicas[tipo], csp=self.csp[tipo])
                del ventanas
            # en datos se guardan unicamente los datos que se vayan a usar de
            # ya sea para entrenamiento o prueba, de momento se descarta
            # guardarlo dentro de las ubi
            directorio = 'Parametros/'
            f.GuardarPkl(self.csp[tipo], directorio + tipo + '_CSP')
            f.GuardarPkl(cara, 'Datos/' + tipo + '_cara_entrenar')
            f.GuardarPkl(clases, 'Datos/' + tipo + '_clases_cara_entrenar')
        
        else:
            if clases:
                ventanas, clases = f.CargarVentanas(
                    tipo, sujetos, self.canales[tipo], clases=True)
                # guardar clases
                f.GuardarPkl(clases, 'Datos/' + tipo + '_clases_cara_probar')
                del clases
            else:
                ventanas = f.CargarVentanas(
                    tipo, sujetos, self.canales[tipo], clases=False)
            
            if self.csp[tipo] is None:
                directo = 'Parametros/'
                self.csp[tipo] = f.AbrirPkl(directo + tipo + '_CSP.pkl')
            
            if propio:
                ventanas = self.csp[tipo].transform(ventanas)
                cara = f.Caracteristicas(
                    ventanas, self.caracteristicas[tipo], csp=self.csp[tipo])
            else:
                cara = self.csp[tipo].transform(ventanas)
            del ventanas
            f.GuardarPkl(cara, 'Datos/' + tipo + '_cara_probar')
   
   
    def Clasificacion(self, sujetos, entrenar=True, graficar=True):
        """
        

        Parameters
        ----------
        entrenar : TYPE, optional
            sobre si se enctrena un modelo de lo contrario unicamente 
            se carga. El predeterminado es True.
        graficar : TYPE, optional
            Se imprimen graficas, en el caso de entrenar == True, el 
            historial de entrenamiento a demás de las matrices de
            confución, en el caso de entrenar==False, solo se 
            imprimen las matrices.

        Returns
        -------
        None.

        """
        # abrir las caracteristicas calculadas
        directo = 'Datos/'
        direccion = 'Parametros/Sujeto_' + str(sujetos) + '/Canales/'
        carac = dict.fromkeys(['EEG', 'EMG'])
        seleccion = dict.fromkeys(['EEG', 'EMG'])
        for tipo in ['EEG', 'EMG']:
            if entrenar:
                carac[tipo] = f.AbrirPkl(
                    directo + tipo + '_cara_entrenar.pkl')
            else:
                carac[tipo] = f.AbrirPkl(
                    directo + tipo + '_cara_probar.pkl')
            self.parcial[tipo] = f.AbrirPkl(
                direccion + 'resultados_' + tipo + '.pkl')
            seleccion[tipo] = np.array(
                self.parcial[tipo]['Rendimiento'], dtype='float') > 0.5
        
        # seleccionando con PSO
        caracteristicas = np.concatenate(
            (carac['EEG'][:, seleccion['EEG']], 
            carac['EMG'][:, seleccion['EMG']]), axis=1)
        
        # donde se guarda el clasificador
        directo_modelo =  self.direccion + '/Clasificador/'
        
        if entrenar:
            clases = f.AbrirPkl(directo + 'EMG_clases_cara_entrenar.pkl')
            
            # dividir datos de validación y entrenamiento 
            train, val, class_train, class_val = train_test_split(
                caracteristicas, clases, test_size=self.porcen_validacion, 
                stratify= clases)
            
            num_cara = np.shape(val)[1]
            # creación de modelo a entrenar
            self.modelo = f.ClasificadorMultiple(num_cara, self.num_clases)
            
            mlp = self.modelo.fit(
               train, class_train, shuffle=True, epochs=self.epocas, 
               batch_size=self.lotes, validation_data=(val, class_val))
            
            self.modelo.save(directo_modelo + 'modelo.h5')
            
            # Para las matrices de confución
            # A los datos de validación
            prediccion_val = self.modelo.predict(val)
            confusion_val = confusion_matrix(
                np.argmax(class_val, axis=1), np.argmax(prediccion_val, axis=1))
            
            # Guardar y graficar información de entrenamiento
            f.GraficarEntrenamiento(self.direccion + '/General/', mlp)
            f.Graficar(
                self.direccion + '/General/', confusion_val, self.nombre_clases, 
                titulo='Validación')
            
        else:
            # cargar el clasificador entrenado
            if self.modelo is None:
                self.modelo = load_model(
                    self.direccion + '/Clasificador/modelo.h5')
                
            clases = f.AbrirPkl(directo + 'EMG_clases_cara_probar.pkl')
            
            eva = self.modelo.evaluate(
                caracteristicas, clases, verbose=1, return_dict=True)
            print("La precición del modelo: {:5.2f}%".format(
                100 * eva['categorical_accuracy']))
            
            # Aplicar a los datos de prueba
            prediccion = self.modelo.predict(caracteristicas)
            # pos procesamiento, asignación final de clase
            # num_vent_agrupar = int(self.tam_ventana_ms/self.paso_ms)
            # prediccion = f.DeterminarClase(prediccion, num_vent_agrupar)
            
            confusion_pru = confusion_matrix(
                np.argmax(clases, axis=1), np.argmax(prediccion, axis=1))
            
            _, exactitud = f.PresicionClases(
                confusion_pru)
            
            # Graficar y guardar metricas de prueba
            f.Graficar(
                self.direccion + '/General/', confusion_pru, self.nombre_clases, 
                titulo='Prueba')
            f.GuardarPkl(
                eva, self.direccion + '/General/metricas.pkl')
            f.GuardarPkl(exactitud, self.direccion + '/exactitud_prueba')
            
    
    def Procesamiento(self, proceso):
        """Método Procesamiento

        Se definen los parámetros predeterminados de la
        interfaz.

        Parameters
        ----------
        proceso: STR, el tipo de proceso a realizar, puede ser
            'entrenar' o 'cargar'

        Returns
        -------
        """
        # Variable para revisar el progreso del entrenamiento
        # Ya se cargaron las librerías
        self.progreso = {'EMG': 0, 'EEG': 0, 'General': 0}

        # Para el caso del entrenamiento
        if proceso == "entrenar":
            # Crear los directorios donde guardar los datos
            self.direccion, self.ubi = f.Directorios(self.sujeto)
            
            # los registros que se usan para entrenamiento, prueba, etc.
            self.DeterminarRegistros()
            
            # rescatar los canales con los cuales entrenar
            directo = 'Parametros/Sujeto_' + str(self.sujeto) + '/Canales/'
            for tipo in ['EMG', 'EEG']:
                # rendimiento = f.AbrirPkl(directo + 'rendimiento_' + tipo + '.pkl')
                # self.canales[tipo] = f.ElegirCanales(
                #     rendimiento, directo, tipo, num_canales = self.num_ci[tipo])
                # # self.canales[tipo] = f.SeleccionarCanales(
                # #     tipo, directo, num_canales=self.num_ci[tipo]) 
                # self.num_canales[tipo] = len(self.canales[tipo])
                # self.num_ci[tipo] = self.num_canales[tipo]
                
                caracteristicas_todas = True # lo puedo volver un parametro
                canales_todos = True
                if not canales_todos:
                    self.parcial[tipo] = f.AbrirPkl(directo + "resultados_" + tipo +".pkl")
                if not caracteristicas_todas:
                    # las caracteristicas (resultados PSO)
                    
                    self.caracteristicascanal[tipo] = f.SeleccionarCaracteristicas(self.parcial[tipo])
                else:
                    self.caracteristicascanal[tipo] = dict()
                    for canal in self.canales[tipo]:
                        self.caracteristicascanal[tipo][canal] = principal.caracteristicas[tipo]
                        
                        
                # nueva selección 
                # self.canales[tipo] = list(self.caracteristicas[tipo].keys())
                # self.num_canales[tipo] = len(self.canales[tipo])
                # self.num_ci[tipo] = self.num_canales[tipo]
                
            # Me parece que es necesario modificar aquí de forma que se pueda
            # cargar los canales y caracteristicas elegidas anteriormente
            
            # Guardar la configuración del modelo
            # self.GuardarParametros()
            
            # Entrenamiento de clasificadores en dos hilos
            # No se encontró mejoría al entrenarlos en dos hilos
            # hilo_entrenamiento_EMG = threading.Thread(
                # target = self.Entrenamiento, args = ('EMG',))
            # hilo_entrenamiento_EEG = threading.Thread(
                # target = self.Entrenamiento, args = ('EEG',))
            # Empieza la ejecución de ambos hilos
            # hilo_entrenamiento_EMG.start()
            # hilo_entrenamiento_EEG.start()
            # # Espera que terminen la ejecución de ambos hilos
            # hilo_entrenamiento_EMG.join()
            # hilo_entrenamiento_EEG.join()
            
            # # realiza la combinación de los clasificadores entrenados
            # # ejecutar dos procesos de forma secuencial no funciona
            # # entrenamiento = process(target = self.Entrenamiento('EMG'))
            # # entrenamiento.start()
            # # entrenamiento.join()
            # # entrenamiento = process(target = self.Entrenamiento('EEG'))
            # # entrenamiento.start()
            # # entrenamiento.join()
            
            # # Ejecutado de forma secuencial
            # self.Entrenamiento('EMG')
            # self.Entrenamiento('EEG')
            
            # if self.balancear:
            #     self.CombinacionCargada(crear_directorio=False)
            # else:
            #     self.Combinacion()
            self.Entrenar()

        # Para el caso de cargar los datos
        elif proceso == "cargar":
            # Determina el de mejor rendimiento
            self.direccion, self.ubi, existe = f.DeterminarDirectorio(
                self.sujeto, 'Combinada')
            # se comprueba que existen datos a cargar
            
            # existe = True
            if existe:
                # la nueva carga
                self.direccion, self.ubi, existe_emg = f.DeterminarDirectorio(
                    self.sujeto, 'EMG', tam_ventana=self.tam_ventana_ms)
                if existe_emg:
                    self.CargarParametros(tipo='EMG')
                    self.CargarDatos('EMG')

                self.direccion, self.ubi, existe_eeg = f.DeterminarDirectorio(
                    self.sujeto, 'EEG', tam_ventana=self.tam_ventana_ms)
                if existe_eeg:
                    self.CargarParametros(tipo='EEG')
                    self.CargarDatos('EEG')

                # hilo_cargar_combinacion.start()
                # hilo_cargar_combinacion.join()
                if existe_eeg and existe_emg:
                    # determinar los registros de prueba
                    # sacar los datos de dataset
                    # se toma las de EMG ya que son más pequeñas
                    # datos = f.ExtraerDatos(self.directorio, self.sujeto, 'EMG')
                    # Determinar de los registros
                    self.DeterminarRegistros()
                    # Se combinan los clasificadores
                    self.CombinacionCargada()

            # if existe:
            #     # Cargar los parametros del sistema
            #     self.CargarParametros()
            #     # Entrenamiento de clasificadores en dos hilos
            #     hilo_cargar_emg = threading.Thread(
            #         target=self.CargarDatos, args=('EMG',))
            #     hilo_cargar_eeg = threading.Thread(
            #         target=self.CargarDatos, args=('EEG',))
            #     # hilo_cargar_combinacion = threading.Thread(
            #     #     target=self.Combinacion)
            #     # hilo_cargar_combinacion = threading.Thread(
            #     #     target = self.CargarCombinacion)

            #     # Empieza la ejecución de ambos hilos
            #     hilo_cargar_emg.start()
            #     hilo_cargar_eeg.start()
            #     # Espera que terminen la ejecución de ambos hilos
            #     hilo_cargar_emg.join()
            #     hilo_cargar_eeg.join()
            #     # hilo_cargar_combinacion.start()
            #     # hilo_cargar_combinacion.join()
            #     self.CombinacionCargada()

            #     # Así se cargan los mejores clasificadores pero no
            #     # no calcula ninguna metrica ya que no hay datos para
            #     # prueba
            #     # realiza la combinación de los clasificadores entrenados
            #     # self.direccion, self.ubi, existe = f.DeterminarDirectorio(
            #     #     self.sujeto, 'EMG')
            #     # print(self.ubi)
            #     # self.CargarDatos('EMG')
            #     # self.direccion, self.ubi, existe = f.DeterminarDirectorio(
            #     #     self.sujeto, 'EEG')
            #     # print(self.ubi)
            #     # self.CargarDatos('EEG')
            #     # self.direccion, self.ubi = f.Directorios(self.sujeto)
            #     # self.CargarCombinacion()
    
        elif proceso == 'canales':
            # rescatar los canales con los cuales entrenar
            # directo = 'Parametros/Sujeto_' + str(self.sujeto) + '/Canales/'
            # for tipo in ['EMG', 'EEG']:
            #     # rendimiento = f.AbrirPkl(directo + 'rendimiento_' + tipo + '.pkl')
            #     # self.canales[tipo] = f.ElegirCanales(
            #     #     rendimiento, directo, tipo, num_canales = self.num_ci[tipo])
            #     self.canales[tipo] = f.SeleccionarCanales(
            #         tipo, directo, num_canales=self.num_ci[tipo])
            #     self.num_canales[tipo] = self.num_ci[tipo]
            # Determinar de los registros
            self.DeterminarRegistros()
            # self.DeterminarCanales('EMG')
            # self.DeterminarCanales('EEG')
            for tipo in ['EEG', 'EMG']:
                # directo = 'Parametros/Sujeto_' + str(self.sujeto) + '/Canales/'
                # rendimiento = f.AbrirPkl(directo + 'rendimiento_' + tipo + '.pkl')
                # self.canales[tipo] = f.ElegirCanales(
                #     rendimiento, directo, tipo, num_canales = self.num_ci[tipo])
                # self.num_canales[tipo] = len(self.canales[tipo])
                
                self.Seleccion(tipo, sel_canales=True, sel_cara=False)
                # self.Seleccion(tipo, sel_canales=False, sel_cara=True) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                
        
            
                
        # Actualiza la variable para hacer seguimiento al progreso
        self.ActualizarProgreso('General', 1.00)


# mujeres: 
# [2, 6, 8, 9, 11, 13, 18, 19, 20, 23]
# Elegidas aleatoreamente: [23,  8,  2]
    
# hombres:
# [1, 3, 4, 5, 7, 10, 12, 14, 15, 16 ,17, 21, 22, 24, 25]
# Elegidos aleatoreamente: [22, 21, 15,  7]

sujeto = [9, 14]
sujetos = [2, 7, 8, 15, 21, 22, 23]

solo_sujeto = True
multi_sujeto = False
sel_canal_cara = True # proceso de selección de canales y caracterisitcas
sel_canal = False # ya se realizó una selección de canales y caracteristicas
prepro = True

if solo_sujeto:
    # sujetos = sujetos + sujeto
    for suj in sujetos:
        principal = Modelo()
        principal.ObtenerParametros(suj)
        if sel_canal_cara:
            principal.Procesamiento('canales')
        if sel_canal: 
            directo = 'Parametros/Sujeto_' + str(suj) + '/Canales/'
            for tipo in ['EMG', 'EEG']:
                rendimiento = f.AbrirPkl(
                    directo + 'rendimiento_' + tipo + '.pkl')
                principal.canales[tipo] = f.ElegirCanales(
                    rendimiento, directo, tipo, 
                    num_canales = principal.num_ci[tipo])
                principal.num_canales[tipo] = len(principal.canales[tipo])
                principal.num_ci[tipo] = principal.num_canales[tipo] # creo que esta linea es redundante
                principal.nombres[tipo] = []
                for canal in principal.canales[tipo]:
                    principal.nombres[tipo].append(
                        f.NombreCanal(canal, invertir=True))
                del rendimiento
        
        for i in range(5):
            principal.Procesamiento('entrenar')
        del principal

if multi_sujeto:
    ws = Modelo()
    ws.ObtenerParametros(sujetos)
    
    if prepro:
        print('Inico preprocesamiento')
        for sub in sujetos+sujeto:
            ws.Preprocesamiento('EMG', sub)
            ws.Preprocesamiento('EEG', sub, guardar_clases=False)
        print('Final preprocesamiento')

    # abrir los canales seleccionados
    ws.direccion, ws.ubi = f.Directorios(sujetos)
        
    if sel_canal_cara:
        print('Inicio selección de canales y caracteristicas')    
        for tipo in ['EMG', 'EEG']:
            ws.SeleccionCanales(sujetos, tipo, ws.nombres[tipo])
            ws.SeleccionCaracteristicas(sujetos, tipo)
        print('Final de selección de canales y caracteristicas')
    else: 
        # Para cargar los canales seleccionados
        directo = 'Parametros/Sujeto_' + str(ws.sujeto) + '/Canales/'
        for tipo in ['EEG', 'EMG']:
            rendimiento = f.AbrirPkl(directo + 'rendimiento_' + tipo + '.pkl')
            ws.canales[tipo] = f.ElegirCanales(
                rendimiento, directo, tipo, num_canales = ws.num_ci[tipo])
            ws.num_canales[tipo] = len(ws.canales[tipo])
            # ws.SeleccionCaracteristicas(sujetos, tipo)
            del rendimiento
    
    print('Inicio extracción de caracteristicas seleccionadas') 
    for tipo in ['EEG', 'EMG']:
        ws.ExtraccionCaracteristicas(tipo, sujetos, entrenar=True)
        ws.ExtraccionCaracteristicas(tipo, sujeto, entrenar=False)
    print('Final Inicio extracción de caracteristicas seleccionadas')
    
    print('Inicio de clasiicación')
    ws.Clasificacion(sujetos)
    ws.Clasificacion(sujetos, entrenar=False)
    print('Final de clasificación')
    
    print('Final del proceso')
    

# else:
#     if sel_canal_cara:
#         ws.Procesamiento('canales')
#     else:
#         #para revisar el rendimiento de lo optenido en la seleccion de canales
#         rendimiento = dict()
#         for sujeto in sujetos:
#             directo = 'D:/Proyectos/ICCH/Parametros/Sujeto_' + str(sujeto) + '/Canales/'
#             for tipo in ['EMG', 'EEG']:
#                 rendimiento[str(sujeto) + "_" + tipo] = f.AbrirPkl(directo + 'resultados_canales_' + tipo + '.pkl')


# for i in range(5):
#     for sujeto in lista:
#         principal = Modelo()
#         principal.ObtenerParametros(sujeto)
#         principal.Procesamiento('entrenar')
#         del principal

# # Entrenar realizar eltrenamiento grande
# lista = [25]
# # Definicíones temporales de los datos
# # cambiar a la hora de integralo en la interfaz
# directorio = 'Dataset'
# # Datos y canales a utilizar
# nombres = dict()
# # 'EMG_ref'
# nombres['EMG'] = [
#     'EMG_1', 'EMG_2', 'EMG_3', 'EMG_4', 'EMG_5', 'EMG_6'
#     ]
# # Sobre corteza motora
# nombres['EEG'] = [
#             'Fz', 'FC3', 'FC1', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
#             'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz',
#             'P2', 'POz']

# nombre_clases = [
#             'Click izq.', 'Click der.', 'Izquierda', 'Derecha', 'Arriba', 'Abajo', 'Reposo'
#         ]

# # Cambio de numero de CI - igual a canales
# for sujeto in lista:
#     principal = Modelo()
#     principal.Parametros(
#             directorio, sujeto, nombres, nombre_clases, f_tipo='butter',
#             b_tipo='bandpass', frec_corte={
#                 'EMG': np.array([8, 520]), 'EEG': np.array([6, 24])},
#             f_orden=5, m={'EMG': 2, 'EEG': 10}, tam_ventana_ms=300, paso_ms=60,
#             descarte_ms = {
#                 'EMG': {'Activo': 300, 'Reposo': 3000},
#                 'EEG': {'Activo': 300, 'Reposo': 3000}}, reclamador_ms={
#                 'EMG': {'Activo': 3400, 'Reposo': 560},
#                 'EEG': {'Activo': 3400, 'Reposo': 560}},
#             porcen_prueba=0.2, porcen_validacion=0.1,
#             calcular_ica={'EMG': False, 'EEG': False},
#             num_ci={'EMG': 6, 'EEG': 21}, determinar_ci=False, epocas=128,
#             lotes=64)
#     principal.Procesamiento('entrenar')
#     del principal

# import winsound
# for i in range(3):
#     winsound.PlaySound("G:/ASUS/Music/Woof.wav", winsound.SND_FILENAME)
