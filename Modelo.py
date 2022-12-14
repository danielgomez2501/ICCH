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
import threading
# dividir la base de datos
from sklearn.model_selection import train_test_split
# Para matrices de confusión
from sklearn.metrics import confusion_matrix
from tensorflow.math import argmax  # para convertir de one hot a un vector
# Mis funciones
import Funciones as f


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
        self.m = {'EMG': 2, 'EEG': 10}
        self.tam_ventana_ms = 300  # en ms
        self.paso_ms = 60  # en ms
        self.descarte_ms = {
            'EMG': {'Activo': 300, 'Reposo': 3000},
            'EEG': {'Activo': 300, 'Reposo': 3000}}
        self.reclamador_ms = {
            'EMG': {'Activo': 3500, 'Reposo': 1000},
            'EEG': {'Activo': 3500, 'Reposo': 1000}}
        self.porcen_prueba = 0.2
        self.porcen_validacion = 0.1
        self.calcular_ica = {'EMG': False, 'EEG': False}
        self.num_ci = {'EMG': 6, 'EEG': 20}
        self.epocas = 1024
        self.lotes = 32
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
        self.num_ventanas = dict.fromkeys(['EMG', 'EEG'])
        # Variables a guardar
        self.filtro = dict.fromkeys(['EMG', 'EEG'])
        self.ica_total = dict.fromkeys(['EMG', 'EEG'])
        self.whiten = dict.fromkeys(['EMG', 'EEG'])
        self.modelo = dict.fromkeys(['EMG', 'EEG'])
        self.confusion = {'EMG': dict.fromkeys(['Validacion', 'Prueba']),
                          'EEG': dict.fromkeys(['Validacion', 'Prueba']),
                          'Combinada': dict.fromkeys(['Validacion', 'Prueba'])}
        self.metricas = dict.fromkeys(['EMG', 'EEG'])
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
        directorio = 'Dataset'
        # Datos y canales a utilizar
        nombres = dict()
        # 'EMG_ref'
        nombres['EMG'] = [
            'EMG_1', 'EMG_2', 'EMG_3', 'EMG_4', 'EMG_5', 'EMG_6'
        ]
        # 10-20
        nombres['EEG'] = [
            'FP1', 'F7', 'F3', 'Fz', 'T7', 'C3', 'Cz', 'P7', 'P3', 'Pz',
            'FP2', 'F4', 'F8', 'C4', 'T8', 'P4', 'P8', 'O1', 'Oz', 'O2'
        ]
        # Sobre corteza motora ¿?
        # nombres['EEG'] = [
        #     'FC5', 'FC3', 'FC1', 'Fz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 
        #     'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
        #     'Cz'
        #     ]
        nombre_clases = [
            'Click izq.', 'Click der.', 'Izquierda', 'Derecha', 'Arriba', 'Abajo', 'Reposo'
        ]

        self.Parametros(
            directorio, sujeto, nombres, nombre_clases, f_tipo='butter',
            b_tipo='bandpass', frec_corte={
                'EMG': np.array([8, 520]), 'EEG': np.array([6, 24])},
            f_orden=5, m={'EMG': 2, 'EEG': 10}, tam_ventana_ms=300, paso_ms=60,
            descarte_ms={
                'EMG': {'Activo': 300, 'Reposo': 3000},
                'EEG': {'Activo': 300, 'Reposo': 3000}}, reclamador_ms={
                'EMG': {'Activo': 3400, 'Reposo': 600},
                'EEG': {'Activo': 3400, 'Reposo': 600}},
            porcen_prueba=0.2, porcen_validacion=0.1,
            calcular_ica={'EMG': False, 'EEG': False},
            num_ci={'EMG': 4, 'EEG': 16}, determinar_ci=False, epocas=2,
            lotes=64)

    def Parametros(
            self, directorio, sujeto, nombres, nombre_clases, f_tipo='butter',
            b_tipo='bandpass', frec_corte=None, f_orden=5, m=None,
            tam_ventana_ms=300, paso_ms=60, descarte_ms=None, reclamador_ms=None,
            porcen_prueba=0.2, porcen_validacion=0.1, calcular_ica=None,
            num_ci=None, determinar_ci=False, epocas=1024, lotes=32):
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
        descarte_ms: DICT,  indica el tiempo a saltar para luego
            empezar a tomar los datos después de una bandera, en ms:
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
            en la cual tomar ventanas despues de una bandera, en ms:
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
                'EMG': {'Activo': 300, 'Reposo': 3000},
                'EEG': {'Activo': 300, 'Reposo': 3000}}
        if reclamador_ms is None:
            reclamador_ms = {
                'EMG': {'Activo': 3500, 'Reposo': 1000},
                'EEG': {'Activo': 3500, 'Reposo': 1000}}
        if calcular_ica is None:
            calcular_ica = {'EMG': False, 'EEG': False}
        if num_ci is None:
            num_ci = {'EMG': 4, 'EEG': 4}

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
        self.calcular_ica = calcular_ica
        self.num_ci = num_ci
        self.epocas = epocas
        self.lotes = lotes

        # se calculan de acuerdo a los parámetros dados
        # calcular el número de clases
        self.num_clases = len(nombre_clases)
        # traduce los nombres de canales del estándar 10-10 a los del dataset
        self.canales['EMG'] = f.TraduciorNombresCanales(nombres['EMG'])
        self.canales['EEG'] = f.TraduciorNombresCanales(nombres['EEG'])
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
        if self.num_ci['EEG'] < 4:
            self.num_ci['EEG'] = 4

    def ParametrosTipo(
            self, tipo, directorio, sujeto, nombres, nombre_clases, f_tipo='butter',
            b_tipo='bandpass', frec_corte=None, f_orden=5, m=None,
            tam_ventana_ms=300, paso_ms=60, descarte_ms=None, reclamador_ms=None,
            porcen_prueba=0.2, porcen_validacion=0.1, calcular_ica=None,
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
            if calcular_ica is None:
                calcular_ica['EEG'] = False
            if num_ci is None:
                num_ci['EEG'] = 4

        elif tipo == 'EMG':
            if frec_corte is None:
                frec_corte['EMG'] = np.array([8, 520])
            if m is None:
                m['EMG'] = 2
            if descarte_ms is None:
                descarte_ms['EMG'] = {'Activo': 300, 'Reposo': 3000}
            if reclamador_ms is None:
                reclamador_ms['EMG'] = {'Activo': 3500, 'Reposo': 1000}
            if calcular_ica is None:
                calcular_ica['EMG'] = False
            if num_ci is None:
                num_ci['EMG'] = 4

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
        self.canales[tipo] = f.TraduciorNombresCanales(nombres[tipo])
        self.num_canales[tipo] = len(self.canales[tipo])
        # para los componentes independientes
        if determinar_ci:
            # Cantidad de componentes independientes a calcular
            # El número de CI corresponde a la mitad de los canales usados
            self.num_ci[tipo] = int(self.num_canales[tipo] / 2)
        # Para asegurar que haya por lo menos 4 ci, ya que de lo contrario no
        # se puede aplicar las maxpool de la CNN.
        if self.num_ci[tipo] < 4:
            self.num_ci[tipo] = 4

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
                self.configuracion['porcentaje validacion'],
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
                self.configuracion['porcentaje validacion'],
                self.configuracion['calcular ica'], self.configuracion['numero ci'],
                False, self.configuracion['epocas'], self.configuracion['lotes'])

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
        datos = f.ExtraerDatos(self.directorio, self.sujeto, tipo)

        # Actualiza la variable para hacer seguimiento al progreso
        print('Se extrae la información de la base de datos para ' + tipo)
        self.ActualizarProgreso(tipo, 0.15)
        # -----------------------------------------------------------------------------
        # Filtro
        self.filtro[tipo] = f.DisenarFiltro(
            self.f_tipo, self.b_tipo, self.f_orden, self.frec_corte[tipo],
            datos['Frecuencia muestreo'])

        # Actualiza la variable para hacer seguimiento al progreso
        print('Se diseña el filtro para ' + tipo)
        self.ActualizarProgreso(tipo, 0.21)
        # -----------------------------------------------------------------------------
        # Función para sub muestreo
        # inicializar las listas
        senales_subm, clases = f.Submuestreo(
            self.directorio, tipo, datos, self.sujeto, 1,
            self.canales[tipo], self.nombre_clases, self.filtro[tipo],
            self.m[tipo])
        senales = [senales_subm]
        clases_OH = [clases]
        for sesion in range(2, 4):
            senales_subm, clases = f.Submuestreo(
                self.directorio, tipo, datos, self.sujeto, sesion,
                self.canales[tipo], self.nombre_clases, self.filtro[tipo],
                self.m[tipo])
            senales.append(senales_subm)
            clases_OH.append(clases)

        del senales_subm, clases

        # Calcular a partir de frecuencias de sub muestreo
        self.frec_submuestreo[tipo] = int(
            datos['Frecuencia muestreo'] / self.m[tipo])

        self.tam_ventana[tipo] = int(
            self.tam_ventana_ms * 0.001 * self.frec_submuestreo[tipo])
        self.paso_ventana[tipo] = int(
            self.paso_ms * 0.001 * self.frec_submuestreo[tipo])

        # Actualiza la variable para hacer seguimiento al progreso
        print('Se aplica el filtro y se realiza el submuestreo para ' + tipo)
        self.ActualizarProgreso(tipo, 0.44)
        # -----------------------------------------------------------------------------
        # Enventanado

        # Descarte de datos ambiguos
        # Dado al gran numero de ventanas, se calculan las ventanas por
        # cada sesión, luego se descartan las que no se usan, para
        # finalmente, concatenarlas.

        # Valores para descarte:
        # traducción de tiempos de descarte y reclamador a número de muestras
        descarte = dict.fromkeys(['Activo', 'Reposo'])
        descarte['Activo'] = int(
            self.descarte_ms[tipo]['Activo'] * self.frec_submuestreo[tipo] / (self.paso_ventana[tipo] * 1000))
        descarte['Reposo'] = int(
            self.descarte_ms[tipo]['Reposo'] * self.frec_submuestreo[tipo] / (self.paso_ventana[tipo] * 1000))
        reclamador = dict.fromkeys(['Activo', 'Reposo'])
        reclamador['Activo'] = int(
            self.reclamador_ms[tipo]['Activo'] * self.frec_submuestreo[tipo] / (self.paso_ventana[tipo] * 1000))
        reclamador['Reposo'] = int(
            self.reclamador_ms[tipo]['Reposo'] * self.frec_submuestreo[tipo] / (self.paso_ventana[tipo] * 1000))
        # se determina la clase de reposo
        clase_reposo = np.asarray(clases_OH[0].iloc[0], dtype='int8')

        # Se disponen las sesiones en una lista, y se inicializa
        vent, clas = f.Enventanado(
            senales[0], clases_OH[0], datos, 0, self.tam_ventana_ms,
            self.paso_ms, self.frec_submuestreo[tipo], self.num_canales[tipo],
            self.num_clases)
        vent, clas = f.DescartarVentanas(
            vent, clas, clase_reposo, datos['Banderas'], reclamador,
            descarte)
        clase = [clas]
        del clas
        venta = [vent]
        del vent
        # ciclo para las demás secciones
        for sesion in range(1, 3):
            vent, clas = f.Enventanado(
                senales[sesion], clases_OH[sesion], datos, sesion,
                self.tam_ventana_ms, self.paso_ms, self.frec_submuestreo[tipo],
                self.num_canales[tipo], self.num_clases)
            vent, clas = f.DescartarVentanas(
                vent, clas, clase_reposo, datos['Banderas'], reclamador,
                descarte)
            clase.append(clas)
            del clas
            venta.append(vent)
            del vent
        del clases_OH
        # Las sesiones se disponen en una única matriz de datos
        clases = np.vstack((clase[0], clase[1], clase[2]))
        del clase
        ventanas = np.vstack((venta[0], venta[1], venta[2]))
        del venta

        # Actualiza la variable para hacer seguimiento al progreso
        print('Se realiza el enventanado para ' + tipo)
        self.ActualizarProgreso(tipo, 0.50)
        # -----------------------------------------------------------------------------
        # División y balanceo del dataset

        # Dividir datos de entrenamiento, test y validación; Además se realiza 
        # el balance de base de datos médiate sub muestreo aleatorio
        # División y balanceo de dataset datos EMG
        train, class_train, validation, class_validation, test, self.class_test = f.Division(
            ventanas, clases, self.porcen_prueba, self.porcen_validacion,
            self.num_clases, todasclases=True)
        del ventanas, clases

        # para revisar la cantidad de ventanas disponibles
        self.num_ventanas[tipo] = dict.fromkeys(['Entrenamiento', 'Validacion', 'Prueba'])
        self.num_ventanas[tipo]['Entrenamiento'] = len(train)
        self.num_ventanas[tipo]['Validacion'] = len(validation)
        self.num_ventanas[tipo]['Prueba'] = len(test)

        # Actualiza la variable para hacer seguimiento al progreso
        print('Se divide y balancea el dataset para ' + tipo)
        self.ActualizarProgreso(tipo, 0.55)
        # -----------------------------------------------------------------------------
        # Extracción de características
        # Cálculo de FastICA
        # Variables a calcular para poder calcular el ICA
        if self.calcular_ica[tipo]:
            # if tipo == 'Isa':
            senales = np.concatenate(senales[:], axis=1)
            # El ICA en donde se calcula la matriz de whitening y luego
            # se calcula el ICA independiente para cada ventana
            train, validation, test, self.ica_total[tipo], self.whiten[tipo] = f.VICA(
                train, validation, test, senales, self.num_ci[tipo],
                self.tam_ventana[tipo], self.paso_ventana[tipo])
            # train, validation, test, self.ica_total[tipo], self.whiten[tipo] = f.FICA(
            #     train, validation, test, self.num_ci[tipo], self.tam_ventana[tipo], 
            #     self.paso_ventana[tipo])
            del senales
            print ('Se calculan los CI para ' + tipo)
        else:
            self.num_ci[tipo] = self.num_canales[tipo]

        # Actualiza la variable para hacer seguimiento al progreso
        self.ActualizarProgreso(tipo, 0.77)
        # -----------------------------------------------------------------------------
        # Clasificador
        # diseñar, entrenar y revisar el rendimiento de los clasificadores
        self.modelo[tipo], cnn, self.metricas[tipo], self.confusion[tipo], self.prediccion[tipo] = f.Clasificador(
            train, class_train, validation, class_validation,
            test, self.class_test, self.direccion, tipo, self.num_ci[tipo],
            self.tam_ventana[tipo], self.nombre_clases, self.num_clases,
            self.epocas, self.lotes)

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
        info.update(presicion_clases)
        info.update(self.metricas[tipo])
        f.GuardarMetricas(info)
        # Diccionario donde se guarda la configuración de la interfaz
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

        self.confusion['Combinada'] = confusion_matrix(
            argmax(self.class_test, axis=1),
            argmax(self.prediccion['Combinada'], axis=1))

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

            # inicializar las listas
            senales_subm, clases = f.Submuestreo(
                self.directorio, tipo, datos, self.sujeto, 1,
                self.canales[tipo], self.nombre_clases, self.filtro[tipo],
                self.m[tipo])
            senales = [senales_subm]
            clases_OH = [clases]
            for sesion in range(2, 4):
                senales_subm, clases = f.Submuestreo(
                    self.directorio, tipo, datos, self.sujeto, sesion,
                    self.canales[tipo], self.nombre_clases, self.filtro[tipo],
                    self.m[tipo])
                senales.append(senales_subm)
                clases_OH.append(clases)

            del senales_subm, clases
            # Enventanado

            # Descarte de datos ambiguos
            # Dado al gran numero de ventanas, se calculan las ventanas por
            # cada sesión, luego se descartan las que no se usan, para
            # finalmente, concatenarlas.

            # Valores para descarte:
            # traducción de tiempos de descarte y reclamador a número de muestras
            descarte = dict.fromkeys(['Activo', 'Reposo'])
            descarte['Activo'] = int(
                self.descarte_ms[tipo]['Activo'] * self.frec_submuestreo[tipo] / (self.paso_ventana[tipo] * 1000))
            descarte['Reposo'] = int(
                self.descarte_ms[tipo]['Reposo'] * self.frec_submuestreo[tipo] / (self.paso_ventana[tipo] * 1000))
            reclamador = dict.fromkeys(['Activo', 'Reposo'])
            reclamador['Activo'] = int(
                self.reclamador_ms[tipo]['Activo'] * self.frec_submuestreo[tipo] / (self.paso_ventana[tipo] * 1000))
            reclamador['Reposo'] = int(
                self.reclamador_ms[tipo]['Reposo'] * self.frec_submuestreo[tipo] / (self.paso_ventana[tipo] * 1000))
            # se determina la clase de reposo
            clase_reposo = np.asarray(clases_OH[0].iloc[0], dtype='int8')

            # Se disponen las sesiones en una lista, y se inicializa
            vent, clas = f.Enventanado(
                senales[0], clases_OH[0], datos, 0, self.tam_ventana_ms,
                self.paso_ms, self.frec_submuestreo[tipo], self.num_canales[tipo],
                self.num_clases)
            vent, clas = f.DescartarVentanas(
                vent, clas, clase_reposo, datos['Banderas'], reclamador,
                descarte)
            clase = [clas]
            del clas
            venta = [vent]
            del vent
            # ciclo para las demás sesiones
            for sesion in range(1, 3):
                vent, clas = f.Enventanado(
                    senales[sesion], clases_OH[sesion], datos, sesion,
                    self.tam_ventana_ms, self.paso_ms, self.frec_submuestreo[tipo],
                    self.num_canales[tipo], self.num_clases)
                vent, clas = f.DescartarVentanas(
                    vent, clas, clase_reposo, datos['Banderas'], reclamador,
                    descarte)
                clase.append(clas)
                del clas
                venta.append(vent)
                del vent
            del clases_OH
            # Las sesiones se disponen en una única matriz de datos
            clases = np.vstack((clase[0], clase[1], clase[2]))
            del clase
            ventanas = np.vstack((venta[0], venta[1], venta[2]))
            del venta

            # División prueba
            _, test[tipo], _, class_test_un = train_test_split(
                ventanas, clases, test_size=self.porcen_prueba, shuffle=False)
            del ventanas, clases

        print('Determinando ventanas de prueba')
        # Balanceo doble aplicado a todas las clases
        tipos_clases = np.identity(self.num_clases, dtype='int8')
        for clase in tipos_clases:
            test['EEG'], test['EMG'], class_test_un = f.BalanceDoble(
                test['EEG'], test['EMG'], class_test_un, clase)
        self.class_test = class_test_un
        del class_test_un

        for tipo in ['EEG', 'EMG']:
            # Aplicación de ICA
            if self.calcular_ica[tipo]:
                print('Calculando CI para ' + tipo)
                test[tipo] = f.AplicarICA(
                    len(test[tipo]), self.num_ci[tipo], self.tam_ventana[tipo],
                    self.ica_total[tipo], test[tipo])

            print('Realizando la clasificación de ventas de prueba')
            # Determinar predicción
            self.prediccion[tipo] = self.modelo[tipo].predict(test[tipo])

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
        if tipo == 'EMG':
            self.modelo[tipo] = f.ClasificadorEMG(
                self.num_ci[tipo], self.tam_ventana[tipo], self.num_clases)
        elif tipo == 'EEG':
            self.modelo[tipo] = f.ClasificadorEEG(
                self.num_ci[tipo], self.tam_ventana[tipo], self.num_clases)

        # Loads the weights
        self.modelo[tipo].load_weights(
            self.direccion + '/Clasificador/' + tipo + "/" + tipo + "_cp.ckpt")

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
            # Guardar la configuración del modelo
            self.GuardarParametros()
            # Entrenamiento de clasificadores en dos hilos
            # No se encontró mejoría al entrenarlos en dos hilos
            # hilo_entrenamiento_EMG = threading.Thread(
            #     target = self.Entrenamiento, args = ('EMG',))
            # hilo_entrenamiento_EEG = threading.Thread(
            #     target = self.Entrenamiento, args = ('EEG',))
            # # Empieza la ejecución de ambos hilos
            # hilo_entrenamiento_EMG.start()
            # hilo_entrenamiento_EEG.start()
            # # Espera que terminen la ejecución de ambos hilos
            # hilo_entrenamiento_EMG.join()
            # hilo_entrenamiento_EEG.join()
            # realiza la combinación de los clasificadores entrenados
            self.Entrenamiento('EMG')
            self.Entrenamiento('EEG')
            self.CombinacionCargada(crear_directorio=False)

        # Para el caso de cargar los datos
        elif proceso == "cargar":
            # Determina el de mejor rendimiento
            self.direccion, self.ubi, existe = f.DeterminarDirectorio(
                self.sujeto, 'Combinada')
            # se comprueba que existen datos a cargar
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

        # Actualiza la variable para hacer seguimiento al progreso
        self.ActualizarProgreso('General', 1.00)


# principal = Modelo()
lista = [2, 7, 11, 13, 21, 25]
sujeto = 2
principal = Modelo()
principal.ObtenerParametros(sujeto)
principal.Procesamiento('cargar')

# principal.ObtenerParametros(sujeto)
# principal.Procesamiento('entrenar')
# Entrenar realizar eltrenamiento grande
# for sujeto in lista:
#     principal = Modelo()
#     principal.ObtenerParametros(sujeto)
#     principal.Procesamiento('entrenar')
#     del principal

"""Para obtener los parametros de la interfaz a entrenar
"""
"""# Definicíones temporales de los datos
# cambiar a la hora de integralo en la interfaz
directorio = 'Dataset'
# Datos y canales a utilizar
nombres = dict()
# 'EMG_ref'
nombres['EMG'] = [
   'EMG_1', 'EMG_2', 'EMG_3', 'EMG_4', 'EMG_5', 'EMG_6'
   ]
# 10-20
nombres['EEG'] = [
        'FP1', 'F7', 'F3', 'Fz', 'T7', 'C3', 'Cz', 'P7', 'P3', 'Pz',
        'FP2', 'F4', 'F8', 'C4', 'T8', 'P4', 'P8', 'O1', 'Oz', 'O2'
    ]
        # Sobre corteza motora ¿?
        # nombres['EEG'] = [
        #     'FC5', 'FC3', 'FC1', 'Fz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1',
        #     'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
        #     'Cz'
        #     ]
nombre_clases = [
            'Click izq.', 'Click der.', 'Izquierda', 'Derecha', 'Arriba', 'Abajo', 'Reposo'
        ]

for i in range(3):
    for sujeto in lista:
        principal.Parametros(
            directorio, sujeto, nombres, nombre_clases, f_tipo='butter',
            b_tipo='bandpass', frec_corte={
                'EMG': np.array([8, 520]), 'EEG': np.array([6, 24])},
            f_orden=5, m={'EMG': 2, 'EEG': 10}, tam_ventana_ms=300, paso_ms=60,
            descarte_ms={
                'EMG': {'Activo': 300, 'Reposo': 3000},
                'EEG': {'Activo': 300, 'Reposo': 3000}}, reclamador_ms={
                'EMG': {'Activo': 3500, 'Reposo': 1000},
                'EEG': {'Activo': 3500, 'Reposo': 1000}},
            porcen_prueba=0.2, porcen_validacion=0.1,
            calcular_ica={'EMG': False, 'EEG': False},
            num_ci={'EMG': 4, 'EEG': 16}, determinar_ci=False, epocas=1024,
            lotes=32)
        principal.Procesamiento('entrenar')
"""
