# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 12:54:34 2022

@author: Daniel
"""

'''
#-----------------------------------------------------------------------------
# Por hacer
#-----------------------------------------------------------------------------
Pruebas de mejora de la interfaz:
    - Version 0.70
        tal cual planteado en el documento
    - Versioón 0.71
        se quita ICA de señales de EMG

Revisar si necesito utilizar las señales para cuando se cargan los 
datos.

Revisar que los datos extraidos en la variable datos sean iguales para
EMG y EEG

Cambiar el los electrodos de EEG para que se posicionen sobre la 
corteza motora.

Revisar si toca quitar los outlayers para el entrenamiento de las
señales

Probar si quitando ICA en datos de EMG mejora la clasificación.

Reducir el numero de parametros a entrenar por cada red neuronal de 
forma que el numero de parametros a entrenar sea unas 5 ó 10 veces 
menor que el numero de ventanas de entrenamiento    
    El numero de ventanas:
        Ventanas totales: 12,827
        Ventanas de entrenamiento: 10,261
        Ventanas de pruba: 2,566

Optimizar algoritmo para balanceo de datos, para todas las clases.

#-----------------------------------------------------------------------------
# Notas de verciones
#-----------------------------------------------------------------------------

0.10:   Primera version (solo realizaba hasta la extracción de 
        caracteristicas).

0.20:   Se añadió la clasificación.

0.30:   Se crearon funciones para diversas tareas repetitivas o que se 
        puedan modificar ciertos parametros, sin tener que volver a 
        escribir codigo.

0.33:   Se re escribe el codigo de acuerdo a la guia de estilo para 
        codigo de python PEP 8, pero se omite la recomendación sobre el 
        uso casi obligatorio del idioma ingles, dado que consumiria 
        demaciado tiempo traduccir todas las variables y comentarios.
        
0.40:   Se implementan los Puntos de control en el entrenamiento del 
        clasificador, a demás de que se reducen el numero de parametros
        entrenables de los clasificadores, y se modifica el llamado a 
        la función encargada de la creación de la extructura del 
        clasificador.
        
0.44:   Se corrigen lineas para la impreción de graficas de las 
        señales, se imprime el rendimiento del calsificador en terminos 
        de matrices de confunción, se combinan la salida de los dos 
        clasificadores.
        

0.50:   Se realiza balance de la base de datos mediante el submuestreo 
        de esta se actualiza la combinan la salida de los dos 
        clasificadores. se dividen los datos en entrenamiento, 
        validación y prueba.
    
0.51:   Se empieza a realizar guardado de la transformación FastICA 
        para un calculo individal para cada una de las ventanas. Se 
        agregan los datos de validación de forma separada.

0.52:   Se busca re calcular el FastICA para que se lo pueda guardar,
        Se corrige el balance de datos y se actualizan las graficas.
        
0.53:   Se corrige el balanceo de datos para que correspondan los datos
        de EEG y EMG de forma que se pudan usar en la combinación.
        
0.54:   Se reducen ciertas lineas de codigo, y se cambian las funciones 
        que dan la estructura a las redes neuronales, haciendo que 
        reciban el parametro de numero de clases para la clasificación.

0.60:   Se ajusta a funciones para integrarlo con la interfaz grafica 
        de usuario, se cambia la forma en la que se cargan y guardan 
        los datos. Se hace seguimiento de la interfaz mediante la 
        variable progreso. Los datos se guardan al final de la 
        ejecución, No realiza carga de datos.

0.61:   Se modifica el llamado a las funciones para el entrenamiento,
        se busca convertilo en función o metodo.

0.62:   Se busca diseñar las funciones para la carga de datos.

0.70:   Se ajusta el codigo para que se ejecute en una clase.

0.71:   Se busca mejorar el rendimiento del clasificador.
        Probabilidad aleatorea en 7 clases: 1/7 = 0.14285714
        Anterior (Solo una sesión): 43%
        El actual: 0.18162239770279973, Id 000, con versión 0.70.
        Se realiza descarte de ventanas que enten en areas de 
        transición de las tareas de movimiento.
'''

#-----------------------------------------------------------------------------
# Librerias
#-----------------------------------------------------------------------------
print('Cargando Librerias ...')

# General
import numpy as np
import threading
# Para matrices de confución
from sklearn.metrics import confusion_matrix
from tensorflow.math import argmax # para convertir de one hot a un vector
# Mis funciones
import Funciones as f

#-----------------------------------------------------------------------------
# Procesamiento
#-----------------------------------------------------------------------------

class Interfaz(object):
    """Clase Interfaz 
    
    Contiene todos los parametros de la HBCI, con los metodos 
    "Entrenamiento" y "Carga" se puede efectuar el entrenamiento o la 
    carga de los parametros de la interfaz.
    """
    
    def __init__(self):
        #super(ClassName, self).__init__()

        # Para la ejecución en hilos
        self.lock = threading.Lock()


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
                self.progreso['EMG'] + self.progreso['EEG'])/2
        self.lock.release()


    def ObtenerParametros(self, sujeto):
        """Para obtener los parametros de la interfaz a entrenar
        """
        # Definicíones temporales de los datos
        # cambiar a la hora de integralo en la interfaz
        directorio = 'D://ASUS/Documents/Tareas/Trabajo de grado/Trabajo de grado/Dataset'
        #sujeto = sujeto
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
            'Forward', 'Backward', 'Left', 'Right', 'Up', 'Down', 'Rest'
            ]

        self.Parametros(
            directorio, sujeto, nombres, nombre_clases, f_tipo = 'butter', 
            b_tipo = 'bandpass', frec_corte = {'EMG':np.array([8, 520]), 
            'EEG':np.array([6, 24])},  f_orden = 5, m = {'EMG': 2, 'EEG': 10}, 
            tam_ventana_ms = 260, paso_ms = 60, descarte_ms = {
            'EMG': {'Activo': 300, 'Reposo': 3000}, 
            'EEG': {'Activo': 300, 'Reposo': 3000}}, reclamador_ms = {
            'EMG': {'Activo': 3500, 'Reposo': 1000},
            'EEG': {'Activo': 3500, 'Reposo': 1000}}, 
            porcen_prueba = 0.2, porcen_validacion = 0.1,
            calcular_ica = {'EMG': False, 'EEG': False},  
            num_ci = {'EMG': 6, 'EEG': 20}, determinar_ci = False, epocas =1024, 
            lotes = 32)


    def Parametros(
        self, directorio, sujeto, nombres, nombre_clases, f_tipo = 'butter', 
        b_tipo = 'bandpass', frec_corte = {'EMG':np.array([8, 520]), 
        'EEG':np.array([6, 24])}, f_orden = 5, m = {'EMG': 2, 'EEG': 25}, 
        tam_ventana_ms = 260, paso_ms = 60, descarte_ms = {
        'EMG': {'Activo': 300, 'Reposo': 3000}, 
        'EEG': {'Activo': 300, 'Reposo': 3000}}, reclamador_ms = {
        'EMG': {'Activo': 3500, 'Reposo': 1000},
        'EEG': {'Activo': 3500, 'Reposo': 1000}}, 
        porcen_prueba = 0.2, porcen_validacion = 0.1, 
        calcular_ica = {'EMG': False, 'EEG': False}, 
        num_ci = {'EMG': 4, 'EEG': 4}, determinar_ci = True, epocas = 1024, 
        lotes = 16):
        """Metodo parametros:
            
        Se definen los parametros predeterminados de la 
        interfaz.
        
        Parameters
        ----------
        directorio: STR, dirección del directorio donde se encuetra la
            base de datos.
        sujeto: INT, Número del sujeto, los disponibles son 2, 7, 11, 13, 21, 25
        nombres: DICT, Nombres de los canales: 
                'EMG': LIST, nombres de acuerdo al dataset.
                'EEG': LIST, nombres de acuerdo al estandar 10-10.
        nombres_clases: LIST, contiene el nombre de las clases
        f_tipo: STR, tipo de filtro, predeterminado 'butter'.
        b_tipo: STR, tipo de banda de paso, predeterminado 'bandpass'.
        frec_corte: DICT, valores de las bandas de paso en Hz:
                'EMG': np.ARRAY, frecuencias, pred: np.array([8, 520]).
                'EEG': np.ARRAY, frecuencias, pred: np.array([6, 24]).
        f_orden: INT, orden del friltro a calcular, pred: 5.
        m: DICT, factor de submuestreo:
                'EMG': INT, factor de submuestreo para emg, pred: 2.
                'EEG': INT. factor de submuestreo para eeg, pred: 25.
        tam_ventana_ms: INT, tamaño de ventana en ms, pred: 260.
        paso_ms: INT, paso entre ventanas en ms, pred: 60.
        descarte_ms: DICT,  indica el el tiempo a saltar para luego 
            empesar a tomar los datos despues de una bandera, en ms:
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
                        pred: 3500.
                'EMG': DICT, duración de franja de tiempo para emg:
                    'Activo': INT, duración para movimiento,
                        pred: 1000.
                    'Reposo': INT, duración para estado de reposo,
                        pred: 1000.
        porcen_prueba: FLOAT, procentaje de datos de prueba, pred: 0.2.
        porcen_validacion: FLOAT, procentaje de datos de validación, 
            pred: 0.2.
        calcular_ica:  DICT, indica se realiza el ICA:
                'EMG': BOOL, indica si aplicar ICA a EMG, pred: False
                'EEG': BOOL, indica si aplicar ICA a EEG, pred: False
        num_ci: DICT, indica los componentes independiente a calcular:
                'EMG': INT, componentes independientes, minimo 4
                'EEG': INT, componentes independientes, minimo 4
        determinar_ci: BOOL, permite el calculo automatico del numero de
            ci, siendo igual a la mitad del numero de canales y mayor
            que 4.
        epocas: INT, cantidad de epocas (epoch) de entrenamiento, 
            pred: 1024
        lotes: INT, cantidad de lotes (batch) de entrenamiento, 
            pred: 16

        Returns
        -------     
        """
        # Parametros generales
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

        # se calculan de acuerdo a los parametros dados
        # calcular el numero de clases
        self.num_clases = len(nombre_clases)
        # traduce los nombres de canales del estandar 10-10 a los del dataset
        self.canales = dict.fromkeys(['EMG', 'EEG'])
        self.canales['EMG'] = f.TraduciorNombresCanales(nombres['EMG'])
        self.canales['EEG'] = f.TraduciorNombresCanales(nombres['EEG'])
        self.num_canales = dict.fromkeys(['EMG', 'EEG'])
        self.num_canales['EMG'] = len(self.canales['EMG'])
        self.num_canales['EEG'] = len(self.canales['EEG'])
        # para los componentes independientes
        if determinar_ci:
            # Cantidad de componentes independientes a calcular
            # El numero de CI corresponde a la mitad de los canales usados
            self.num_ci['EMG'] = int(self.num_canales['EMG'] / 2)
            self.num_ci['EEG'] = int(self.num_canales['EEG'] / 2)
        # Para asegurar que hayan por lo menos 4 ci ya que de lo contrario no
        # se puede aplicar las maxpool de la CNN.
        if self.num_ci['EMG'] < 4:
            self.num_ci['EMG'] = 4
        if self.num_ci['EEG'] < 4:
            self.num_ci['EEG'] = 4


    def Entrenamiento(self, tipo):
        """Metodo Entrenamiento:
            
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
        #-----------------------------------------------------------------------------
        # Extraer datos
        datos = f.ExtraerDatos(self.directorio, self.sujeto, tipo)
        
        # Actualiza la variable para hacer seguimiento al progreso
        self.ActualizarProgreso(tipo, 0.15)

        #-----------------------------------------------------------------------------
        # Filtro
        self.filtro[tipo] = f.DisenarFiltro(
                self.f_tipo, self.b_tipo, self.f_orden, self.frec_corte[tipo], 
                datos['Frecuencia muestreo'])

        # Actualiza la variable para hacer seguimiento al progreso
        self.ActualizarProgreso(tipo, 0.21)

        #-----------------------------------------------------------------------------
        # Diseñar las clases One Hot
        # Tomar la clase de onehot y asignarla a la clases oh de forma que cada
        # indice corresponda con las banderas. 
        # Dataframe para las clases one-hot
        clases_OH = []
        for i in range(3):
            clases_OH.append(f.ClasesOneHot(
                self.nombre_clases, self.num_clases, 
                datos['Final grabacion'][i], datos['Banderas'][i], 
                datos['One Hot'][i]))

        # Actualiza la variable para hacer seguimiento al progreso
        self.ActualizarProgreso(tipo, 0.25)
        
        #-----------------------------------------------------------------------------
        # Funcion para submuestreo
        senales = []
        for sesion in range(1,4):
            senales.append(f.Submuestreo(
                self.directorio, tipo, datos, self.sujeto, sesion, 
                self.canales[tipo], self.nombre_clases, self.filtro[tipo], 
                self.m[tipo]))

        # Calcular a partir de frecuencias de submuestreo
        self.frec_submuestreo[tipo] = int(
                datos['Frecuencia muestreo'] / self.m[tipo])

        self.tam_ventana[tipo] = int(
                self.tam_ventana_ms * 0.001 * self.frec_submuestreo[tipo])
        self.paso_ventana[tipo] = int(
                self.paso_ms * 0.001 * self.frec_submuestreo[tipo])

        # Actualiza la variable para hacer seguimiento al progreso
        self.ActualizarProgreso(tipo, 0.44)

        #-----------------------------------------------------------------------------
        # Enventanado
        # realizar el enventanado, inicialización de los arreglos
        # ventanas = np.empty(self.num_canales[tipo])
        # clases = np.empty(self.num_clases, dtype='int8')
        ventanas, clases = f.Enventanado(
            senales[0], clases_OH[0], datos, 0, self.tam_ventana_ms, 
            self.paso_ms, self.frec_submuestreo[tipo], self.num_canales[tipo], 
            self.num_clases)
        # realizar el enventanado concatenando las ventanas de todas las sesiones
        for sesion in range(1,3):
            vent, clas = f.Enventanado(
                senales[sesion], clases_OH[sesion], datos, sesion, 
                self.tam_ventana_ms, self.paso_ms, self.frec_submuestreo[tipo], 
                self.num_canales[tipo], self.num_clases)
            ventanas = np.concatenate((ventanas,vent), axis=0)
            clases = np.concatenate((clases, clas), axis=0)
        del vent, clas

        # Actualiza la variable para hacer seguimiento al progreso
        self.ActualizarProgreso(tipo, 0.50)

        #-----------------------------------------------------------------------------
        # Divición y balanceo del dataset
        # Descarte de datos ambiguos (Revisar si hacer)
        # vector one-hot con la clase de reposo
        clase_reposo = np.asarray(clases[0], dtype = int)
        
        # traducción de tiempos descarte y reclamador a num. muestras
        descarte = dict.fromkeys(['Activo', 'Reposo'])
        descarte['Activo'] = int(
            self.descarte_ms[tipo]['Activo']*self.frec_submuestreo[tipo]/(self.paso_ventana[tipo]*1000))
        descarte['Reposo'] = int(
            self.descarte_ms[tipo]['Reposo']*self.frec_submuestreo[tipo]/(self.paso_ventana[tipo]*1000))
        reclamador = dict.fromkeys(['Activo', 'Reposo'])
        reclamador['Activo'] = int(
            self.reclamador_ms[tipo]['Activo']*self.frec_submuestreo[tipo]/(self.paso_ventana[tipo]*1000))
        reclamador['Reposo'] = int(
            self.reclamador_ms[tipo]['Reposo']*self.frec_submuestreo[tipo]/(self.paso_ventana[tipo]*1000))

        ventanas, clases = f.DescartarVentanas(
            ventanas, clases, clase_reposo, datos['Banderas'], reclamador, 
            descarte)
        #-----------------------------------------------------------------------------
        # Dividir datos de entrenamiento, test y validación; Además se realiza 
        # el balanceo de base de datos mediate submuestreo aleatoreo
        # Divición y balanceo de dataset datos EMG
        train, class_train, validation, class_validation, test, self.class_test[tipo] = f.Division(
            ventanas, clases, self.porcen_prueba, self.porcen_validacion, 
            self.num_clases, todasclases=True)
        # para revisar la cantidad de ventanas disponibles
        self.num_ventanas[tipo] = dict.fromkeys(['Entrenamiento', 'Validacion', 'Prueba'])
        self.num_ventanas[tipo]['Entrenamiento'] = len(train)
        self.num_ventanas[tipo]['Validacion'] = len(validation)
        self.num_ventanas[tipo]['Prueba'] = len(test)

        # Actualiza la variable para hacer seguimiento al progreso
        self.ActualizarProgreso(tipo, 0.55)

        #-----------------------------------------------------------------------------
        # Extraccion de caracteristicas
        # Calculo de FastICA
        # ICA antiguo, donde se calcula una transformación en CI para todas las señales
        #train, validation, test, self.ica_total[tipo], self.whiten[tipo] = f.FICA(
        #    train, validation, test, self.num_ci[tipo], self.tam_ventana[tipo], 
        #    self.paso_ventana[tipo])
        # variación ICA
        #train, validation, test, self.ica_total[tipo], self.whiten[tipo] = f.NICA(
        #    train, validation, test, self.num_ci[tipo], self.tam_ventana[tipo], 
        #    self.paso_ventana[tipo])
        # ICA nuevo
        # Variables a calcular para poder calcular el ICA
        if self.calcular_ica[tipo]:
        #if tipo == 'Isa':
            senales = np.concatenate(senales[:], axis = 1)
            # El ICA en donde se calcula la matriz de whitening y luego se calcula el ICA independiente para cada ventana
            train, validation, test, self.ica_total[tipo], self.whiten[tipo] = f.VICA(
                train, validation, test, senales, self.num_ci[tipo],
                self.tam_ventana[tipo], self.paso_ventana[tipo])
            # train, validation, test, self.ica_total[tipo], self.whiten[tipo] = f.FICA(
            #     train, validation, test, self.num_ci[tipo], self.tam_ventana[tipo], 
            #     self.paso_ventana[tipo])
            del senales
        else:
            self.num_ci[tipo] = self.num_canales[tipo]

        # Actualiza la variable para hacer seguimiento al progreso
        self.ActualizarProgreso(tipo, 0.77)

        #-----------------------------------------------------------------------------
        # Clasificador
        # diseñar, entrenar y revisar el rendimiento de os clasificadores
        self.modelo[tipo], cnn, self.metricas[tipo], self.confusion[tipo], self.prediccion[tipo] = f.Clasificador(
            train, class_train, validation, class_validation, 
            test, self.class_test[tipo], self.direccion, tipo, self.num_ci[tipo], 
            self.tam_ventana[tipo], self.nombre_clases, self.num_clases, 
            self.epocas, self.lotes)

        # valor de la presisión general del modelo entrenado
        self.exactitud[tipo] = 100 * self.metricas[tipo]['categorical_accuracy']
        print("La exactitud del modelo: {:5.2f}%".format(
                    100 * self.metricas[tipo]['categorical_accuracy']))

        # Función para graficar las matrices de confución y las graficas de
        # entrenamiento
        f.Graficas(
                self.direccion, cnn, self.confusion[tipo], self.nombre_clases, 
                tipo)

        # Actualiza la variable para hacer seguimiento al progreso
        self.ActualizarProgreso(tipo, 0.90)
        # Entrenado

        #-----------------------------------------------------------------------------
        # Guardar datos
        # actualización de dirección
        path = self.direccion + '/Procesamiento/'
        # Guardar filtros
        f.GuardarPkl(self.filtro[tipo], path + 'filtro_' + tipo + '.pkl')
        # Guardar datos de ICA
        if self.calcular_ica[tipo]:
            f.GuardarPkl(self.whiten[tipo], path+'whiten_' + tipo + '.pkl')
            f.GuardarPkl(self.ica_total[tipo], path + 'ica_' + tipo + '.pkl')
        # Guardar datos de matrices de confución
        f.GuardarPkl(
                self.confusion[tipo]['Validacion'], 
                path + 'CM_val_' + tipo + '.pkl')
        # Guardar datos de historial de entrenamiento
        # cnn_emg
        # Guardar metricas de entrenamiento
        f.GuardarPkl(
                self.metricas[tipo], 
                self.direccion + '/General/metricas_' + tipo + '.pkl')
        # Diccionario donde se guardan las metricas de entrenamiento
        info = {
                'Sujeto': self.sujeto, 'Id': self.ubi,
                'Tipo de señales': tipo, 'Exactitud': self.exactitud[tipo]}
        # calcular presicón por clases mediante matriz de confución
        presicion_clases,_ = f.PresicionClases(self.confusion[tipo]['Prueba'])
        presicion_clases = dict(zip(self.nombre_clases, presicion_clases))
        # concatenar en un solo diccionario
        info.update(presicion_clases)
        info.update(self.metricas[tipo])
        f.GuardarMetricas(info)
        # revisar si guardar los parametros del clasificador.

        self.ActualizarProgreso(tipo, 0.95)


    def Combinacion(self):
        """Metodo Combinaciion:
            
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
        # matriz de pesos
        self.w = f.CalculoPesos(
            self.confusion['EMG']['Validacion'], 
            self.confusion['EEG']['Validacion'])

        # vector de decición
        self.prediccion['Combinada'] = self.prediccion['EMG']*self.w[0] + self.prediccion['EEG']*self.w[1]

        self.confusion['Combinada'] = confusion_matrix(
            argmax(self.class_test['EMG'], axis = 1), 
            argmax(self.prediccion['Combinada'], axis = 1))

        f.GraficaMatrizConfusion(
            self.confusion['Combinada'], self.nombre_clases, self.direccion)

        # Guardar datos
        f.GuardarPkl(self.w, self.direccion + '/Procesamiento/w.pkl')
        # Calculo de exactitud y presición por clases
        presicion_clases, self.exactitud['Combinada'] = f.PresicionClases(
                self.nombre_clases)
        # convertir a diccionario los valores de presición
        presicion_clases = dict(zip(self.nombre_clases, presicion_clases))
        self.exactitud['Combinada'] = self.exactitud['Combinada']*100
        
        info = {
                'Sujeto': self.sujeto, 'Id': self.ubi,
                'Tipo de señales': 'Combinada', 
                'Exactitud': (self.exactitud['Combinada'])}
        # concatenar en un solo diccionario
        info.update(presicion_clases)
        f.GuardarMetricas(info)
        # Actualizar el valor del progreso
        self.ActualizarProgreso('General', 0.99)


    def CargarDatos(self, tipo):
        """Metodo CargarDatos:
        
        Parameters
        ----------
        tipo: STR, el tipo de señales para entrenamiento, puede ser 
            'EMG' o 'EEG'

        Returns
        -------     
        """
        #-----------------------------------------------------------------------------
        # Extraer datos
        datos = f.ExtraerDatos(self.directorio, self.sujeto, tipo)
        
        # Actualiza la variable para hacer seguimiento al progreso
        self.progreso[tipo] = 0.15 

        #-----------------------------------------------------------------------------
        # Filtro
        self.filtro[tipo] = f.AbrirPkl(
                self.direccion + '/Procesamiento/filtro_' + tipo + '.pkl')

        # Actualiza la variable para hacer seguimiento al progreso
        self.progreso[tipo] = 0.20

        #-----------------------------------------------------------------------------
        # Diseñar las clases One Hot
        # Tomar la clase de onehot y asignarla a la clases oh de forma que cada
        # indice corresponda con las banderas. 
        # Dataframe para las clases one-hot
        # clases_OH = []
        # for i in range(3):
        #     clases_OH.append(f.ClasesOneHot(
        #         self.nombre_clases, self.num_clases, 
        #         datos['Final grabacion'][i], datos['Banderas'][i], 
        #         datos['One Hot'][i]))

        # Actualiza la variable para hacer seguimiento al progreso
        self.progreso[tipo] = 0.25
        
        #-----------------------------------------------------------------------------
        # Funcion para submuestreo
        # senales = []
        # for sesion in range(1,4):
        #     senales.append(f.Submuestreo(
        #         self.directorio, tipo, datos, self.sujeto, sesion, 
        #         self.canales[tipo], self.nombre_clases, self.filtro[tipo], 
        #         self.m[tipo]))

        # Calcular a partir de frecuencias de submuestreo
        self.frec_submuestreo[tipo] = int(
                datos['Frecuencia muestreo'] / self.m[tipo])

        self.tam_ventana[tipo] = int(
                self.tam_ventana_ms * 0.001 * self.frec_submuestreo[tipo])
        self.paso_ventana[tipo] = int(
                self.paso_ms * 0.001 * self.frec_submuestreo[tipo])

        # Actualiza la variable para hacer seguimiento al progreso
        self.ActualizarProgreso(tipo, 0.44)

        #-----------------------------------------------------------------------------
        # Enventanado
        # realizar el enventanado, inicialización de los arreglos
        # ventanas, clases = f.Enventanado(
        #     senales[0], clases_OH[0], datos, 0, self.tam_ventana_ms, 
        #     self.paso_ms, self.frec_submuestreo[tipo], self.num_canales[tipo], 
        #     self.num_clases)
        # realizar el enventanado concatenando las ventanas de todas las sesiones
        # for sesion in range(1, 3):
        #     vent, clas = f.Enventanado(
        #         senales[sesion], clases_OH[sesion], datos, sesion, 
        #         self.tam_ventana_ms, self.paso_ms, self.frec_submuestreo[tipo], 
        #         self.num_canales[tipo], self.num_clases)
        #     ventanas = np.concatenate((ventanas,vent), axis=0)
        #     clases = np.concatenate((clases, clas), axis=0)
        # del vent, clas

        # Actualiza la variable para hacer seguimiento al progreso
        self.ActualizarProgreso(tipo, 0.50)

        #-----------------------------------------------------------------------------
        # Divición y balanceo del dataset
        # Descarte de datos de imaginación motora (Revisar si hacer)

        #-----------------------------------------------------------------------------
        # Dividir datos de entrenamiento, test y validación; Además se realiza 
        # el balanceo de base de datos mediate submuestreo aleatoreo
        # Divición y balanceo de dataset datos EMG
        # _, _, _, _, test, self.class_test[tipo] = f.Division(
        #     ventanas, clases, self.porcen_prueba, self.porcen_validacion, 
        #     self.num_clases, todasclases=True)

        # Actualiza la variable para hacer seguimiento al progreso
        self.ActualizarProgreso(tipo, 0.55)

        #-----------------------------------------------------------------------------
        # Extraccion de caracteristicas
        # Cargar FastICA entrenado
        self.ica_total[tipo] = f.AbrirPkl(
                self.direccion + '/Procesamiento/ica_' + tipo + '.pkl')

        # Aplicar FastICA, y guarda las señales obtenidas
        # self.test = f.AplicarICA(
        #     len(test), self.num_ci[tipo], self.tam_ventana[tipo], 
        #     self.ica_total[tipo], test)

        # Actualiza la variable para hacer seguimiento al progreso
        self.ActualizarProgreso(tipo, 0.77)

        #-----------------------------------------------------------------------------
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

        # Cargar metricas
        self.metricas[tipo] = f.AbrirPkl(
                self.direccion + '/General/metricas_' + tipo + '.pkl')

        # valor de la presisión general del modelo entrenado
        self.exactitud[tipo] = 100 * self.metricas[tipo]['categorical_accuracy']
        print("La exactitud del modelo: {:5.2f}%".format(
                    100 * self.metricas[tipo]['categorical_accuracy']))

        # Actualiza la variable para hacer seguimiento al progreso
        self.ActualizarProgreso(tipo, 0.90)
        # Entrenado


    def CargarCombinacion(self):
        """Metodo Combinaciion:
            
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
        # matriz de pesos
        self.w = f.AbrirPkl(
                self.direccion + '/Procesamiento/w.pkl')

        # Actualizar el valor del progreso
        self.ActualizarProgreso('General', 0.99)


    def Procesamiento(self, proceso):
        """Metodo Proce:
            
        Se definen los parametros predeterminados de la 
        interfaz.
        
        Parameters
        ----------
        proceso: STR, el tipo de proceso a realizar, puede ser 
            'Entrenar' o 'Cargar'

        Returns
        -------     
        """
        # Variable para revisar el progreso del entrenamiento
        # Ya se cargaron las librerias
        self.progreso = {'EMG': 0.1, 'EEG': 0.1, 'General': 0.1}

        # revidar como pasar parametros
        # se necesita definir por lo menos sujeto, nombres y nombre_clases
        # para sacar los parametros de la interfaz
        #self.ObtenerParametros()

        # Se calculan a medida que se va ejecutando el código
        # Diccionario para frecuencias de submuestreo
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
        self.confusion = dict.fromkeys(['EMG', 'EEG', 'Combinada'])
        self.metricas = dict.fromkeys(['EMG', 'EEG'])
        self.exactitud = dict.fromkeys(['EMG', 'EEG', 'Combinada'])
        # Para revisar las predicciones y la convinación
        self.prediccion = dict.fromkeys(['EMG', 'EEG', 'Combinada'])
        self.class_test = dict.fromkeys(['EMG', 'EEG'])
        self.test = dict.fromkeys(['EMG', 'EEG'])

        # Para el caso del entrenamiento
        if proceso == "Entrenar":
            # Crear los directorios donde guardar los datos
            self.direccion, self.ubi = f.Directorios(self.sujeto)
            # Entrenamiento de clasificadores en dos hilos
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
            self.Combinacion()

        # Para el caso de cargar los datos
        elif proceso == "Cargar":
            # Determina el de mejor rendimiento
            self.direccion, self.ubi, existe = f.DeterminarDirectorio(
                self.sujeto, 'Combinada')
            # se comprueba que existen datos a cargar
            if existe:
                # Entrenamiento de clasificadores en dos hilos
                hilo_cargar_EMG = threading.Thread(
                    target = self.CargarDatos, args = ('EMG',))
                hilo_cargar_EEG = threading.Thread(
                    target = self.CargarDatos, args = ('EEG',))
                hilo_cargar_Combinacion = threading.Thread(
                    target = self.CargarCombinacion)
    
                # Empieza la ejecución de ambos hilos
                hilo_cargar_EMG.start()
                hilo_cargar_EEG.start()
                hilo_cargar_Combinacion.start()
                # Espera que terminen la ejecución de ambos hilos
                hilo_cargar_EMG.join()
                hilo_cargar_EEG.join()
                hilo_cargar_Combinacion.join()
                # realiza la combinación de los clasificadores entrenados
                #self.CargarDatos('EMG')
                #self.CargarDatos('EEG')
                #self.CargarCombinacion()

        # Actualiza la variable para hacer seguimiento al progreso
        self.ActualizarProgreso('General', 1.00)


principal = Interfaz()
lista = [2, 7, 11, 13, 21, 25]
# lista = [2, 21]
exactitud = dict.fromkeys(lista)
# principal.ObtenerParametros(11)
# principal.Procesamiento('Entrenar')
# exactitud[11] = principal.exactitud
for sujeto in lista:
    principal.ObtenerParametros(sujeto)
    principal.Procesamiento('Entrenar')
    exactitud[sujeto] = principal.exactitud
    
