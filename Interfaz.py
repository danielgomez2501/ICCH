
"""
#!/usr/bin/env python
# libreria mouse 0.5.7 ¿?

import mouse
mouse.move("500", "500")
mouse.click() # default to left click
# mouse.right_click()
# mouse.double_click(button='left')
# mouse.double_click(button='right')
# mouse.press(button='left')
# mouse.release(button='left')
"""
# from typing import List

# Librería Pynput
"""
# control de mouse
from pynput.mouse import Button, Controller

# importing time package para esperar
import time

mouse = Controller()

# Read pointer position
print('The current pointer position is {0}'.format(
    mouse.position))

# Set pointer position
mouse.position = (-1500, 300)
print('Now we have moved it to {0}'.format(
    mouse.position))

# Double click; this is different from pressing and releasing
mouse.click(Button.left, 2)


# Press and release
mouse.press(Button.left)
# mouse.press(Button.right)
for x in range(10):
    # Move pointer relative to current position
    mouse.move(5, 5)
    #time.sleep(0.01)
#
# mouse.release(Button.right)
mouse.release(Button.left)


# Read pointer position
print('The new pointer position is {0}'.format(
    mouse.position))
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 12:35:52 2022

@author: Daniel
"""

# NOTAS:
"""
#-----------------------------------------------------------------------------
# Por revisar
#-----------------------------------------------------------------------------

por alguna razón cuando se hace aumento a la imagen de resumen
todo de descuadra ni puta idea de que es lo que pasa allí

#-----------------------------------------------------------------------------
# Por hacer
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Notas de versiones
#-----------------------------------------------------------------------------

0.10:   Primera versión, contiene todas las ventanas, se inicia la
        integración de los algoritmos ya desarrollados, aun no realiza
        el control del cursor.

0.20    Se busca integrar el procesamiento de señales, siendo el
        entrenamiento y la carga de datos.

"""

# from functools import partial
import threading  # Hilos
import time  # para cosas de tiempo
# Kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.lang import Builder
# from kivy.properties import ObjectProperty
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.clock import Clock
# Funciones y metodos creados
# import Funciones as f


# Valores globales
progreso = 0  # revisa el progreso en el entrenamiento o carga de datos
cancelar = threading.Event()  # para detener la ejecución de un hilo

# Ajustar el tamaño de la interfaz
tam_ven_x = 420
tam_ven_y = 420
Window.size = (tam_ven_x, tam_ven_y)


# funciones
def procesamiento(proceso, sujeto, parametros):
    """
    Para realizar la carga o entrenamiento de la interfaz.

    Parameters
    ----------
    proceso: STR, decide si entrenar o cargar los datos, dos posibles:
        "entrenar" y "cargar".

    sujeto: int, número del sujeto de la base de datos.

    parametros: Objeto, las características sobre la estructura de
        la ICCH.

    Returns
    -------

    """
    global progreso
    # Inicializa la variable de progreso
    progreso = 0

    while progreso < 1:
        time.sleep(0.1)
        progreso += 0.1

        # para cancelar el hilo
        if cancelar.is_set():
            progreso = 0
            # Se puede usar un callback para parar el entrenamiento de la RNA
            return

    print("se terminó el proceso de " + proceso + " del sujeto " + str(sujeto))


# Clases

# Las características de la interfaz
class Caracteristicas:
    """Se definen todas las características y datos de la interfaz.

    Tiene la configuración predeterminada de los parámetros de la
    interfaz.

    Parameters
    ----------

    Returns
    -------

    """

    # Iniciación
    def __init__(self):
        # super(Caracteristicas, self).__init__()

        # Lista de sujetos
        self.lista_sujetos = [2, 7, 11, 13, 21, 25]
        # sujeto
        self.sujeto = 0

    def parametros(self):
        """Método para modificar los parámetros del modelo
        """
        pass


# Interfaz grafica
# Ventana de configuración
class Configuracion(Widget):
    """Ventana de inicio de la interfaz

    Permite la elección de sujeto y el proceso a realizar, también,
    un botón para iniciar proceso seleccionado con los datos del
    sujeto seleccionado

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, **kwargs):
        # para conservar los atributos no modificados
        super(Configuracion, self).__init__(**kwargs)
        # Atributos
        self.proceso = "no seleccionado"  # tipo de proceso a realizar (Entrenamiento o carga)
        # Agregar elementos de lista al menú desplegable
        self.ids.sujeto_sel.values = ['Sujeto ' + str(sujeto) for sujeto in Caracteristicas.lista_sujetos]
        # Definición del reloj para inicializar
        self.reloj = Clock.schedule_interval(self.actualizar, 0.2)

    def ejecutar(self):
        """Proceso del botón ejecutar
        """
        # Variables globales
        global progreso
        global cancelar

        cancelar.clear()  # Para reiniciar la bandera del hilo
        # selección de proceso
        if self.proceso != "no seleccionado" and Caracteristicas.sujeto != 0:

            # hilo para el proceso
            hilo_proceso = threading.Thread(
                    target=procesamiento, args=(
                        self.proceso, Caracteristicas.sujeto, Caracteristicas,))
            hilo_proceso.start()

            # ajuste de widged de progreso
            self.reloj()

            # cambio de ventanas
            Aplicacion.Ventanam.transition.direction = "left"
            Aplicacion.Ventanam.current = "progreso"

    def actualizar(self, _):
        """Para monitorizar el valor de progreso

        Permite revisar el valor de progreso también llama al método
        de avance para actualizar la ventana de Progreso de acuerdo
        con el valor del progreso, también, en el caso de ser
        presionado el botón de cancelar, se detiene el proceso de
        reloj para actualizar dicha ventana.
        """
        global progreso
        # método para revisar el avance del progreso
        self.avance()
        # revisar si se presionó cancelar
        if cancelar.is_set():
            self.reloj.cancel()

    # Es estático ya que no hace uso de ningún atributo de esta clase
    @staticmethod
    def seleccion_sujeto(value):
        """Determina el sujeto seleccionado
        """
        # se extrae el número del sujeto seleccionado
        Caracteristicas.sujeto = int(value.split(' ')[-1])
        Aplicacion.Resumen.ids.texto_sujeto.text = f"Sujeto: {Caracteristicas.sujeto}"

    def seleccion_proceso(self, value):
        """Determina el proceso seleccionado
        """

        if value == "Entrenar":
            Aplicacion.Progreso.ids.tipo_proceso.text = "Entrenando parámetros"
            self.proceso = "entrenar"
        elif value == "Cargar datos":
            Aplicacion.Progreso.ids.tipo_proceso.text = "Cargando parámetros"
            self.proceso = "cargar"
        else:
            self.proceso = "no seleccionado"

    def metricas(self):
        """Muestra las métricas del clasificador entrenado o cargado

        Cuando se acaba de entrenar un clasificador se deben presentar
        las métricas del ultimo clasificador cargado; mientras, que
        cuando se carga un clasificador se presentarían las métricas
        del clasificador con mejor exactitud combinada.
        """
        # Ubicación del archivo
        # directo = 'Parametros/Rendimiento.csv'

        # para entrenamiento

        # Determinar el mejor clasificador
        # path, ubi, existe = f.DeterminarDirectorio(self.sujeto)
        pass

    def avance(self):
        """Determina el progreso del proceso seleccionado.

        En el caso de que el progreso se complete (progreso == 1)
        permite el cambio a la pantalla de resumen, se reinicia el
        valor del progreso y se elimina el reloj.
        """

        # obtiene el valor actual
        global progreso

        # Actualizar el valor
        Aplicacion.Progreso.ids.barra_progreso.value = progreso
        Aplicacion.Progreso.ids.descripcion_progreso.text = f"Progreso {int(progreso*100)}%"

        # revisar si el progreso sé completó
        # y cambia a la ventana siguiente
        if progreso >= 1:
            # Actualiza las métricas
            # self.metricas()
            # Cambiar ventana
            Aplicacion.Ventanam.transition.direction = "left"
            Aplicacion.Ventanam.current = "resumen"
            # valor de progreso
            Aplicacion.Progreso.ids.barra_progreso.value = 0
            progreso = 0
            # Actualizar la etiqueta
            Aplicacion.Progreso.ids.descripcion_progreso.text = "Progreso 0%"

            #  quitar el reloj generado para actualizar el widget
            self.reloj.cancel()


# Ventana de progreso
class Progreso(Widget):
    """Ventana de Progreso de la interfaz

    Muestra el avance que se va realizando a la hora de realizar
    el proceso de entrenamiento o carga de datos de la interfaz,
    tiene un botón que permite parar el proceso.

    Parameters
    ----------

    Returns
    -------

    """

    def cancelar(self):
        global cancelar
        cancelar.set()  # el evento de cancelar el hilo
        # movimiento de la ventana
        Aplicacion.Ventanam.transition.direction = "right"
        Aplicacion.Ventanam.current = "configuracion"
        self.ids.barra_progreso.value = 0
        self.ids.descripcion_progreso.text = "Progreso 0%"


# Ventana de resumen
class Resumen(Widget):
    """Ventana de resumen de la interfaz

    Presenta las métricas de rendimiento de la ICCH entrenada o
    cargada, esto mediante una imagen de una matriz de confusión
    y la precisión general en un texto, además cuenta con dos
    botones, uno permite regresar al inicio, mientras que el
    segundo permite pasar al proceso de utilizar la interfaz.

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, **kwargs):
        """ Función de iniciación
        """
        # para conservar los atributos no modificados
        super(Resumen, self).__init__(**kwargs)
        # para revisar que no exista el botón de reajuste
        self.aparece = False
        # ajustar el valor de la escala
        self.escala = self.ids.scat.scale
        # El botón de reajustar
        # Widget para luego poner el botón
        self.layout = None
        # el botón
        self.reajuste = None

    # botones
    def volver(self):
        """Botón para volver a la pantalla de inicio
        """
        # Reajustar la imagen de métricas en caso de interacción
        self.reajustar(None)
        # transición de ventana
        Aplicacion.Ventanam.transition.direction = "right"
        Aplicacion.Ventanam.current = "configuracion"

    def iniciar(self):
        """ Botón para iniciar el uso de la interfaz
        """

        # reajustar la imagen de métricas en caso de interacción
        self.reajustar(None)
        # transición de ventana
        Aplicacion.Ventanam.transition.direction = "left"
        Aplicacion.Ventanam.current = "monitor"

    def reajustar(self, _):
        """ Botón reajustar métricas después de interacción
        """
        # Reposición de la imagen
        self.ids.scat.scale = self.escala
        self.ids.scat.pos = (
            Window.size[0]/2-self.ids.scat.width*0.45, Window.size[1]*0.3)
        # Quitar el botón de reajuste si ya está en pantalla
        if self.aparece:
            # quita el botón
            self.layout.remove_widget(self.reajuste)
            # quita el layout
            self.remove_widget(self.layout)
            # indica que ya no aparece el botón
            self.aparece = False

    # Aumento en la imagen
    def interaccion(self):
        """Genera el botón de reajuste
        """
        # revisa si ya existe
        if not self.aparece:
            # Widget para luego poner el botón
            # toca ajustarlo en un layout
            self.layout = FloatLayout(size=(self.width, self.height))
            self.add_widget(self.layout)
            # el botón
            self.reajuste = (Button(text="reajustar"))
            self.reajuste.pos_hint = {"x": 0.7, "y": 0.3}
            self.reajuste.size_hint = (0.25, 0.07)
            self.reajuste.bind(on_release=self.reajustar)
            self.layout.add_widget(self.reajuste)
            # aparece el botón en pantalla
            self.aparece = True
        # revisar que no se salga de la pantalla
        # se considera que está en pantalla cuando:
        # -250<x<350; -90<y<310
        # Calculo de los límites para el aumento
        limx = [-255*self.ids.scat.scale+30, -80*self.ids.scat.scale+430]
        limy = [-245*self.ids.scat.scale+165, -75*self.ids.scat.scale+390]
        # limite en x
        if limx[0] > self.ids.scat.pos[0]:
            self.ids.scat.pos = (limx[0], self.ids.scat.pos[1])
        elif self.ids.scat.pos[0] > limx[1]:
            self.ids.scat.pos = (limx[1], self.ids.scat.pos[1])
        # límite en y
        if limy[0] > self.ids.scat.pos[1]:
            self.ids.scat.pos = (self.ids.scat.pos[0], limy[0])
        elif self.ids.scat.pos[1] > limy[1]:
            self.ids.scat.pos = (self.ids.scat.pos[0], limy[1])

    # revisa la interacción que hay sobre la ventana
    def on_touch_down(self, touch):
        """Evento revisa si se utiliza el scroll para hacer aumento

        Al usar el scrool sobre las métricas se hará aumento sobre la
        imagen.
        """
        # para cuando se hace scroll
        if touch.is_mouse_scrolling:
            # tomar el valor de la posición antes de la escalada
            pos = self.ids.scat.pos
            if touch.button == "scrolldown":
                if self.ids.scat.scale < 2:
                    self.ids.scat.scale = self.ids.scat.scale*1.1
            elif touch.button == "scrollup":
                if self.ids.scat.scale > 1:
                    self.ids.scat.scale = self.ids.scat.scale*0.9
            # Para recolocarlo en la posición que estaba
            self.ids.scat.pos = pos
            self.interaccion()
        # para que no afecte los demás funcionamientos
        else:
            super(Resumen, self).on_touch_down(touch)


# Ventana de monitor
class Monitor(Widget):
    """Ventana de monitor de la interfaz.

    Presentaría los comandos del ratón entendidos por la interfaz
    mediante imágenes que cambian el color de acuerdo al comando,
    cuenta con un botón para regresar a la ventana de resumen.

    Parameters
    ----------

    Returns
    -------

    """
    # Iniciación
    def __init__(self, **kwargs):
        # para conservar los atributos no modificados
        super(Monitor, self).__init__(**kwargs)

    # Estático por que no utilizá atributos de este método
    @staticmethod
    def parar():
        """Botón para parar el análisis de las señales
        """
        # Transición de la ventana
        Aplicacion.Ventanam.transition.direction = "right"
        Aplicacion.Ventanam.current = "resumen"


# interfaz principal
class Aplicacion(App):
    """Interfaz grafica de usuario.

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, **kwargs):
        # para conservar los atributos no modificados
        super(Aplicacion, self).__init__(**kwargs)
        # Interfaz Grafica
        # ventana maestra
        self.Ventanam = ScreenManager()
        # Ventanas
        # De configuración
        self.Configuracion = Configuracion()
        # De entrenamiento
        self.Progreso = Progreso()
        # De resumen
        self.Resumen = Resumen()
        # De monitor
        self.Monitor = Monitor()

    def build(self):
        """ Inicio de la interfaz

        El orden en que se agregan las ventanas determina cuál es la
        primera en mostrarse
        """

        # De resumen
        screen = Screen(name="resumen")
        screen.add_widget(self.Resumen)
        self.Ventanam.add_widget(screen)

        # Interfaz grafica
        # De configuración
        screen = Screen(name="configuracion")
        screen.add_widget(self.Configuracion)
        self.Ventanam.add_widget(screen)
        # De entrenamiento
        screen = Screen(name="progreso")
        screen.add_widget(self.Progreso)
        self.Ventanam.add_widget(screen)

        # De monitor
        screen = Screen(name="monitor")
        screen.add_widget(self.Monitor)
        self.Ventanam.add_widget(screen)
        # Color del fondo de la aplicación en rgba
        # rgba: Rojo, Verde, Azul, Transparencia
        Window.clearcolor = (1, 1, 1, 1)

        # Retorna de esta manera haya múltiples pantallas
        return self.Ventanam


# Ejecución de la interfaz
if __name__ == "__main__":
    # Objeto que contiene las características de la interfaz
    Caracteristicas = Caracteristicas()

    # Para escoger el archivo de Kivy
    Builder.load_file("principal.kv")
    # Objeto para la interfaz
    Aplicacion = Aplicacion()

    # Hilo para ejecutar la interfaz gráfica.
    threading.Thread(target=Aplicacion.run()).start()
