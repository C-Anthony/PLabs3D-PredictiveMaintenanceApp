# Imports
# tkinter
import tkinter as tk
from tkinter import ttk, filedialog
from tkinter import N, S, W, E, HORIZONTAL, NO, VERTICAL
from tkinter import Label, LabelFrame, Scale, Button, Toplevel, Entry, Canvas, Scrollbar, Frame
from tkinter.scrolledtext import ScrolledText
from tkinter.ttk import Treeview
# matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# librosa
import librosa
import librosa.display
# otros
import re
import os
import os.path
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import imageio
from datetime import datetime
import queue
import logging
import platform
import schedule
import time
import threading

logger = logging.getLogger('application_log')


class QueueHandler(logging.Handler):
    """Class to send logging records to a queue
    It can be used from different threads
    The ConsoleUi class polls this queue to display records in a ScrolledText widget
    """
    # Example from Moshe Kaplan: https://gist.github.com/moshekaplan/c425f861de7bbf28ef06
    # (https://stackoverflow.com/questions/13318742/python-logging-to-tkinter-text-widget) is not thread safe!
    # See https://stackoverflow.com/questions/43909849/tkinter-python-crashes-on-new-thread-trying-to-log-on-main-thread

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(record)


class ConsoleUi:
    """Poll messages from a logging queue and display them in a scrolled text widget"""

    def __init__(self, frame):
        self.frame = frame
        # Create a ScrolledText widget
        self.scrolled_text = ScrolledText(frame, state='disabled', height=8, width=55)
        self.scrolled_text.grid(row=0, column=0, sticky=(N, S, W, E))
        self.scrolled_text.configure(font='TkFixedFont')
        self.scrolled_text.tag_config('INFO', foreground='black')
        self.scrolled_text.tag_config('DEBUG', foreground='gray')
        self.scrolled_text.tag_config('WARNING', foreground='orange')
        self.scrolled_text.tag_config('ERROR', foreground='red')
        self.scrolled_text.tag_config('CRITICAL', foreground='red', underline=1)
        # Create a logging handler using a queue
        self.log_queue = queue.Queue()
        self.queue_handler = QueueHandler(self.log_queue)
        formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%H:%M:%S')
        self.queue_handler.setFormatter(formatter)
        logger.addHandler(self.queue_handler)
        # Start polling messages from the queue
        self._job = self.frame.after(100, self.poll_log_queue)

    def display(self, record):
        msg = self.queue_handler.format(record)
        self.scrolled_text.configure(state='normal')
        self.scrolled_text.insert(tk.END, msg + '\n', record.levelname)
        self.scrolled_text.configure(state='disabled')
        # Autoscroll to the bottom
        self.scrolled_text.yview(tk.END)

    def poll_log_queue(self):
        # Check every 100ms if there is a new message in the queue to display
        while 1:
            try:
                record = self.log_queue.get(block=False)
            except queue.Empty:
                break
            else:
                self.display(record)
        self._job = self.frame.after(100, self.poll_log_queue)

    def cancel_jobs_from_queue(self):
        if self._job is not None:
            self.frame.after_cancel(self._job)
            self._job = None


class Tabla:
    """Classe para crear una tabla en la que se puede añadir y suprimir elementos, tal como refrescar el contenido"""
    def __init__(self, master, columns_data, height=10, row=0, column=0, rowspan=1, columnspan=1, sticky=None):
        # Crear tabla
        self.tabla = Treeview(master=master, height=height)
        # Crear columnas en funcion de la lista (omitir la primera)
        self.tabla['columns'] = [column_data[0] for column_data in columns_data]
        self.tabla.column("#0", width=0, stretch=NO)  # Para que no aparezca la primera columna
        for column_data in columns_data:
            self.tabla.column(column=column_data[0], width=column_data[2], anchor=W)
            self.tabla.heading(column=column_data[0], text=column_data[1], anchor=W)
        # Insertar en el frame
        self.tabla.grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan, sticky=sticky)

    def add_line_csv(self, tiempo1, tiempo2, tipo=""):
        # Creacion de fecha con formato
        now = datetime.now()
        format_now = now.strftime("%d/%m/%Y %H:%M:%S")
        #  Recuperar los datos a añadir y ponerlos en un dataframe
        self.datos = [[format_now, tiempo1, tiempo2, tipo]]
        columnas = ['Creacion', 'Tiempo1', 'Tiempo2', 'Tipo']
        df_nueva_entrada = pd.DataFrame(self.datos, columns=columnas)
        # Append dataframe to existente
        try:
            path = self.directory_place + '/' + self.archivo  # modificar con el verdadero archivo
        except AttributeError:
            logger.log(logging.ERROR, "Archivos no cargados")
            return
        df_nueva_entrada.to_csv(path, index=False, mode='a', header=not os.path.isfile(path))
        # Actualizar datos de la tabla
        self.load_content_csv(self.directory_place, self.archivo)
        # Mostrar un mensaje de confirmacion al usuario
        logger.log(logging.INFO, "Incidencia grabada")

    def add_line_txt(self, nueva_incidencia):
        # Añadir a la lista la incidencia recuperada del usuario
        self.tipos_incidencias.append(nueva_incidencia)
        # Añadir el dato en la tabla
        self.tabla.insert(parent='',
                          index='end',
                          values=(nueva_incidencia,))

    def del_line_csv(self):
        try:
            # Dataframe ya cargado en self.incidencias_datos, encontrar la linea correspondiente y suprimirla
            self.incidencias_datos = self.incidencias_datos.drop(self.incidencias_datos.loc[
                                                                     self.incidencias_datos['Creacion'] ==
                                                                     self.tabla.item(self.tabla.selection())['values'][
                                                                         0]].index[0])
        except IndexError:  # Si no se ha seleccionado ninguna linea
            logger.log(logging.ERROR, "Por favor seleccionar una incidencia")
            return
        except AttributeError:
            logger.log(logging.ERROR, "Archivos no cargados")
            return
        # Sobreescribir el dataframe en el csv
        self.incidencias_datos.to_csv(self.directory_place + '/' + self.archivo, index=False)
        # Actualizar datos de la tabla
        self.load_content_csv(self.directory_place, self.archivo)
        # Log
        logger.log(logging.INFO, "Incidencia eliminada")

    def del_line_txt(self):
        try:
            self.tipos_incidencias.remove(self.tabla.item(self.tabla.selection())['values'][0])
        except IndexError:
            return
        self.tabla.delete(self.tabla.selection()[0])

    def load_content_txt(self, archivo):
        """ Carga las incidencias en la tabla a partir de un txt"""
        # Recuperar los datos del archivo de texto
        with open(archivo) as file:
            self.tipos_incidencias = file.readlines()
        # Limpieza de datos de la lista y se añade el tipo Otros
        self.tipos_incidencias = [re.sub(r"[^ a-zA-Z0-9]", "", tipo) for tipo in self.tipos_incidencias]
        # Recorrer cada linea para insertarla en la tabla
        for tipo in self.tipos_incidencias:
            self.tabla.insert(parent='',
                              index='end',
                              values=(tipo,))

    def load_content_csv(self, directory_place, archivo):
        """ Carga las incidencias en la tabla a partir de un csv"""
        self.directory_place = directory_place
        self.archivo = archivo
        # Recuperar el archivo de incidencias actualizado
        incidencias_datos = pd.read_csv(directory_place + '/' + archivo)
        # Reordenar las incidencias segun tiempo1
        incidencias_datos = incidencias_datos.sort_values(by=['Tiempo1'], ascending=True)
        # Reset Index
        incidencias_datos.reset_index(inplace=True, drop=True)
        # Recuperar una version del df actualizado para la posible supresion de lineas
        self.incidencias_datos = incidencias_datos
        # Limpiar la tabla de datos
        self.tabla.delete(*self.tabla.get_children())

        # Para cada incidencia, cargar la tabla con los datos
        for index, row in incidencias_datos.iterrows():
            self.tabla.insert(parent='',
                              index='end',
                              values=(row['Creacion'],
                                      row['Tiempo1'],
                                      row['Tiempo2'],
                                      row['Tipo']))

    def reset(self):
        """Limpia la tabla de incidencias de los datos que pudieran existir"""
        self.tabla.delete(*self.tabla.get_children())


class VideoFrameByFrame:
    """ Integrate the video in a label of tkinter, and permit to go frame by frame forward or backward
    or to select a frame in the range of the video"""

    def __init__(self, video, label):
        vid_reader = imageio.get_reader(video)
        mdata = vid_reader.get_meta_data()
        self.frame_per_second = mdata['fps']
        self.frame_number = 0
        self.label = label
        self.list_frames = [frame for frame in vid_reader.iter_data()]

    def next_frame(self):
        self.frame_number += 1
        self.render_image()

    def previous_frame(self):
        self.frame_number -= 1
        self.render_image()

    def on_spot_frame(self, on_spot_frame_number):
        if float(on_spot_frame_number) < 0:
            self.frame_number = 0
        else:
            self.frame_number = int(float(on_spot_frame_number) * self.frame_per_second)
        self.render_image()

    def render_image(self):
        try:
            image_frame = Image.fromarray(self.list_frames[self.frame_number])
        except IndexError:  # Si se sale del rango, vuelve al minimo o al maximo del rango
            if abs(self.frame_number - 0) < abs(self.frame_number - len(self.list_frames)-1):
                self.frame_number = 0
                image_frame = Image.fromarray(self.list_frames[self.frame_number])
            else:
                self.frame_number = len(self.list_frames)-1
                image_frame = Image.fromarray(self.list_frames[self.frame_number])
        image_frame = image_frame.resize((320, 240))
        frame_image = ImageTk.PhotoImage(image_frame)
        self.label.config(image=frame_image)
        self.label.image = frame_image


class AudioGraphic:
    """Integrate two graphics for one audio file, with integration of a cursor, and zoom / pan"""

    def __init__(self, audio, tiempo_max, frame, row, column, columnspan, color_spectrograma, color_waveplot):
        # Variables
        self.audio = audio
        self.tiempo_max = tiempo_max
        self.frame = frame
        self.row = row
        self.column = column
        self.columnspan = columnspan
        self.color_spectrograma = color_spectrograma
        self.color_waveplot = color_waveplot
        self.press = None
        self.dy0 = 0

        # Recuperacion de los datos de audio
        self.y, self.sr = librosa.load(self.audio)

        # Creacion de Specshow
        self.fig_specshow, self.ax_specshow = plt.subplots(figsize=(20, 2), dpi=50)
        # Creacion grafico para spectrum
        self.db = librosa.amplitude_to_db(np.abs(librosa.stft(self.y)), ref=np.max)
        self.img = librosa.display.specshow(self.db, y_axis='log', x_axis='time', sr=self.sr, ax=self.ax_specshow)
        # Fijar limites
        self.ax_specshow.set_xlim([0, self.tiempo_max])
        self.ax_specshow.set_facecolor('xkcd:black')
        # Mostrar grafico
        self.canvas_specshow = FigureCanvasTkAgg(self.fig_specshow, master=self.frame)
        self.canvas_specshow.get_tk_widget().grid(row=self.row, column=self.column, columnspan=self.columnspan)
        self.canvas_specshow.draw()
        # Crear copia de background
        self.specshow_bg = self.canvas_specshow.copy_from_bbox(self.ax_specshow.bbox)
        # Añadir linea vertical (cursor)
        self.specshow_ly = self.ax_specshow.axvline(color=self.color_spectrograma)

        # Creacion de Waveplot
        self.fig_waveplot, self.ax_waveplot = plt.subplots(figsize=(20, 2), dpi=50)
        # Creacion grafico para waveplot
        librosa.display.waveplot(self.y, sr=self.sr, ax=self.ax_waveplot)
        # Fijar limites
        self.ax_waveplot.set_xlim([0, self.tiempo_max])
        # Mostrar grafico
        self.canvas_waveplot = FigureCanvasTkAgg(self.fig_waveplot, master=self.frame)
        self.canvas_waveplot.get_tk_widget().grid(row=self.row+1, column=self.column, columnspan=self.columnspan)
        self.canvas_waveplot.draw()
        # Crear copia de figure background
        self.waveplot_bg = self.canvas_waveplot.copy_from_bbox(self.ax_waveplot.bbox)
        # Añadir linea vertical (cursor)
        self.waveplot_ly = self.ax_waveplot.axvline(color=self.color_waveplot)

    def cursor_move(self, zoom, tiempo):
        tiempo = float(tiempo)

        if zoom == 0:
            self.canvas_specshow.restore_region(self.specshow_bg)  # Restaura el gráfico original
        else:
            self.canvas_specshow.restore_region(self.specshow_bg_2)  # Restaura el gráfico con el zoom
        self.specshow_ly.set_xdata(tiempo)
        self.ax_specshow.draw_artist(self.specshow_ly)
        self.canvas_specshow.blit(self.ax_specshow.bbox)

        if zoom == 0:
            self.canvas_waveplot.restore_region(self.waveplot_bg)   # Restaura el gráfico original
        else:
            self.canvas_waveplot.restore_region(self.waveplot_bg_2)  # Restaura el gráfico con el zoom
        self.waveplot_ly.set_xdata(tiempo)
        self.ax_waveplot.draw_artist(self.waveplot_ly)
        self.canvas_waveplot.blit(self.ax_waveplot.bbox)

    def zoom_graph(self, zoom, x):
        # Coordenadas del punto de origen (coordenadas y limitadas al punto central de cada grafico)
        self.x = float(x)
        zoom = int(zoom)

        try:  # Si no hay coordenadas de origen
            type(self.antiguo_zoom)
        except AttributeError:
            self.antiguo_zoom = 1
            self.base_xlim_specshow = self.ax_specshow.get_xlim()
            self.base_xlim_waveplot = self.ax_waveplot.get_xlim()
            self.base_ylim_specshow = self.ax_specshow.get_ylim()
            self.base_ylim_waveplot = self.ax_waveplot.get_ylim()
            self.y_specshow = (self.base_ylim_specshow[1]-self.base_ylim_specshow[0])/2
            self.y_waveplot = 0
        if zoom == 0:
            self.ax_specshow.set_xlim(self.base_xlim_specshow)
            self.ax_specshow.set_ylim(self.base_ylim_specshow)
            self.ax_specshow.figure.canvas.draw_idle()
            self.ax_specshow.draw_artist(self.specshow_ly)

            self.ax_waveplot.set_xlim(self.base_xlim_waveplot)
            self.ax_waveplot.set_ylim(self.base_ylim_waveplot)
            self.ax_waveplot.figure.canvas.draw_idle()
            self.ax_waveplot.draw_artist(self.waveplot_ly)

        else:
            if self.antiguo_zoom > zoom:
                self.zoom = 2
                self.light_zoom = 1.2
            elif self.antiguo_zoom < zoom:
                self.zoom = 0.5
                self.light_zoom = 1/1.2
            else:
                self.zoom = 1

            self.antiguo_zoom = zoom
            # get the current x and y limits
            cur_xlim_specshow = self.ax_specshow.get_xlim()
            cur_ylim_specshow = self.ax_specshow.get_ylim()
            cur_xlim_waveplot = self.ax_waveplot.get_xlim()
            cur_ylim_waveplot = self.ax_waveplot.get_ylim()
            # set the range
            cur_xrange_specshow = (cur_xlim_specshow[1] - cur_xlim_specshow[0]) * .5
            cur_yrange_specshow = (cur_ylim_specshow[1] - cur_ylim_specshow[0]) * .5
            cur_xrange_waveplot = (cur_xlim_waveplot[1] - cur_xlim_waveplot[0]) * .5
            cur_yrange_waveplot = (cur_ylim_waveplot[1] - cur_ylim_waveplot[0]) * .5
            # set new limits for specshow
            self.ax_specshow.set_xlim([self.x - cur_xrange_specshow * self.zoom,
                         self.x + cur_xrange_specshow * self.zoom])
            self.ax_specshow.set_ylim([self.y_specshow - cur_yrange_specshow*self.light_zoom,
                         self.y_specshow + cur_yrange_specshow*self.light_zoom])
            self.ax_specshow.figure.canvas.draw_idle()
            self.canvas_specshow.flush_events()  # Evitar tener muchos dibujos a la vez
            self.specshow_bg_2 = self.canvas_specshow.copy_from_bbox(self.ax_specshow.bbox)  # Guarda BG con zoom
            # set new limits for waveplot
            self.ax_waveplot.set_xlim([self.x - cur_xrange_waveplot * self.zoom,
                              self.x + cur_xrange_waveplot * self.zoom])
            self.ax_waveplot.set_ylim([self.y_waveplot - cur_yrange_waveplot * self.light_zoom,
                              self.y_waveplot + cur_yrange_waveplot * self.light_zoom])
            self.ax_waveplot.figure.canvas.draw_idle()
            self.canvas_waveplot.flush_events()  # Evitar tener muchos dibujos a la vez
            self.waveplot_bg_2 = self.canvas_waveplot.copy_from_bbox(self.ax_waveplot.bbox)  # Guarda BG con zoom

    def connect(self):
        self.cid_waveplot_press = self.ax_waveplot.figure.canvas.mpl_connect('button_press_event',
                                                                             self.on_press_waveplot)
        self.cid_waveplot_release = self.ax_waveplot.figure.canvas.mpl_connect('button_release_event',
                                                                               self.on_release_waveplot)
        self.cid_waveplot_motion = self.ax_waveplot.figure.canvas.mpl_connect('motion_notify_event',
                                                                              self.on_motion_waveplot)

        self.cid_specshow_press = self.ax_specshow.figure.canvas.mpl_connect('button_press_event',
                                                                             self.on_press_specshow)
        self.cid_specshow_release = self.ax_specshow.figure.canvas.mpl_connect('button_release_event',
                                                                               self.on_release_specshow)
        self.cid_specshow_motion = self.ax_specshow.figure.canvas.mpl_connect('motion_notify_event',
                                                                              self.on_motion_specshow)

    def on_press_waveplot(self, event):
        if event.inaxes != self.ax_waveplot.axes:
            return
        self.press = self.ax_waveplot.get_ylim(), event.ydata

    def on_motion_waveplot(self, event):
        if self.press is None or event.inaxes != self.ax_waveplot.axes:
            return
        y0, ypress = self.press
        dy1 = event.ydata - ypress
        if self.dy0 != dy1:
            self.ax_waveplot.set_ylim(y0 - dy1)
            self.ax_waveplot.figure.canvas.draw_idle()
            self.dy0 = dy1

    def on_release_waveplot(self, event):
        self.press = None
        self.ax_waveplot.figure.canvas.draw_idle()

    def on_press_specshow(self, event):
        if event.inaxes != self.ax_specshow.axes:
            return
        self.press = self.ax_specshow.get_ylim(), event.ydata

    def on_motion_specshow(self, event):
        if self.press is None or event.inaxes != self.ax_specshow.axes:
            return
        y0, ypress = self.press
        dy1 = event.ydata - ypress
        if self.dy0 != dy1:
            self.ax_specshow.set_ylim(y0 - dy1)
            self.ax_specshow.figure.canvas.draw_idle()
            self.dy0 = dy1

    def on_release_specshow(self, event):
        self.press = None
        self.ax_specshow.figure.canvas.draw_idle()

    def disconnect(self):
        self.ax_waveplot.figure.canvas.mpl_disconnect(self.cid_waveplot_press)
        self.ax_waveplot.figure.canvas.mpl_disconnect(self.cid_waveplot_motion)
        self.ax_waveplot.figure.canvas.mpl_disconnect(self.cid_waveplot_release)

        self.ax_specshow.figure.canvas.mpl_disconnect(self.cid_specshow_press)
        self.ax_specshow.figure.canvas.mpl_disconnect(self.cid_specshow_motion)
        self.ax_specshow.figure.canvas.mpl_disconnect(self.cid_specshow_release)


class Graphic:
    """Integrate one graphic for one type of data, with integration of a cursor, and zoom"""

    def __init__(self, datos, variable_tiempo, variable, tiempo_max, frame, row, column, columnspan, color):
        # Variables
        self.datos = datos
        self.variable_tiempo = variable_tiempo
        self.variable = variable
        self.tiempo_max = tiempo_max
        self.frame = frame
        self.row = row
        self.column = column
        self.columnspan = columnspan
        self.color = color

        # Creacion de grafico y plots
        self.fig, self.ax = plt.subplots(figsize=(20, 1), dpi=50)
        # Recuperacion de datos para el plot
        x = self.datos[self.variable_tiempo[1]]
        y = self.datos[self.variable[1]]
        self.res_x, self.res_y = max(x) / 100, (max(y) - min(y)) / 100
        # Plot
        self.ax.plot(x, y, marker='.')
        # Fijar limites
        self.ax.set_xlim([0, self.tiempo_max])
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().grid(row=self.row, column=self.column, columnspan=self.columnspan)
        self.canvas.draw()
        # Crear copia de background
        self.bg = self.canvas.copy_from_bbox(self.ax.bbox)
        # Añadir linea vertical (cursor)
        self.ly = self.ax.axvline(color=self.color)

    def cursor_move(self, zoom, tiempo):
        tiempo = float(tiempo)
        if zoom == 0:
            self.canvas.restore_region(self.bg)  # Restaura el grafico original
        else:
            self.canvas.restore_region(self.bg_2)  # Restaura el grafico con el zoom
        self.ly.set_xdata(tiempo)
        self.ax.draw_artist(self.ly)
        self.canvas.blit(self.ax.bbox)

    def zoom_graph(self, zoom, x, y):
        # Coordenadas del punto de origen
        self.x = float(x)
        self.y = float(y)
        zoom = int(zoom)

        try:
            type(self.antiguo_zoom)
        except AttributeError:
            self.antiguo_zoom = 0
            self.base_xlim = self.ax.get_xlim()
            self.base_ylim = self.ax.get_ylim()
        if zoom == 0:
            self.ax.set_xlim(self.base_xlim)
            self.ax.set_ylim(self.base_ylim)
            self.ax.figure.canvas.draw_idle()
            self.canvas.flush_events()
        else:
            if self.antiguo_zoom > zoom:
                self.zoom = 2
                self.light_zoom = 1.2
            elif self.antiguo_zoom < zoom:
                self.zoom = 0.5
                self.light_zoom = 1 / 1.2
            else:
                self.zoom = 1
            self.antiguo_zoom = zoom
            # Get the current x and y limits
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
            # Set the range
            cur_xrange = (cur_xlim[1] - cur_xlim[0]) * .5
            cur_yrange = (cur_ylim[1] - cur_ylim[0]) * .5
            # Set new limits
            self.ax.set_xlim([self.x - cur_xrange * self.zoom,
                              self.x + cur_xrange * self.zoom])
            self.ax.set_ylim([self.y - cur_yrange * self.light_zoom,
                              self.y + cur_yrange * self.light_zoom])
            self.ax.figure.canvas.draw_idle()
            self.canvas.flush_events()  # Para evitar tener varios dibujos a la vez
            self.bg_2 = self.canvas.copy_from_bbox(self.ax.bbox)  # Background guardado para mover cursor mientras zoom



def extrapolacion(a, b, x):
    """En funcion de un punto a de coordenadas (x1, y1); de un punto b de coordenadas (x2, y2); de un valor x;
      devuelve un punto 'y' correspondiente al valor de x en la recta uniendo los puntos a y b"""
    x1, y1 = a
    x2, y2 = b
    y = (((x - x1) / (x2 - x1)) * (y2 - y1)) + y1
    return y

def rellenar_datos_con_extrapolacion(dataframe):
    """Entrada de un dataframe con datos nulos o sin informacion presentes en el dataframe,
    devuelve un dataframe con datos extrapolados segun los datos proporcionados,
    a partir de la funcion extrapolacion definida anterirmente, a partir del primer punto, y hasta el ultimo punto.
    es decir que si no hay datos al principio, no se extrapolan hasta tener el primer dato (lo mismo para el final).

    Permite tener un dataframe con varias variables con los mismos datos x.
    Es decir que si una variable tiene un dato en  una posicion x, pero la siguiente variable no la tiene,
    se creara esta posicion x para esta segunda variable y se extrapolara su correspondiente y."""

    # Transformar todos los datos en numericos, los que no se pueden transformar se dejan en NaN
    dataframe = dataframe.apply(pd.to_numeric, errors='coerce')

    # Recuperar las columnas donde hay falta de datos
    columnas_filtradas = {colname: data for (colname, data) in dataframe.iteritems() if data.isnull().sum() != 0}

    # Recuperar primeros y ultimos datos de cada columna
    first_last = [(column.first_valid_index(), column.last_valid_index()) for column in columnas_filtradas.values()]

    # A partir del primer índice válido hasta el último índice válido, rellenar los NaN por una extrapolacion
    # Iterar sobre las columnas
    for indice_columnas, (nombre_columna, datos_columna) in enumerate(columnas_filtradas.items()):
        # Iterar sobre valores
        for indice_valor, valor in enumerate(datos_columna[first_last[indice_columnas][0]:first_last[indice_columnas][1]]):
            # Si el valor es nulo
            if pd.isnull(valor):
                # Buscar el indice anterior, y el siguiente indice donde hay datos
                indice_anterior_con_datos = datos_columna[:indice_valor+first_last[indice_columnas][0]].last_valid_index()
                siguiente_indice_con_datos = datos_columna[indice_valor+first_last[indice_columnas][0]:].first_valid_index()

                # Recuperar coordenadas de puntos a y b para la extrapolacion
                a = (indice_anterior_con_datos, datos_columna[indice_anterior_con_datos])
                b = (siguiente_indice_con_datos, datos_columna[siguiente_indice_con_datos])

                # Extrapolacion
                dato_extrapolado = (round(extrapolacion(a, b, indice_valor+first_last[indice_columnas][0]), 2))

                # Grabacion en el dataframe
                dataframe[nombre_columna][indice_valor + first_last[indice_columnas][0]] = dato_extrapolado

    # Devolver el dataframe completado
    return dataframe

def extrapolive(x, caracteristica, dataframe, caracteristica_tiempo):
    """ A partir de una posicion x, en un dataframe dado y para una caracteristica del dataframe dado,
    devuelve un y correspondiente al x, extrapolando con los valores vecinos al x.

    Permite obtener un valor y en un punto x que no existe en un dataframe."""

    # Transformar indice
    indice = min(np.searchsorted(dataframe[caracteristica_tiempo], x), len(dataframe[caracteristica_tiempo])-1)

    # Recuperar el indice anterior y el indice siguiente
    indice_anterior = dataframe[caracteristica_tiempo][:indice].last_valid_index()
    siguiente_indice = dataframe[caracteristica_tiempo][indice:].first_valid_index()

    # Recuperar coordenadas de puntos a y b para la extrapolacion
    a = (dataframe[caracteristica_tiempo][indice_anterior], dataframe[caracteristica][indice_anterior])
    b = (dataframe[caracteristica_tiempo][siguiente_indice], dataframe[caracteristica][siguiente_indice])

    # Si el punto x es mayor que el segundo punto, o si no hay dato de la caracteristica (=NaN),
    # entonces salir con AttributeError, gestionado despues para mostrar "Sin informacion"
    if x > b[0] or pd.isna(a[1]) or pd.isna(b[1]):
        raise AttributeError('Fuera de rango')
    else:
        # Extrapolacion
        dato_extrapolado = (round(extrapolacion(a, b, x), 2))
        return dato_extrapolado


class Application(tk.Frame):

    #### ---- VARIABLES ---- ####
    directory_place = ""  # String para la ubicacion de los archivos
    """ Para los datos:
            1-Nombre de la caracteristica que se muestra en pantalla 
            2-parametro usado en el csv"""
    tiempo_var = ('Segundos', 'segundos')
    extrusor_var = ('Temperatura\nExtrusor', 'extru_temp')
    bed_var = ('Temperatura\nPlataforma', 'bed_temp')
    vent_var = ('Ventilador', 'ventilador')
    x_var = ('X', 'x')
    y_var = ('Y', 'y')
    z_var = ('Z', 'z')

    application_dimensions = (1680,1050)  # width, heigth

    second_per_frame = 1
    is_playing = 0
    has_job = 0


    #### ---- FUNCIONES ---- ####

    # Funcion para la gestion de botones Desde_Hasta:
    def gestion_desde_hasta(self, boton):
        """ Segun el boton apretado, si el otro boton esta 'fijado', entonces graba la incidencia con el tiempo
        marcado en el otro boton y con el tiempo actual.
        Si el otro boton no esta fiajdo, entonces fija el boton seleccionado

        Se usa solamente para los botones desde/hasta."""
        if boton == 'desde':
            if self.boton_DH_hasta['relief'] == 'sunken':
                self.boton_DH_hasta['relief'] = 'raised'
                numero = re.search(r"\d", self.boton_DH_hasta['text'])
                try:
                    self.tabla_incidencias.add_line_csv(self.tiempo, self.boton_DH_hasta['text'][numero.start():],
                                                    self.boton_seleccion_incidencias.get())
                except AttributeError:
                    logger.log(logging.ERROR, "Archivos no cargados")
                    return
            elif self.boton_DH_desde['relief'] == 'sunken':
                self.boton_DH_desde['relief'] = 'raised'
            else:
                self.boton_DH_desde['relief'] = 'sunken'

        elif boton == 'hasta':
            if self.boton_DH_desde['relief'] == 'sunken':
                self.boton_DH_desde['relief'] = 'raised'
                numero = re.search(r"\d", self.boton_DH_desde['text'])
                try:
                    self.tabla_incidencias.add_line_csv(self.boton_DH_desde['text'][numero.start():], self.tiempo,
                                                    self.boton_seleccion_incidencias.get())
                except AttributeError:
                    logger.log(logging.ERROR, "Archivos no cargados")
                    return
            elif self.boton_DH_hasta['relief'] == 'sunken':
                self.boton_DH_hasta['relief'] = 'raised'
            else:
                self.boton_DH_hasta['relief'] = 'sunken'

    # Funcion para recuperar la ubicacion de la carpeta de archivos y cargar archivos
    def recuperar_carpeta(self):
        # Recuperar la carpeta y su "path"
        try:
            self.directory_place = filedialog.askdirectory()
            # Mostrar el "path" de la carpeta en la aplicacion
            self.etiqueta_ubicacion_archivos['text'] = self.directory_place
            # Contar el numero de archivos y mostrarlo
            total_archivos = len([name for name in os.listdir(self.directory_place) if
                                  os.path.isfile(os.path.join(self.directory_place, name))])
        except FileNotFoundError:
            logger.log(logging.ERROR, 'No se ha seleccionado ninguna carpeta')
            return

        def cargar_archivos_y_generar_graficos():
            # Borrar todas las 'figures' de Matplotlib existentes
            plt.close('all')

            # Reiniciar el self.has_job para coger el nuevo fps de los videos y limpiar el schedule
            self.has_job=0
            schedule.clear()

            # Mostrar el numero total de archivos
            self.etiqueta_cantidad_archivos['text'] = "Archivos: " + str(total_archivos)
            # Mostrar texto de espera "Cargando archivos"
            logger.log(logging.INFO, 'Cargando archivos...\nPor favor espere...')
            # Deshabilitar el botón mientras se cargan los archivos.
            self.boton_cargar["state"] = "disabled"

            # Recuperar los archivos en la carpeta indicada
            files = [f for f in os.listdir(self.directory_place)]
            # Reset del tiempo max de los archivos
            self.tiempo_max = 0

            # Por cada archivo en la carpeta
            for f in files:
                # Videos
                if f.endswith('.avi'):
                    if f.find('L') != -1:
                        self.videoL = self.directory_place + '/' + f
                        self.vid_readerL = imageio.get_reader(self.videoL)
                        mdata = self.vid_readerL.get_meta_data()
                        self.second_per_frame = 1/mdata['fps']
                        if self.tiempo_max < mdata['duration']:
                            self.tiempo_max = mdata['duration']
                    if f.find('R') != -1:
                        self.videoR = self.directory_place + '/' + f
                        self.vid_readerR = imageio.get_reader(self.videoR)
                        mdata = self.vid_readerR.get_meta_data()
                        if self.tiempo_max < mdata['duration']:
                            self.tiempo_max = mdata['duration']

                # Audios
                elif f.endswith('.wav'):
                    if f.find('L') != -1:
                        self.audioL = self.directory_place + '/' + f
                        if self.tiempo_max < librosa.get_duration(filename=self.audioL):
                            self.tiempo_max = librosa.get_duration(filename=self.audioL)
                    if f.find('R') != -1:
                        self.audioR = self.directory_place + '/' + f
                        if self.tiempo_max < librosa.get_duration(filename=self.audioR):
                            self.tiempo_max = librosa.get_duration(filename=self.audioR)

                # Datos
                elif f.endswith('.csv'):
                    # si es el archivo de datos o el de incidencias
                    if f == "incidencias.csv":
                        self.incidencias_datos = pd.read_csv(self.directory_place + '/' + f, header=0)
                    else:
                        self.datos = pd.read_csv(self.directory_place + '/' + f)
                        if self.tiempo_max < self.datos.at[self.datos.index[-1], "segundos"]:
                            self.tiempo_max = self.datos.at[self.datos.index[-1], "segundos"]
                        self.datos = rellenar_datos_con_extrapolacion(self.datos)
            # Contar el numero de archivos de audio, vid_reader y datos y mostrarlo
            total_datos = 0
            total_audios = 0
            total_videos = 0
            total_incidencias = 0
            for archivo in os.listdir(self.directory_place):
                if archivo.endswith('.csv'):
                    if archivo != "incidencias.csv":
                        total_datos += 1
                    elif archivo == "incidencias.csv":
                        total_incidencias += 1
                elif archivo.endswith('.wav'):
                    total_audios += 1
                elif archivo.endswith('.avi'):
                    total_videos += 1
            self.etiqueta_cantidad_audios['text'] = "Audio: " + str(total_audios)
            self.etiqueta_cantidad_videos['text'] = "Video: " + str(total_videos)
            self.etiqueta_cantidad_datos['text'] = "Datos: " + str(total_datos)
            self.etiqueta_cantidad_incidencias['text'] = "Incidencias: " + str(total_incidencias)

            # Crear graficos y visualizaciones
            # Videos
            self.videoL_viewer = VideoFrameByFrame(self.videoL, self.label_videoL)
            self.videoL_viewer.on_spot_frame(0)  # Inicializacion del video en la primera imagen
            self.videoR_viewer = VideoFrameByFrame(self.videoR, self.label_videoR)
            self.videoR_viewer.on_spot_frame(0)  # Inicializacion del video en la primera imagen

            # Audios
            # Audio L
            self.grafico_audioL = AudioGraphic(self.audioL, self.tiempo_max, self.data_frame, row=7, column=4, columnspan=3,
                                               color_spectrograma='w', color_waveplot='r')
            self.grafico_audioL.connect()  # Para poder mover el grafico en el eje y con el ratón

            # Audio R
            self.grafico_audioR = AudioGraphic(self.audioR, self.tiempo_max, self.data_frame, row=9, column=4, columnspan=3,
                                               color_spectrograma='w', color_waveplot='r')
            self.grafico_audioR.connect()  # Para poder mover el grafico en el eje y con el ratón

            # Datos
            # Adaptar el tiempo de la barra de tiempo a los datos del archivo de datos
            self.time_slider['from_'] = 0
            self.time_slider['to'] = self.tiempo_max
            self.time_slider['tickinterval'] = self.tiempo_max
            self.label_desde["text"] = "Desde - " + str(0.0)
            self.label_hasta["text"] = "Hasta - " + str(round(self.tiempo_max, 2))

            # Crear graficos de datos
            # Temperatura extrusor
            try:
                self.grafico_extrusor = Graphic(self.datos,
                                                self.tiempo_var,
                                                self.extrusor_var,
                                                self.tiempo_max,
                                                self.data_frame, 13, 4, 3, 'r')
            except KeyError:
                logger.log(logging.WARNING,
                           "No existen datos para la temperatura del extrusor. \nNo se ha creado el gráfico.")

            # Temperatura bed
            try:
                self.grafico_bed = Graphic(self.datos,
                                           self.tiempo_var,
                                           self.bed_var,
                                           self.tiempo_max,
                                           self.data_frame, 14, 4, 3, 'r')
            except KeyError:
                logger.log(logging.WARNING,
                           "No existen datos para la temperatura de la plataforma. \nNo se ha creado el gráfico.")

            # Ventilador
            try:
                self.grafico_vent = Graphic(self.datos,
                                            self.tiempo_var,
                                            self.vent_var,
                                            self.tiempo_max,
                                            self.data_frame, 15, 4, 3, 'r')
            except KeyError:
                logger.log(logging.WARNING,
                           "No existen datos para el ventilador. \nNo se ha creado el gráfico.")

            # Eje X
            try:
                self.grafico_x = Graphic(self.datos,
                                         self.tiempo_var,
                                         self.x_var,
                                         self.tiempo_max,
                                         self.data_frame, 16, 4, 3, 'r')
            except KeyError:
                logger.log(logging.WARNING,
                           "No existen datos para el eje X. \nNo se ha creado el gráfico.")

            # Eje Y
            try:
                self.grafico_y = Graphic(self.datos,
                                         self.tiempo_var,
                                         self.y_var,
                                         self.tiempo_max,
                                         self.data_frame, 17, 4, 3, 'r')
            except KeyError:
                logger.log(logging.WARNING,
                           "No existen datos para el eje Y. \nNo se ha creado el gráfico.")

            # Eje Z
            try:
                self.grafico_z = Graphic(self.datos,
                                         self.tiempo_var,
                                         self.z_var,
                                         self.tiempo_max,
                                         self.data_frame, 18, 4, 3, 'r')
            except KeyError:
                logger.log(logging.WARNING,
                           "No existen datos para el eje Z. \nNo se ha creado el gráfico.")

            # Insertar los datos de incidencias en la tabla de incidencias
            try:
                self.tabla_incidencias.load_content_csv(self.directory_place, 'incidencias.csv')
            except FileNotFoundError:
                logger.log(logging.WARNING, "Archivo de incidencias inexistente. \nSe creará al grabar la primera incidencia.")
                self.tabla_incidencias.reset()

            # Reset zoom por si se ha usado la app antes con otros archivos
            self.zoom_slider.config(state='active')
            self.zoom_slider.set(0)
            self.zoom_slider.config(state='disabled')

            # Reset de la barra de tiempo por si se ha usado la app antes
            self.time_slider.set(0)

            # Restablecer el botón en su estado normal
            self.boton_cargar["state"] = "normal"

            # Log
            logger.log(logging.INFO, 'Archivos cargados.')

            # Quitar la ventana de espera
            self.ventana_espera.destroy()

            # Actualizar la region de scroll para las scrollbars
            self.scroll_canvas.create_window(0, 0, anchor='nw', window=self.data_frame)
            self.scroll_canvas.update_idletasks()
            self.scroll_canvas.config(scrollregion=self.scroll_canvas.bbox('all'))

        # Mostrar una ventana de espera mientras carguen los archivos
        self.ventana_espera = Toplevel()
        self.ventana_espera.title = "Cargando archivos..."
        self.ventana_espera.geometry('800x100')
        self.ventana_espera.attributes('-topmost', True)  # Force window to stay over the others
        if platform.system() == 'Windows':
            self.ventana_espera.wm_iconbitmap('recursos/icon3.ico')  # Icono de la aplicacion
        elif platform.system() == 'Linux':
            self.ventana_espera.wm_iconbitmap('@recursos/icon3.xbm')  # Icono de la aplicacion

        self.label_espera = Label(self.ventana_espera,
                                  text="Cargando archivos, por favor, espere... \n"
                                       "La aplicación no responderá mientras cargan los archivos, es un proceso normal "
                                       "que puede durar varios minutos en caso de grandes archivos. \n"
                                       "Por favor, no cerrar la ventana principal hasta el final del proceso")
        self.label_espera.grid(row=0, column=0, padx=25, pady=20)

        # Lanzar la carga de los archivos y la generacion de graficos despues de tener la ventana de espera activa
        root.after(50, cargar_archivos_y_generar_graficos)

    # Funcion para usar los botones de zoom (negativo y positivo)
    def zoom_buttons(self, zoom):
        # Desactivar botones hasta el final del zoom
        self.zoom_less_but.config(state='disabled')
        self.zoom_more_but.config(state='disabled')
        self.reset_zoom_but.config(state='disabled')
        # Recuperar el zoom
        actual_zoom = self.zoom_slider.get()
        # Fijar limites de zoom (0 y 100):
        if actual_zoom == 0 and zoom < 0 or actual_zoom == 100 and zoom > 0:
            return
        else:
            self.zoom_slider.config(state='active')
            self.zoom_slider.set(actual_zoom + zoom)
            self.zoom_slider.config(state='disabled')
            self.aplicar_zoom(actual_zoom + zoom)
        # Volver a activar los botones
        self.zoom_less_but.config(state='active')
        self.zoom_more_but.config(state='active')
        self.reset_zoom_but.config(state='active')

    # Funcion para usar el boton de reset de zoom
    def reset_zoom(self):
        self.zoom_slider.config(state='active')
        self.zoom_slider.set(0)
        self.zoom_slider.config(state='disabled')
        self.aplicar_zoom(0)

    # Funcion para aplicar un zoom determinado a los gráficos
    def aplicar_zoom(self, zoom):
        self.zoom = int(zoom)
        tiempo_actual = self.time_slider.get()

        try:
            self.grafico_audioR.zoom_graph(self.zoom, tiempo_actual)
        except AttributeError:
            if self.directory_place != "":
                logger.log(logging.ERROR, "Error-Zoom en Audio R no realizado")

        try:
            self.grafico_audioL.zoom_graph(self.zoom, tiempo_actual)
        except AttributeError:
            if self.directory_place != "":
                logger.log(logging.ERROR, "Error-Zoom en Audio L no realizado")

        try:
            self.grafico_extrusor.zoom_graph(self.zoom, tiempo_actual, extrapolive(tiempo_actual,
                                                                                  self.extrusor_var[1],
                                                                                  self.datos,
                                                                                  self.tiempo_var[1]))
        except AttributeError:
            if self.directory_place != "":
                logger.log(logging.ERROR, "Error-Zoom en Extrusor no realizado")
        try:
            self.grafico_bed.zoom_graph(self.zoom, tiempo_actual, extrapolive(tiempo_actual,
                                                                             self.bed_var[1],
                                                                             self.datos,
                                                                             self.tiempo_var[1]))
        except AttributeError:
            if self.directory_place != "":
                logger.log(logging.ERROR, "Error-Zoom en Plataforma no realizado")
        try:
            self.grafico_vent.zoom_graph(self.zoom, tiempo_actual, extrapolive(tiempo_actual,
                                                                              self.vent_var[1],
                                                                              self.datos,
                                                                              self.tiempo_var[1]))
        except AttributeError:
            if self.directory_place != "":
                logger.log(logging.ERROR, "Error-Zoom en Ventilador no realizado")
        try:
            self.grafico_x.zoom_graph(self.zoom, tiempo_actual, extrapolive(tiempo_actual,
                                                                           self.x_var[1],
                                                                           self.datos,
                                                                           self.tiempo_var[1]))
        except AttributeError:
            if self.directory_place != "":
                logger.log(logging.ERROR, "Error-Zoom en eje X no realizado")
        try:
            self.grafico_y.zoom_graph(self.zoom, tiempo_actual, extrapolive(tiempo_actual,
                                                                           self.y_var[1],
                                                                           self.datos,
                                                                           self.tiempo_var[1]))
        except AttributeError:
            if self.directory_place != "":
                logger.log(logging.ERROR, "Error-Zoom en eje Y no realizado")
        try:
            self.grafico_z.zoom_graph(self.zoom, tiempo_actual, extrapolive(tiempo_actual,
                                                                           self.z_var[1],
                                                                           self.datos,
                                                                           self.tiempo_var[1]))
        except AttributeError:
            if self.directory_place != "":
                logger.log(logging.ERROR, "Error-Zoom en eje Z no realizado")

    # Funcion para recuperar el tiempo de la barra de navegacion en el tiempo y mover los diferentes elementos
    def mover_tiempo(self, tiempo):
        self.tiempo = float(tiempo)
        zoom_actual = self.zoom_slider.get()

        try:
            index_datos = min(np.searchsorted(self.datos[self.tiempo_var[1]], self.tiempo),
                              len(self.datos[self.tiempo_var[1]]) - 1)
            self.tiempo_datos = self.datos[self.tiempo_var[1]][index_datos]
        except AttributeError:
            index_datos, self.tiempo_datos = 0, self.tiempo

        self.label_tiempo_valor["text"] = self.tiempo_datos
        self.boton_puntual["text"] = "Puntual - " + str(self.tiempo)
        self.boton_desde["text"] = "Desde - " + str(self.tiempo)
        self.boton_hasta["text"] = "Hasta - " + str(self.tiempo)
        if self.boton_DH_desde['relief'] == 'raised':
            self.boton_DH_desde["text"] = "Desde - " + str(self.tiempo)
        if self.boton_DH_hasta['relief'] == 'raised':
            self.boton_DH_hasta["text"] = "Hasta - " + str(self.tiempo)
        try:
            self.grafico_extrusor.cursor_move(zoom_actual, self.tiempo)
            self.etiqueta_temp_E0_value.config(text=extrapolive(self.tiempo,
                                                                self.extrusor_var[1],
                                                                self.datos,
                                                                self.tiempo_var[1]))
        except (KeyError, AttributeError):
            self.etiqueta_temp_E0_value.config(text="Sin información")

        try:
            self.grafico_bed.cursor_move(zoom_actual, self.tiempo)
            self.etiqueta_temp_bed_value.config(text=extrapolive(self.tiempo,
                                                                 self.bed_var[1],
                                                                 self.datos,
                                                                 self.tiempo_var[1]))
        except (KeyError, AttributeError):
            self.etiqueta_temp_bed_value.config(text="Sin información")
        try:
            self.grafico_vent.cursor_move(zoom_actual, self.tiempo)
            self.etiqueta_vent_value.config(text=extrapolive(self.tiempo,
                                                             self.vent_var[1],
                                                             self.datos,
                                                             self.tiempo_var[1]))
        except (KeyError, AttributeError):
            self.etiqueta_vent_value.config(text="Sin información")
        try:
            self.grafico_x.cursor_move(zoom_actual, self.tiempo)
            self.etiqueta_X_value.config(text=extrapolive(self.tiempo,
                                                          self.x_var[1],
                                                          self.datos,
                                                          self.tiempo_var[1]))
        except (KeyError, AttributeError):
            self.etiqueta_X_value.config(text="Sin información")
        try:
            self.grafico_y.cursor_move(zoom_actual, self.tiempo)
            self.etiqueta_Y_value.config(text=extrapolive(self.tiempo,
                                                          self.y_var[1],
                                                          self.datos,
                                                          self.tiempo_var[1]))
        except (KeyError, AttributeError):
            self.etiqueta_Y_value.config(text="Sin información")
        try:
            self.grafico_z.cursor_move(zoom_actual, self.tiempo)
            self.etiqueta_Z_value.config(text=extrapolive(self.tiempo,
                                                          self.z_var[1],
                                                          self.datos,
                                                          self.tiempo_var[1]))
        except (KeyError, AttributeError):
            self.etiqueta_Z_value.config(text="Sin información")

        # Movimiento de cursor en los audios
        try:
            self.grafico_audioL.cursor_move(zoom_actual, self.tiempo)
        except AttributeError:
            pass

        try:
            self.grafico_audioR.cursor_move(zoom_actual, self.tiempo)
        except AttributeError:
            pass

        # Movimiento de cursor en los videos
        try:
            self.videoL_viewer.on_spot_frame(self.tiempo)
        except AttributeError:
            pass
        try:
            self.videoR_viewer.on_spot_frame(self.tiempo)
        except AttributeError:
            pass

    # Funcion para recuperar los tipos de incidencias
    def recuperar_tipos_incidencia(self):
        with open("recursos/tipo_incidencias.txt") as file:
            self.tipos_incidencias = file.readlines()
        # Limpieza de datos de la lista y se añade el tipo Otros
        self.tipos_incidencias = [re.sub(r"[^ a-zA-Z0-9]", "", tipo) for tipo in self.tipos_incidencias]
        self.tipos_incidencias.append("Otros")
        self.boton_seleccion_incidencias["values"] = self.tipos_incidencias

    # Funcion para guardar o cancelar los cambios en la ventana de edicion de tipos de incidencias
    def cancelar_guardar_cambios_tipos(self, opcion, tipos_incidencias=[]):
        if opcion == 0:  # No se guardan los cambios
            self.ventana_editar_tipos_incidencia.destroy()  # Cerrar ventana de edicion
            logger.log(logging.INFO, 'Cambios no guardados')
        elif opcion == 1:
            try:
                tipos_incidencias.remove('Otros')
            except ValueError:
                pass
            tipos_incidencias = [tipo + "\n" for tipo in tipos_incidencias]
            file = open("recursos/tipo_incidencias.txt", mode='w')
            for tipo in tipos_incidencias:
                file.write(tipo)
            file.close()
            self.ventana_editar_tipos_incidencia.destroy()  # Cerrar ventana de edicion
            logger.log(logging.INFO, 'Tipos de incidencias actualizados')
            self.recuperar_tipos_incidencia()  # Actualizar lista de tipos de incidencias

    # Funcion para mostrar una ventana de edicion para los tipos de incidencias
    def edit_tipos_incidencia(self):
        self.ventana_editar_tipos_incidencia = Toplevel()
        self.ventana_editar_tipos_incidencia.title = "Editar tipos de incidencias"
        self.ventana_editar_tipos_incidencia.resizable(1, 1)
        if platform.system() == 'Windows':
            self.ventana_editar_tipos_incidencia.wm_iconbitmap('recursos/icon3.ico')
        elif platform.system() == 'Linux':
            self.ventana_editar_tipos_incidencia.wm_iconbitmap('@recursos/icon3.xbm')

        # Titulo (Label)
        self.label_edit_tipos_incidencia = Label(self.ventana_editar_tipos_incidencia,
                                                 text="Edicion de tipos de incidencias")
        self.label_edit_tipos_incidencia.grid(row=0, column=0, columnspan=4)

        # Label para dejar espacio
        self.space0 = Label(self.ventana_editar_tipos_incidencia, text='')
        self.space0.grid(row=1, column=0, sticky=N + S)

        # Treeview (Lista de los elementos presentes
        columnas_tipos_incidencias = [('Tipo', 'Tipo de incidencia', 200)]
        self.tabla_tipos_incidencias = Tabla(self.ventana_editar_tipos_incidencia,
                                             height=10,
                                             columns_data=columnas_tipos_incidencias,
                                             row=2,
                                             column=0,
                                             columnspan=4)

        # Cargar los tipos de incidencias
        self.tabla_tipos_incidencias.load_content_txt("recursos/tipo_incidencias.txt")

        # Label para dejar espacio
        self.space1 = Label(self.ventana_editar_tipos_incidencia, text='')
        self.space1.grid(row=3, column=0, sticky=N + S)

        # Boton eliminar
        self.boton_eliminar_tipo_incidencia = ttk.Button(self.ventana_editar_tipos_incidencia)
        self.boton_eliminar_tipo_incidencia['text'] = "Eliminar tipo de incidencia seleccionado"
        self.boton_eliminar_tipo_incidencia['command'] = lambda: self.tabla_tipos_incidencias.del_line_txt()
        self.boton_eliminar_tipo_incidencia.grid(row=4, column=0, columnspan=4, sticky=W + E)

        # Label para dejar espacio
        self.space2 = Label(self.ventana_editar_tipos_incidencia, text='')
        self.space2.grid(row=5, column=0, sticky=N + S)

        # Label + Entry box + boton añadir
        self.label_nueva_incidencia = Label(self.ventana_editar_tipos_incidencia, text='Nuevo tipo: ')
        self.label_nueva_incidencia.grid(row=6, column=0)

        self.entrada_nueva_incidencia = Entry(self.ventana_editar_tipos_incidencia)
        self.entrada_nueva_incidencia.grid(row=6, column=1, columnspan=2)

        self.boton_anadir_nueva_incidencia = ttk.Button(self.ventana_editar_tipos_incidencia)
        self.boton_anadir_nueva_incidencia['text'] = 'Añadir'
        self.boton_anadir_nueva_incidencia['command'] = lambda: self.tabla_tipos_incidencias.add_line_txt(
            self.entrada_nueva_incidencia.get())
        self.boton_anadir_nueva_incidencia.grid(row=6, column=3)

        # Label para dejar espacio
        self.space3 = Label(self.ventana_editar_tipos_incidencia, text='')
        self.space3.grid(row=7, column=0, sticky=N + S)

        # Boton guardar cambios y cerrar ventana
        self.boton_tipos_incidencias_guardar = ttk.Button(self.ventana_editar_tipos_incidencia)
        self.boton_tipos_incidencias_guardar['text'] = 'Guardar cambios'
        self.boton_tipos_incidencias_guardar.grid(row=8, column=0, columnspan=2, sticky=W + E)
        self.boton_tipos_incidencias_guardar['command'] = lambda: self.cancelar_guardar_cambios_tipos(1, self.tabla_tipos_incidencias.tipos_incidencias)

        # Boton cancelar y cerrar ventana
        self.boton_tipos_incidencias_cancelar = ttk.Button(self.ventana_editar_tipos_incidencia)
        self.boton_tipos_incidencias_cancelar['text'] = "Cancelar"
        self.boton_tipos_incidencias_cancelar.grid(row=8, column=2, columnspan=2, sticky=W + E)
        self.boton_tipos_incidencias_cancelar['command'] = lambda: self.cancelar_guardar_cambios_tipos(0)

    # Funcion para modificar el tiempo en la barra de tiempo
    def slight_change_time_slider(self, tiempo):
        if self.time_slider.get() + tiempo >= self.tiempo_max:
            self.time_slider.set(self.tiempo_max)
            self.play_pause()
        else:
            self.time_slider.set(self.time_slider.get() + tiempo)

    # Funcion para poder leer (play) de manera continua. En realidad esa funcion repite el proceso elegido
    def run_continuously(self, interval=1):
        """Continuously run, while executing pending jobs at each
        elapsed time interval.
        @return cease_continuous_run: threading. Event which can
        be set to cease continuous run. Please note that it is
        *intended behavior that run_continuously() does not run
        missed jobs*. For example, if you've registered a job that
        should run every minute and you set a continuous run
        interval of one hour then your job won't be run 60 times
        at each interval but only once.
        """
        cease_continuous_run = threading.Event()

        class ScheduleThread(threading.Thread):
            @classmethod
            def run(cls):
                while not cease_continuous_run.is_set():
                    schedule.run_pending()
                    time.sleep(interval)

        continuous_thread = ScheduleThread()
        continuous_thread.start()
        return cease_continuous_run

    # Funcion que permite leer o parar de leer los videos, audios, datos
    def play_pause(self):
        if not self.has_job:
            schedule.every(self.second_per_frame).seconds.do(self.slight_change_time_slider,
                                                             tiempo=self.second_per_frame)
            self.has_job = 1

        if not self.is_playing:
            self.is_playing = 1
            # Start the background thread
            self.stop_run_continuously = self.run_continuously(interval=self.second_per_frame)
        else:
            self.is_playing = 0
            # Stop the background thread
            self.stop_run_continuously.set()


    #### ---- CONSTRUCTOR ---- ####
    def __init__(self, root):
        super().__init__(root)
        self.ventana = root
        self.ventana.geometry(str(self.application_dimensions[0])+'x'+str(self.application_dimensions[1]))  # Tamaño de la ventana de la aplicación
        self.ventana.title('App Mantenimiento Predictivo Impresora 3D')  # Título de la ventana
        if platform.system() == 'Windows':
            self.ventana.wm_iconbitmap('recursos/icon3.ico')  # Icono de la aplicacion para Windows
        elif platform.system() == 'Linux':
            self.ventana.wm_iconbitmap('@recursos/icon3.xbm')  # Icono de la aplicacion para Linux
        self.ventana.resizable(1, 1)  # Para poder redimensionar la ventana
        self.tiempo = 0  # Al cargar el archivo sin mover el cursor, permite grabar una incidencia con el tiempo 0

        # Canvas de la izquierda
        self.canvas_izquierda = Canvas(self.ventana)
        self.canvas_izquierda.grid(row=0, column=0, sticky=N+S+E+W)

        #### ---- ARCHIVOS ---- ####
        # Contenedor para cargar los archivos
        self.file_frame = LabelFrame(self.canvas_izquierda, text="Selección de archivos")
        self.file_frame.grid(row=0, column=0)  # columnspan=3
        # Label para avisar de la carga de los archivos
        self.label_carga_archivos = Label(self.file_frame)
        self.label_carga_archivos['text'] = "La carga de los archivos puede tardar algunos instantes " \
                                            "\nen los que la applicación se bloqueara de manera temporal."
        self.label_carga_archivos.grid(row=0, column=0, columnspan=7, sticky=W + E)
        # Boton para cargar los archivos
        self.boton_cargar = Button(self.file_frame,
                                   text="Cargar archivos",
                                   command=self.recuperar_carpeta)
        self.boton_cargar.grid(row=1, column=0, columnspan=7, sticky=W + E)
        # Label ubicacion de los archivos
        self.etiqueta_ubicacion_archivos = Label(self.file_frame, text='')
        self.etiqueta_ubicacion_archivos.grid(row=2, column=0, columnspan=7)
        # Label Cantidad de archivos
        self.etiqueta_cantidad_archivos = Label(self.file_frame, text='Archivos: ')
        self.etiqueta_cantidad_archivos.grid(row=5, column=0, columnspan=7, sticky=W + E)
        # Label Numero de videos
        self.etiqueta_cantidad_videos = Label(self.file_frame, text='Video: ')
        self.etiqueta_cantidad_videos.grid(row=6, column=0)
        # Label Numero de audios
        self.etiqueta_cantidad_audios = Label(self.file_frame, text='Audio: ')
        self.etiqueta_cantidad_audios.grid(row=6, column=2)
        # Label Numero de datos
        self.etiqueta_cantidad_datos = Label(self.file_frame, text='Datos: ')
        self.etiqueta_cantidad_datos.grid(row=6, column=4)
        # Label Numero de archivos de incidencias
        self.etiqueta_cantidad_incidencias = Label(self.file_frame, text="Incidencias: ")
        self.etiqueta_cantidad_incidencias.grid(row=6, column=6)


        #### ---- VIDEOS, AUDIOS Y DATOS ---- ####
        # Dimensiones del contenedor principal
        self.right_container_width = self.application_dimensions[0]-495
        self.right_container_heigth = self.application_dimensions[1]-35

        # Contenedor principal
        self.right_frame = Frame(self.ventana, width=self.right_container_width, height=self.right_container_heigth)
        self.right_frame.grid(row=0, column=2, rowspan=4)

        # Contenedor para canvas con scrollbar
        self.scroll_canvas = Canvas(self.right_frame,
                                    width=self.right_container_width,
                                    height=self.right_container_heigth)

        # Scrollbars
        self.hbar = Scrollbar(self.right_frame, orient=HORIZONTAL)
        self.hbar.grid(row=0, column=1, sticky=W+E)
        self.hbar.config(command=self.scroll_canvas.xview)
        self.vbar = Scrollbar(self.right_frame, orient=VERTICAL)
        self.vbar.grid(row=1, column=0, sticky=N+S)
        self.vbar.config(command=self.scroll_canvas.yview)

        # Contenedor para visualizar los videos, audios y datos
        self.data_frame = LabelFrame(self.scroll_canvas,
                                     text="Visualización de datos",
                                     width=self.right_container_width,
                                     height=self.right_container_heigth)
        self.data_frame.grid(row=0, column=0)
        # self.data_frame = LabelFrame(self.ventana, text="Visualización de datos")
        # self.data_frame.grid(row=0, column=2, rowspan=4)

        # Boton -0.01 para el tiempo
        self.time_less_001_but = Button(self.data_frame, text=' - 0.01')
        self.time_less_001_but.grid(row=1, column=0, sticky=W + E, pady=(15,0))
        self.time_less_001_but.config(command=lambda: self.slight_change_time_slider(-0.01))
        self.time_less_001_but.config(width=10)

        # Boton -0.10 para el tiempo
        self.time_less_010_but = Button(self.data_frame, text=' - 0.10')
        self.time_less_010_but.grid(row=1, column=1, sticky=W + E, pady=(15,0))
        self.time_less_010_but.config(command=lambda: self.slight_change_time_slider(-0.1))
        self.time_less_010_but.config(width=10)

        # Boton de lectura - pausa
        if platform.system() == 'Windows':
            self.img1 = Image.open('recursos/play_pause.ico')
        elif platform.system() == 'Linux':
            self.img1 = Image.open('@recursos/play_pause.xbm')
        self.img1 = self.img1.resize((20, 20))
        self.img1 = ImageTk.PhotoImage(self.img1)
        self.play_pause_but = Button(self.data_frame, image=self.img1)
        self.play_pause_but.grid(row=1, column=2, pady=(15,0))
        self.play_pause_but.config(command=self.play_pause)

        # Barra de navegacion en el tiempo
        self.time_slider = Scale(self.data_frame,
                                 label="Tiempo",
                                 from_=0,
                                 to=100,
                                 length=800,
                                 orient=HORIZONTAL,
                                 showvalue=1,
                                 command=self.mover_tiempo,
                                 tickinterval=100,  # Intervalos del índice (abajo)
                                 resolution=0.01)  # Rango del número (decimal con 2 cifras: 0.01)
        self.time_slider.grid(row=1, column=3, columnspan=6, padx=(25, 0))

        # Boton +0.10 para el tiempo
        self.time_more_010_but = Button(self.data_frame, text=' + 0.10')
        self.time_more_010_but.grid(row=1, column=9, sticky=W + E, pady=(15,0))
        self.time_more_010_but.config(command=lambda: self.time_slider.set(self.time_slider.get()+0.1))
        self.time_more_010_but.config(width=10)

        # Boton +0.01 para el tiempo
        self.time_more_001_but = Button(self.data_frame, text=' + 0.01')
        self.time_more_001_but.grid(row=1, column=10, sticky=W + E, pady=(15,0))
        self.time_more_001_but.config(command=lambda: self.slight_change_time_slider(0.01))
        self.time_more_001_but.config(width=10)

        # Label para dejar espacio
        self.space5 = Label(self.data_frame, text='')
        self.space5.grid(row=2, column=0, columnspan=5, sticky=W + E)

        # Video L
        self.etiqueta_videoL = Label(self.data_frame, text='Video L')
        self.etiqueta_videoL.grid(row=3, column=4)
        self.label_videoL = Label(self.data_frame)
        self.label_videoL.grid(row=4, column=4)
        # Video R
        self.etiqueta_videoR = Label(self.data_frame, text='Video R')
        self.etiqueta_videoR.grid(row=3, column=6)
        self.label_videoR = Label(self.data_frame)
        self.label_videoR.grid(row=4, column=6)

        # Label para dejar espacio
        self.space6 = Label(self.data_frame, text='')
        self.space6.grid(row=5, column=0, columnspan=5, sticky=W + E)

        # Label audios
        self.etiqueta_audios = Label(self.data_frame, text='Audios')
        self.etiqueta_audios.grid(row=6, column=5)
        # Label Audio L
        self.etiqueta_audioL = Label(self.data_frame, text='Audio L')
        self.etiqueta_audioL.grid(row=7, column=0, columnspan=2, rowspan=2)
        # Label Audio R
        self.etiqueta_audioR = Label(self.data_frame, text='Audio R')
        self.etiqueta_audioR.grid(row=9, column=0, columnspan=2, rowspan=2)

        # Label para dejar espacio
        self.space7 = Label(self.data_frame, text='')
        self.space7.grid(row=11, column=0, columnspan=5, sticky=W + E)

        # Label datos
        self.etiqueta_datos = Label(self.data_frame, text='Datos')
        self.etiqueta_datos.grid(row=12, column=5)
        # Label de Tiempo
        self.label_tiempo = Label(self.data_frame, text='Tiempo CSV:')
        self.label_tiempo.grid(row=11, column=9, columnspan=2)
        # Valor Tiempo
        self.label_tiempo_valor = Label(self.data_frame, text=0)
        self.label_tiempo_valor.grid(row=12, column=9, columnspan=2)
        # Label Datos - Temp E0
        self.etiqueta_temp_E0 = Label(self.data_frame, text=self.extrusor_var[0])
        self.etiqueta_temp_E0.grid(row=13, column=0, columnspan=2)
        self.etiqueta_temp_E0_value = Label(self.data_frame, text='0')
        self.etiqueta_temp_E0_value.grid(row=13, column=9, columnspan=2)
        # Label Datos - Temp Bed
        self.etiqueta_temp_bed = Label(self.data_frame, text=self.bed_var[0])
        self.etiqueta_temp_bed.grid(row=14, column=0, columnspan=2)
        self.etiqueta_temp_bed_value = Label(self.data_frame, text='0')
        self.etiqueta_temp_bed_value.grid(row=14, column=9, columnspan=2)
        # Label Datos - Vent
        self.etiqueta_vent = Label(self.data_frame, text=self.vent_var[0])
        self.etiqueta_vent.grid(row=15, column=0, columnspan=2)
        self.etiqueta_vent_value = Label(self.data_frame, text='0')
        self.etiqueta_vent_value.grid(row=15, column=9, columnspan=2)
        # Label Datos - X
        self.etiqueta_X = Label(self.data_frame, text=self.x_var[0])
        self.etiqueta_X.grid(row=16, column=0, columnspan=2)
        self.etiqueta_X_value = Label(self.data_frame, text='0')
        self.etiqueta_X_value.grid(row=16, column=9, columnspan=2)
        # Label Datos - Y
        self.etiqueta_Y = Label(self.data_frame, text=self.y_var[0])
        self.etiqueta_Y.grid(row=17, column=0, columnspan=2)
        self.etiqueta_Y_value = Label(self.data_frame, text='0')
        self.etiqueta_Y_value.grid(row=17, column=9, columnspan=2)
        # Label Datos - Z
        self.etiqueta_Z = Label(self.data_frame, text=self.z_var[0])
        self.etiqueta_Z.grid(row=18, column=0, columnspan=2)
        self.etiqueta_Z_value = Label(self.data_frame, text='0')
        self.etiqueta_Z_value.grid(row=18, column=9, columnspan=2)

        # Label para dejar espacio
        self.space8 = Label(self.data_frame, text='')
        self.space8.grid(row=19, column=0, columnspan=5, sticky=W + E)

        # Boton de reset para el zoom
        self.reset_zoom_but = Button(self.data_frame, text='Reset')
        self.reset_zoom_but.grid(row=20, column=0, sticky=W + E, pady=(15,0))
        self.reset_zoom_but.config(command=self.reset_zoom)

        # Boton - para el zoom
        self.zoom_less_but = Button(self.data_frame, text=' - ')
        self.zoom_less_but.grid(row=20, column=1, sticky=W+E, pady=(15,0))
        self.zoom_less_but.config(command=lambda: self.zoom_buttons(-10))

        # Barra de zoom
        self.zoom_slider = Scale(self.data_frame,
                                 label="Zoom",
                                 from_=0,
                                 to=100,
                                 length=800,
                                 orient=HORIZONTAL,
                                 showvalue=1,
                                 tickinterval=50,  # Intervalos del índice (abajo)
                                 resolution=10)  # Rango del número (decimal con 2 cifras: 0.01)
        self.zoom_slider.grid(row=20, column=3, columnspan=6, padx=(25, 0))
        self.zoom_slider.config(state='disabled')

        # Boton + para el zoom
        self.zoom_more_but = Button(self.data_frame, text=' + ')
        self.zoom_more_but.grid(row=20, column=9, sticky=W+E, columnspan=2, pady=(15,0))
        self.zoom_more_but.config(command=lambda: self.zoom_buttons(10))

        # Definir parametros del canvas de la scrollbar
        self.scroll_canvas.create_window(0, 0, anchor='nw', window=self.data_frame)
        self.scroll_canvas.update_idletasks()
        self.scroll_canvas.config(width=self.right_container_width, height=self.right_container_heigth)
        self.scroll_canvas.config(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)
        self.scroll_canvas.config(scrollregion=self.scroll_canvas.bbox('all'))
        self.scroll_canvas.grid(row=1, column=1)

        #### ---- CONSOLE ---- ####
        console_frame = LabelFrame(self.canvas_izquierda, text="Log console")
        console_frame.grid(row=1, column=0)
        self.console = ConsoleUi(console_frame)


        #### ---- INCIDENCIAS ---- ####
        # Contenedor para guardar las incidencias detectadas
        incident_frame = LabelFrame(self.canvas_izquierda, text="Grabación de Incidencias")
        incident_frame.grid(row=2, column=0)

        # Label Tipo de incidencia
        self.etiqueta_tipo_incidencia = Label(incident_frame, text="Seleccionar el tipo de incidencia: ")
        self.etiqueta_tipo_incidencia.grid(row=1, column=1, sticky=W + E)

        # ComboBox para el tipo de incidencias
        self.boton_seleccion_incidencias = ttk.Combobox(incident_frame, state='readonly')
        self.recuperar_tipos_incidencia()

        self.boton_seleccion_incidencias.grid(row=1, column=2, sticky=W + E)
        self.boton_seleccion_incidencias.set(self.tipos_incidencias[0])

        # Boton de parametros incidencias
        if platform.system() == 'Windows':
            self.img2 = Image.open('recursos/parametros.ico')
        elif platform.system() == 'Linux':
            self.img2 = Image.open('@recursos/parametros.xbm')
        self.img2 = self.img2.resize((20, 20))
        self.img2 = ImageTk.PhotoImage(self.img2)
        self.boton_parametros_incidencias = Button(incident_frame, image=self.img2)
        self.boton_parametros_incidencias['command'] = lambda: self.edit_tipos_incidencia()
        self.boton_parametros_incidencias.grid(row=1, column=3)

        # Boton de incidencia puntual
        self.boton_puntual = Button(incident_frame,
                                    text='Puntual',
                                    command=lambda: self.tabla_incidencias.add_line_csv(self.tiempo,
                                                                                        self.tiempo,
                                                                                        self.boton_seleccion_incidencias.get()))
        self.boton_puntual.grid(row=2, column=1, columnspan=2, sticky=W + E)
        # Boton de incidencia Desde ...
        self.boton_desde = Button(incident_frame,
                                  text='Desde',
                                  command=lambda: self.tabla_incidencias.add_line_csv(self.tiempo,
                                                                                      self.time_slider['to'],
                                                                                      self.boton_seleccion_incidencias.get()))
        self.boton_desde.grid(row=3, column=1, sticky=W + E)
        # Label hasta ...
        self.label_hasta = Label(incident_frame, text='Hasta')
        self.label_hasta.grid(row=3, column=2, sticky=W + E)
        # Label desde ...
        self.label_desde = Label(incident_frame, text='Desde')
        self.label_desde.grid(row=4, column=1, sticky=W + E)
        # Boton de incidencia Hasta...
        self.boton_hasta = Button(incident_frame,
                                  text='Hasta',
                                  command=lambda: self.tabla_incidencias.add_line_csv(self.time_slider['from'],
                                                                                      self.tiempo,
                                                                                      self.boton_seleccion_incidencias.get()))
        self.boton_hasta.grid(row=4, column=2, sticky=W + E)
        # Boton de incidencia Desde...Hasta (Desde)
        self.boton_DH_desde = Button(incident_frame,
                                     text='Desde',
                                     command=lambda: self.gestion_desde_hasta("desde"))
        self.boton_DH_desde.grid(row=5, column=1, sticky=W + E)
        # Boton de incidencia Desde...Hasta (Hasta)
        self.boton_DH_hasta = Button(incident_frame,
                                     text='Hasta',
                                     command=lambda: self.gestion_desde_hasta("hasta"))
        self.boton_DH_hasta.grid(row=5, column=2, sticky=W + E)

        # Boton eliminar
        self.boton_eliminar = Button(incident_frame,
                                     text='Eliminar incidencia seleccionada',
                                     command=lambda: self.tabla_incidencias.del_line_csv())
        self.boton_eliminar.grid(row=6, column=1, columnspan=2, sticky=W + E)
        # Tabla de las incidencias guardadas
        columnas = [("Creacion", "Fecha/hora de creacion", 150),
                    ("Tiempo1", "Inicio", 50),
                    ("Tiempo2", "Final", 50),
                    ("Tipo", "Tipo de incidencia", 200)
                    ]
        self.tabla_incidencias = Tabla(incident_frame, columnas, height=8, row=7, column=0, columnspan=4, sticky=W + E)

        # Cerrar ventana de manera segura
        self.ventana.protocol('WM_DELETE_WINDOW', self.quit)
        self.ventana.bind('<Control-q>', self.quit)

    def quit(self, *args):
        """ Funcion para cerrar ventana de manera segura"""
        try:
            self.stop_run_continuously.set()  # Parar el play si esta en marcha
        except AttributeError:  # Si no existe, no se ha usado el play, por lo que da error. Saltarlo
            pass
        self.console.cancel_jobs_from_queue()  # Para la console
        self.ventana.destroy()  # Cerrar la ventana
        root.quit()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    root = tk.Tk()  # Crear instancia de ventana principal
    app = Application(root)  # Se envia a la clase Aplicacion el control sobre la ventana principal
    root.mainloop()  # Bucle de aplicacion
