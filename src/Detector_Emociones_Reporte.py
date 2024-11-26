import tkinter as tk
from tkinter import Label
from customtkinter import *
import customtkinter as Ctk
import os
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
from statistics import mode
from keras.models import load_model
import numpy as np
import time
from datetime import datetime
import signal
import sys
from collections import Counter
from fpdf import FPDF
import matplotlib.pyplot as plt
import subprocess

# Importaciones personalizadas
from utils.datasets import get_labels
from utils.inference import (
    detect_faces,
    draw_text,
    draw_bounding_box,
    apply_offsets,
    load_detection_model
)
from utils.preprocessor import preprocess_input

# Configuración y carga de modelos
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# Carga de modelos
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_target_size = emotion_classifier.input_shape[1:3]

# Hiperparámetros
frame_window = 10
emotion_offsets = (20, 40)

# Variables para almacenar datos
emotion_window = []
emotions_record = []
emotion = [0, 0, 0, 0, 0]  # [Angry, Sad, Happy, Surprise, Neutral]
total_emotions = 0

# Crear el PDF
class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        self.primary_color = (41, 128, 185)    # Azul
        self.secondary_color = (44, 62, 80)    # Gris oscuro
        self.accent_color = (46, 204, 113)     # Verde
        self.light_bg = (236, 240, 241)

    def cover_page(self):
        self.add_page()
        # Fondo superior
        self.set_fill_color(*self.primary_color)
        self.rect(0, 0, 210, 60, 'F')
        
        # Logo o título principal
        self.set_font("Times", "", 32)
        self.set_text_color(255, 255, 255)
        self.set_y(12)
        self.cell(0, 20, "Análisis de Emociones", ln=True, align="C")
        
        # Subtítulo
        self.set_font("Times", "", 16)
        self.cell(0, 10, "Reporte Detallado", ln=True, align="C")
        
        # Información de fecha
        self.set_y(66)
        self.set_text_color(*self.secondary_color)
        self.set_font("Times", "", 12)
        self.cell(0, 10, f"Generado el: {datetime.now().strftime('%d de %B, %Y')}", ln=True, align="C")
        
        # Descripción
        self.set_y(80)
        self.set_font("Times", "", 11)

        self.section_title("Resumen del Reporte: ")
        self.set_font("Times", "", 9)
        self.multi_cell(0, 8, ("Este reporte presenta un análisis comprehensivo de las emociones detectadas "
                              "durante la sesión de observación. Incluye análisis estadísticos, "
                              "visualizaciones y patrones emocionales identificados."), align="C")

    def header(self):
        if self.page_no() != 1:  # No mostrar header en la página de portada
            # Línea superior decorativa
            self.set_fill_color(*self.primary_color)
            self.rect(0, 0, 210, 15, 'F')
            
            # Título de la página
            self.set_y(20)
            self.set_font("Times", "", 12)
            self.set_text_color(*self.secondary_color)
            self.cell(0, 10, "Reporte de Análisis de Emociones", ln=True, align="L")
            
            # Línea separadora
            self.line(10, 33, 200, 33)
            self.ln(15)

    def footer(self):
        self.set_y(-20)
        self.set_font("Times", "", 8)
        self.set_text_color(*self.secondary_color)
        
        # Línea separadora
        self.line(10, 280, 200, 280)
        
        # Número de página
        self.cell(0, 10, f"Página {self.page_no()}", align="C")

    def section_title(self, title):
        self.set_fill_color(*self.primary_color)
        self.set_text_color(255, 255, 255)
        self.set_font("Times", "", 14)
        self.cell(0, 10, f"  {title}", ln=True, fill=True)
        self.ln(5)
        self.set_text_color(*self.secondary_color)

    def add_emotion_entry(self, emotion, probability, timestamp):
        self.set_font("Times", "", 9)
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, f"Hora: {timestamp} - Emoción: {emotion} - Probabilidad: {probability:.2f}", ln=True)

    def add_image(self, image_path, x=10, y=None, w=180):
        self.image(image_path, x=x, y=y, w=w)

    def add_event_highlights_table(self, events):
        self.set_font("Times", "", 10)
        self.cell(0, 10, "Análisis de Eventos Destacados:", ln=True)
        self.ln(5)
        self.set_font("Times", "", 9)

        self.set_fill_color(*self.primary_color)
        # Cabecera de la tabla
        self.set_text_color(255, 255, 255)
        self.cell(40, 10, "Hora", 1, 0, "C", fill=True)
        self.cell(40, 10, "Emoción", 1, 0, "C", fill=True)
        self.cell(40, 10, "Probabilidad", 1, 0, "C", fill=True)
        self.cell(60, 10, "Tipo de Evento", 1, 1, "C", fill=True)

        self.set_fill_color(*self.light_bg)
        self.set_text_color(0, 0, 0)
        # Filas de la tabla
        for event in events:
            self.cell(40, 10, event[0], 1, 0, "C")
            self.cell(40, 10, event[1], 1, 0, "C")
            self.cell(40, 10, f"{event[2]:.2f}", 1, 0, "C")
            self.cell(60, 10, event[3], 1, 1, "C")
        self.ln(10)

    def add_emotion_clusters_table(self, clusters):
        self.set_font("Times", "", 10)
        self.cell(0, 10, "Clústeres de Emociones por Segmento de Tiempo:", ln=True)
        self.ln(5)
        self.set_font("Times", "", 9)

        self.set_fill_color(*self.primary_color)
        # Cabecera de la tabla
        self.set_text_color(255, 255, 255)
        self.cell(60, 10, "Intervalo de Tiempo", 1, 0, "C", fill=True)
        self.cell(60, 10, "Emoción Dominante", 1, 0, "C", fill=True)
        self.cell(60, 10, "Promedio de Probabilidad", 1, 1, "C", fill=True)

        self.set_fill_color(*self.light_bg)
        self.set_text_color(0, 0, 0)
        # Filas de la tabla
        for cluster in clusters:
            self.cell(60, 10, cluster[0], 1, 0, "C")
            self.cell(60, 10, cluster[1], 1, 0, "C")
            self.cell(60, 10, f"{cluster[2]:.2f}", 1, 1, "C")
        self.ln(10)

    def add_recommendations(self):
        self.section_title("Recomendaciones: ")
        self.set_font("Times", "", 9)
        # Analiza las emociones registradas
        negative_emotions = [e for e in emotions_record if e[0] in ['sad', 'angry', 'fear']]
        if len(negative_emotions) > 0:
            self.cell(0, 10, "Se detectaron emociones negativas con frecuencia. Se sugiere:", ln=True)
            self.cell(0, 10, "- Tomar pausas regulares.", ln=True)
            self.cell(0, 10, "- Practicar técnicas de relajación, como respiración profunda.", ln=True)
            self.cell(0, 10, "- Participar en actividades de bienestar, como ejercicio o meditación.", ln=True)
        else:
            self.cell(0, 10, "Las emociones predominantes fueron positivas. Continúe con el mismo enfoque.", ln=True)

        self.ln(10)
    
    def add_summary(self):
        self.set_font("Times", "", 9)
        # Rango de fechas y emociones predominantes
        start_date = emotions_record[0][2] if emotions_record else "N/A"
        end_date = emotions_record[-1][2] if emotions_record else "N/A"
        emotion_counts = Counter([e[0] for e in emotions_record])
        predominant_emotions = emotion_counts.most_common(3)  # Las 3 emociones más comunes

        self.cell(0, 10, f"Rango de fechas: {start_date} - {end_date}", ln=True)
        self.cell(0, 10, "Emociones predominantes:", ln=True)
        for emotion_item, count in predominant_emotions:
            self.cell(0, 10, f"{emotion_item.capitalize()}: {count} veces", ln=True)

        self.ln(3)
    
    def generate_pdf_report(self):
        self.cover_page()
        self.ln(3)

        self.set_font("Times", "", 9)
        self.section_title("Emociones detectadas durante la sesión:")
        self.ln(3)

        for emotion_entry in emotions_record:
            emotion, probability, timestamp = emotion_entry
            self.add_emotion_entry(emotion, probability, timestamp)

        self.add_summary()  # Agrega el resumen al inicio
        # Agregar gráficos y análisis
        generate_emotion_analysis_charts()
        self.add_image("emotion_bars.png")
        self.add_image("emotion_timeline.png")

        events = generate_event_highlights()
        self.add_event_highlights_table(events)

        # Agregar clústeres de emociones
        clusters = generate_emotion_clusters()
        self.add_emotion_clusters_table(clusters)

        self.add_recommendations()  # Agrega las recomendaciones al final

# Función para generar gráficos
def generate_emotion_analysis_charts():
    # Gráfico de barras de frecuencia de emociones
    emotions = [e[0] for e in emotions_record]
    if emotions:
        emotion_counts = Counter(emotions)
        labels, values = zip(*emotion_counts.items())

        plt.figure(figsize=(8, 4))
        plt.bar(labels, values, color="skyblue")
        plt.title("Frecuencia de Emociones")
        plt.xlabel("Emociones")
        plt.ylabel("Frecuencia")
        plt.savefig("emotion_bars.png")
        plt.close()

        # Gráfico de línea temporal de emociones
        timestamps = [e[2] for e in emotions_record]
        probabilities = [e[1] for e in emotions_record]

        plt.figure(figsize=(8, 4))
        plt.plot(timestamps, probabilities, label="Probabilidad", color="purple")
        plt.title("Probabilidad de Emociones a lo Largo del Tiempo")
        plt.xlabel("Tiempo")
        plt.ylabel("Probabilidad")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("emotion_timeline.png")
        plt.close()

# Función para generar eventos destacados
def generate_event_highlights():
    events = []
    for i, (emotion, probability, timestamp) in enumerate(emotions_record):
        if i == 0 or abs(probability - emotions_record[i - 1][1]) > 0.3:
            event_type = "Cambio Significativo" if i > 0 else "Pico de Probabilidad"
            events.append((timestamp, emotion, probability, event_type))
    return events

# Función para generar clústeres de emociones
def generate_emotion_clusters():
    interval = 5 * 60  # Intervalo en segundos (5 minutos)
    clusters = []
    current_cluster = []
    probabilities = []
    
    # Verificar que emotions_record no esté vacío
    if not emotions_record:
        print("No hay datos en emotions_record para generar clústeres.")
        return clusters

    start_time = datetime.strptime(emotions_record[0][2], '%H:%M:%S')
    interval_start_time = start_time.strftime('%H:%M:%S')

    for emotion, probability, timestamp in emotions_record:
        current_time = datetime.strptime(timestamp, '%H:%M:%S')
        if (current_time - start_time).seconds < interval:
            current_cluster.append(emotion)
            probabilities.append(probability)
        else:
            if current_cluster:
                dominant_emotion = mode(current_cluster)
                average_probability = sum(probabilities) / len(probabilities)
                clusters.append((f"{interval_start_time} - {timestamp}", dominant_emotion, average_probability))
            # Reiniciar para el siguiente clúster
            current_cluster = [emotion]
            probabilities = [probability]
            start_time = current_time
            interval_start_time = start_time.strftime('%H:%M:%S')

    # Agregar el último clúster si hay datos restantes
    if current_cluster:
        dominant_emotion = mode(current_cluster)
        average_probability = sum(probabilities) / len(probabilities)
        clusters.append((f"{interval_start_time} - {timestamp}", dominant_emotion, average_probability))
    
    return clusters

# Manejo de salida y generación de reporte
def handle_exit(signal_received, frame):
    print("\nInterrupción detectada. Generando reporte...")
    pdf = PDFReport()
    pdf.generate_pdf_report()
    pdf.output("Reporte_emociones.pdf")
    print("Reporte generado: Reporte_emociones.pdf")
    sys.exit(0)

# Configurar señal para manejar interrupciones (Ctrl+C)
signal.signal(signal.SIGINT, handle_exit)

# Inicialización de Tkinter
root = Ctk.CTk()
Ctk.set_appearance_mode("light")
Ctk.set_default_color_theme("dark-blue")
root.configure(fg_color="#B2DFFF")
root.title("Detector de Emociones")

# Crear un frame para el stream de video
video_frame = tk.Frame(root)
video_frame.pack(side="left")

# Crear una etiqueta para mostrar el stream de video
video_label = Label(video_frame)
video_label.pack()

# Barra superior

def volver_al_inicio():
    # Cerrar la ventana actual
    root.destroy()
    # Ejecutar el script de la ventana inicial
    subprocess.Popen([r"C:\Users\Bryan\Documents\face_classification-master\venv\Scripts\python", 
                      r"C:\Users\Bryan\Documents\Proyecto_IHC\Proyecto_IHC\src\Inicio.py"])

top_frame = Ctk.CTkFrame(root, height=50, fg_color="#7DA4E6")
top_frame.pack(fill="x")
back_button = Ctk.CTkButton(top_frame, text="←", font=("Arial", 20, "bold"), width=40, command=volver_al_inicio)
refresh_button = Ctk.CTkButton(top_frame, text="⟳", font=("Arial", 20, "bold") , width=40, command=lambda: print("Actualizar"))
title_label = Ctk.CTkLabel(top_frame, text="FACIAL", text_color="white", font=("Arial", 20, "bold"))

back_button.pack(side="left", padx=10, pady=5)
refresh_button.pack(side="left", padx=10, pady=5)
title_label.pack(side="right", padx=15)

# Cronómetro
cronometro_frame = Ctk.CTkFrame(root, fg_color="#1E80D1")
cronometro_frame.pack(pady=10, padx=10, fill="x")

cronometro_label = Ctk.CTkLabel(cronometro_frame, text="Tiempo de ejecución: 0 segundos", font=("Arial", 24, "bold"), text_color="#FFFFFF")
cronometro_label.pack(pady=7,padx=40)

# Título "Estadísticas"
stats_label = Ctk.CTkLabel(root, text="ESTADÍSTICAS", font=("Arial", 20, "bold"), text_color="black")
stats_label.pack(pady=7)

# Función para cargar imágenes como CTkImage
def load_ctk_image(path, size=(20, 20)):
    img = Image.open(path)
    return CTkImage(light_image=img, size=size)

# Emojis de emociones (carga de imágenes)
sad_path = "assets/sad.png"
happy_path = "assets/happy.png"
angry_path = "assets/angry.png"
surprise_path = "assets/surprise.png"
neutral_path = "assets/neutral.png"

# Cargar imágenes como CTkImage
tk_sad = load_ctk_image(sad_path)
tk_happy = load_ctk_image(happy_path)
tk_angry = load_ctk_image(angry_path)
tk_surprise = load_ctk_image(surprise_path)
tk_neutral = load_ctk_image(neutral_path)

# Tabla de estadísticas
stats_frame = CTkFrame(root, fg_color="#E6F0FA")
stats_frame.pack(pady=5, padx=20, fill="x")

headers = CTkLabel(stats_frame, text="Emociones       Porcentaje", font=("Arial", 24, "bold"))
headers.pack(pady=10, padx=40, anchor="center")

# Crear etiquetas para mostrar las emociones (sin emojis de texto)
text_label0 = CTkLabel(root, text="Enojo             0%", image=tk_angry, compound="left",  font=("Arial", 20))
text_label0.pack(pady=2)
text_label1 = CTkLabel(root, text="Tristeza          0%", image=tk_sad, compound="left", font=("Arial", 20))
text_label1.pack(pady=2)
text_label2 = CTkLabel(root, text="Alegría           0%", image=tk_happy, compound="left",  font=("Arial", 20))
text_label2.pack(pady=2)
text_label3 = CTkLabel(root, text="Sorpresa          0%", image=tk_surprise, compound="left", font=("Arial", 20))
text_label3.pack(pady=2)
text_label4 = CTkLabel(root, text="Neutral           0%", image=tk_neutral, compound="left", font=("Arial", 20))
text_label4.pack(pady=2)

file_path = os.path.dirname(os.path.realpath(__file__)) 
image_pdf = Ctk.CTkImage(Image.open(file_path + "/assets/download.png"), size=(35,35))


def generar_reporte():
    try:
        pdf = PDFReport()
        
        pdf.generate_pdf_report()
        
        # Guardar el archivo PDF
        nombre_archivo = f"Reporte_Emociones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(nombre_archivo)
        
        # Mostrar confirmación  
        tk.messagebox.showinfo("Reporte Generado", f"Reporte guardado como:\n{nombre_archivo}")
    except Exception as e:
        tk.messagebox.showerror("Error", f"No se pudo generar el reporte:\n{str(e)}")

text_label5 = Label(root, text="Generar Reporte" , background="#B2DFFF", font=("Arial", 24))
text_label5.pack(pady=10)
boton_reporte = Ctk.CTkButton(master=root, 
                                        corner_radius=24,
                                        image=image_pdf,
                                        width=60,
                                        height=50,
                                        fg_color="#FF8000",
                                        text="Descargar", 
                                        font=("Arial", 21, "bold"),
                                        border_spacing=10,
                                        command=generar_reporte)
boton_reporte.pack(padx=5,pady=7)
# Tiempo de inicio
start_time = time.time()

# Función para actualizar el tiempo de ejecución
def update_time():
    elapsed_time = int(time.time() - start_time)
    cronometro_label.configure(text=f"Tiempo de ejecución: {elapsed_time} segundos")
    root.after(1000, update_time)

# Función para actualizar las etiquetas de emociones
def update_emotion_labels():
    global total_emotions
    if total_emotions > 0:
        text_label0.configure(text=f"Enojo             {emotion[0] / total_emotions * 100:.2f}%")
        text_label1.configure(text=f"Tristeza           {emotion[1] / total_emotions * 100:.2f}%")
        text_label2.configure(text=f"Alegría           {emotion[2] / total_emotions * 100:.2f}%")
        text_label3.configure(text=f"Sorpresa            {emotion[3] / total_emotions * 100:.2f}%")
        text_label4.configure(text=f"Neutral           {emotion[4] / total_emotions * 100:.2f}%")
    else:
        text_label0.configure(text="Enojo             0%")
        text_label1.configure(text="Tristeza          0%")
        text_label2.configure(text="Alegría           0%")
        text_label3.configure(text="Sorpresa          0%")
        text_label4.configure(text="Neutral           0%")
    root.after(1000, update_emotion_labels)

# Función para actualizar el frame de video
def update_frame():
    global total_emotions
    ret, bgr_image = video_capture.read()
    if not ret:
        return
    
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # Convertir a RGB antes de mostrar
    faces = detect_faces(face_detection, gray_image)

    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
            emotion[0] += 1
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
            emotion[1] += 1
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
            emotion[2] += 1
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
            emotion[3] += 1
        elif emotion_text == 'neutral':
            color = emotion_probability * np.asarray((120, 120, 120))
            emotion[4] += 1
        else:
            color = emotion_probability * np.asarray((0, 255, 0))
        
        # Actualizar el total de emociones
        total_emotions = np.sum(emotion)
        
        # Registrar emociones cada segundo
        current_time = time.time()
        if current_time - last_record_time[0] >= 1:  # Verifica si ha pasado 1 segundo
            emotions_record.append((emotion_text, emotion_probability, datetime.now().strftime('%H:%M:%S')))
            last_record_time[0] = current_time  # Actualiza el tiempo del último registro

        color = color.astype(int).tolist()
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode, color, 0, -45, 1, 1)

    # Convertir la imagen RGB a formato ImageTk
    img = Image.fromarray(rgb_image)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(10, update_frame)


# Variables para manejar el tiempo de registro
last_record_time = [time.time()]  # Usar lista para permitir modificación dentro de la función

# Iniciar la captura de video
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: No se pudo abrir la cámara.")
    sys.exit()

# Iniciar las actualizaciones
update_frame()
update_time()
update_emotion_labels()

# Función para cerrar la ventana y generar el reporte al cerrar la interfaz
def on_closing():
    print("Cerrando la aplicación. Generando reporte...")
    pdf = PDFReport()
    pdf.generate_pdf_report()
    pdf.output("Reporte_emociones.pdf")
    print("Reporte generado: Reporte_emociones.pdf")
    video_capture.release()
    cv2.destroyAllWindows()
    root.destroy()
    sys.exit()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Ejecutar la interfaz gráfica
root.mainloop()

# Asegurarse de liberar la captura de video al finalizar
video_capture.release()
cv2.destroyAllWindows()