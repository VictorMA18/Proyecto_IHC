from fpdf import FPDF
from statistics import mode
import cv2
from keras.models import load_model
import numpy as np
from datetime import datetime
import signal
import sys
from utils.datasets import get_labels
from utils.inference import detect_faces, draw_text, draw_bounding_box, apply_offsets, load_detection_model
from utils.preprocessor import preprocess_input
import time
import matplotlib.pyplot as plt
from collections import Counter

# Ruta de la fuente personalizada
custom_font_path = "/home/victorma/React-native-apps/HuellitasHogar/assets/fonts/Outfit-Medium.ttf"

# Configuración y carga de modelos
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_target_size = emotion_classifier.input_shape[1:3]

# Hiperparámetros
frame_window = 10
emotion_offsets = (20, 40)
emotion_window = []
emotions_record = []

# Crear el PDF
class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("CustomFont", "", custom_font_path, uni=True)
        self.primary_color = (41, 128, 185)    # Azul
        self.secondary_color = (44, 62, 80)    # Gris oscuro
        self.accent_color = (46, 204, 113)     # Verde
        self.light_bg = (236, 240, 241)

    def cover_page(self):
        self.add_page()
        # Fondo superior
        self.set_fill_color(*self.primary_color)
        self.rect(0, 0, 210, 100, 'F')
        
        # Logo o título principal
        self.set_font("CustomFont", "", 32)
        self.set_text_color(255, 255, 255)
        self.set_y(40)
        self.cell(0, 20, "Análisis de Emociones", ln=True, align="C")
        
        # Subtítulo
        self.set_font("CustomFont", "", 16)
        self.cell(0, 10, "Reporte Detallado", ln=True, align="C")
        
        # Información de fecha
        self.set_y(120)
        self.set_text_color(*self.secondary_color)
        self.set_font("CustomFont", "", 12)
        self.cell(0, 10, f"Generado el: {datetime.now().strftime('%d de %B, %Y')}", ln=True, align="C")
        
        # Descripción
        self.set_y(150)
        self.set_font("CustomFont", "", 11)

        self.section_title("Resumen del Reporte: ")
        self.set_font("CustomFont", "", 9)
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
            self.set_font("CustomFont", "", 12)
            self.set_text_color(*self.secondary_color)
            self.cell(0, 10, "Reporte de Análisis de Emociones", ln=True, align="L")
            
            # Línea separadora
            self.line(10, 33, 200, 33)
            self.ln(15)

    def footer(self):
        self.set_y(-20)
        self.set_font("CustomFont", "", 8)
        self.set_text_color(*self.secondary_color)
        
        # Línea separadora
        self.line(10, 280, 200, 280)
        
        # Número de página
        self.cell(0, 10, f"Página {self.page_no()}", align="C")

    def section_title(self, title):
        self.set_fill_color(*self.primary_color)
        self.set_text_color(255, 255, 255)
        self.set_font("CustomFont", "", 14)
        self.cell(0, 10, f"  {title}", ln=True, fill=True)
        self.ln(5)
        self.set_text_color(*self.secondary_color)


    def add_emotion_entry(self, emotion, probability, timestamp):
        self.set_font("CustomFont", "", 9)
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, f"Hora: {timestamp} - Emoción: {emotion} - Probabilidad: {probability:.2f}", ln=True)

    def add_image(self, image_path, x=10, y=None, w=180):
        self.image(image_path, x=x, y=y, w=w)

    def add_event_highlights_table(self, events):
        self.set_font("CustomFont", "", 10)
        self.cell(0, 10, "Análisis de Eventos Destacados:", ln=True)
        self.ln(5)
        self.set_font("CustomFont", "", 9)

        # Cabecera de la tabla
        self.cell(40, 10, "Hora", 1, 0, "C")
        self.cell(40, 10, "Emoción", 1, 0, "C")
        self.cell(40, 10, "Probabilidad", 1, 0, "C")
        self.cell(60, 10, "Tipo de Evento", 1, 1, "C")

        # Filas de la tabla
        for event in events:
            self.cell(40, 10, event[0], 1, 0, "C")
            self.cell(40, 10, event[1], 1, 0, "C")
            self.cell(40, 10, f"{event[2]:.2f}", 1, 0, "C")
            self.cell(60, 10, event[3], 1, 1, "C")
        self.ln(10)

    def add_emotion_clusters_table(self, clusters):
        self.set_font("CustomFont", "", 10)
        self.cell(0, 10, "Clústeres de Emociones por Segmento de Tiempo:", ln=True)
        self.ln(5)
        self.set_font("CustomFont", "", 9)

        # Cabecera de la tabla
        self.cell(60, 10, "Intervalo de Tiempo", 1, 0, "C")
        self.cell(60, 10, "Emoción Dominante", 1, 0, "C")
        self.cell(60, 10, "Promedio de Probabilidad", 1, 1, "C")

        # Filas de la tabla
        for cluster in clusters:
            self.cell(60, 10, cluster[0], 1, 0, "C")
            self.cell(60, 10, cluster[1], 1, 0, "C")
            self.cell(60, 10, f"{cluster[2]:.2f}", 1, 1, "C")
        self.ln(10)

    def add_recommendations(self):
        self.section_title("Recomendaciones: ")
        self.set_font("CustomFont", "", 9)
        # Analiza las emociones registradas|
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

        self.set_font("CustomFont", "", 9)
        # Rango de fechas y emociones predominantes
        start_date = emotions_record[0][2] if emotions_record else "N/A"
        end_date = emotions_record[-1][2] if emotions_record else "N/A"
        emotion_counts = Counter([e[0] for e in emotions_record])
        predominant_emotions = emotion_counts.most_common(3)  # Las 3 emociones más comunes

        self.cell(0, 10, f"Rango de fechas: {start_date} - {end_date}", ln=True)
        self.cell(0, 10, "Emociones predominantes:", ln=True)
        for emotion, count in predominant_emotions:
            self.cell(0, 10, f"{emotion.capitalize()}: {count} veces", ln=True)

        self.ln(3)
    
    def generate_pdf_report(self):
        self.cover_page()
        self.ln(3)

        self.set_font("CustomFont", "", 9)
        self.section_title("Emociones detectadas durante la sesión:")
        self.ln(3)
        

        for emotion, probability, timestamp in emotions_record:
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

#Grafico
def generate_emotion_analysis_charts():
    # Gráfico de barras de frecuencia de emociones
    emotions = [e[0] for e in emotions_record]
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

def generate_event_highlights():
    # Generar una lista de eventos destacados con el formato (hora, emoción, probabilidad, tipo de evento)
    events = []
    for i, (emotion, probability, timestamp) in enumerate(emotions_record):
        if i == 0 or abs(probability - emotions_record[i - 1][1]) > 0.3:
            event_type = "Cambio Significativo" if i > 0 else "Pico de Probabilidad"
            events.append((timestamp, emotion, probability, event_type))
    return events

def generate_emotion_clusters():
    # Crear clústeres por intervalo de tiempo y calcular la emoción dominante y su probabilidad promedio
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

# Manejo de salida
def handle_exit(signal, frame):
    print("\nInterrupción detectada. Generando reporte...")
    pdf = PDFReport()
    pdf.generate_pdf_report()
    pdf.output("reporte_emociones.pdf")
    print("Reporte generado: reporte_emociones.pdf")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)

# Captura de video
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

# Mantén una variable para almacenar el último tiempo de registro
last_record_time = time.time()

# Ajusta el bucle de captura de video 
# 
# 
while True:
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
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

        # Registrar emociones cada 5 segundos
        current_time = time.time()
        if current_time - last_record_time >= 5:  # Verifica si han pasado 5 segundos
            emotions_record.append((emotion_text, emotion_probability, datetime.now().strftime('%H:%M:%S')))
            last_record_time = current_time  # Actualiza el tiempo del último registro

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)

        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        color = emotion_probability * np.asarray((255, 0, 0)) if emotion_text == 'angry' else \
                emotion_probability * np.asarray((0, 0, 255)) if emotion_text == 'sad' else \
                emotion_probability * np.asarray((255, 255, 0)) if emotion_text == 'happy' else \
                emotion_probability * np.asarray((0, 255, 255)) if emotion_text == 'surprise' else \
                emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int).tolist()
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode, color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

# Generar el reporte si se cierra con 'q'
pdf = PDFReport()
pdf.generate_pdf_report()
pdf.output("reporte_emociones.pdf")
print("Reporte generado: reporte_emociones.pdf")