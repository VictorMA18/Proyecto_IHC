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

# Ruta de la fuente personalizada
custom_font_path = "/home/victorma/React-native-apps/HuellitasHogar/assets/fonts/Outfit-Medium.ttf"  # Cambia esto por la ruta a tu fuente TTF

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
        # Registrar la fuente personalizada
        self.add_font("CustomFont", "", custom_font_path, uni=True)

    def header(self):
        self.set_font("CustomFont", "", 12)  # Usar la fuente personalizada en el encabezado
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, "Reporte de Emociones", ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("CustomFont", "", 8)  # Usar la fuente personalizada en el pie de página
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, f"Página {self.page_no()}", align="C")

    def add_emotion_entry(self, emotion, probability, timestamp):
        self.set_font("CustomFont", "", 9)  # Usar la fuente personalizada para cada entrada de emoción
        if emotion == 'angry' :
            self.set_text_color(0, 0, 0) 
        elif emotion == 'sad':
            self.set_text_color(0, 0, 0) 
        elif emotion == 'happy':
            self.set_text_color(0, 0, 0)
        elif emotion == 'surprise':
            self.set_text_color(0, 0, 0)
        else :
            self.set_text_color(0, 0, 0)

        self.cell(0, 10, f"Hora: {timestamp} - Emoción: {emotion} - Probabilidad: {probability:.2f}", ln=True)

    def generate_pdf_report(self):
        self.add_page()
        self.set_font("CustomFont", "", 10)  # Usar la fuente personalizada para la fecha
        self.cell(0, 10, txt=f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
        self.ln(5)
        
        self.set_font("CustomFont", "", 9)  # Usar la fuente personalizada en negrita para el título
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, "Emociones detectadas durante la sesión:", ln=True)
        self.set_text_color(0, 0, 0)  # Resetear el color a negro
        self.ln(3)

        for emotion, probability, timestamp in emotions_record:
            self.add_emotion_entry(emotion, probability, timestamp)

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

        # Registro de emociones
        emotions_record.append((emotion_text, emotion_probability, datetime.now().strftime('%H:%M:%S')))

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
