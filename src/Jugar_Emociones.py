import tkinter as tk
from PIL import Image, ImageTk
from statistics import mode
import cv2
from keras.models import load_model
import numpy as np
from utils.datasets import get_labels
from utils.inference import detect_faces, draw_text, draw_bounding_box, apply_offsets, load_detection_model
from utils.preprocessor import preprocess_input
from pynput.keyboard import Controller

# Inicializar controlador de teclado
keyboard = Controller()

# Parameters for loading data and images
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# Hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# Loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# Getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# Starting lists for calculating modes
emotion_window = []

# Variables para almacenar teclas asignadas a emociones
emotion_keys = {"Felicidad": None, "Tristeza": None, "Enojo": None, "Sorpresa": None, "Neutral": None}

# Método para presionar teclas
def press_repeatedly(key):
    keyboard.press(key)
    keyboard.release(key)

# Inicializar Tkinter
root = tk.Tk()
root.title("Detector de Emociones")
root.attributes('-topmost', True)

# Centrar la ventana en la pantalla verticalmente
window_width, window_height = 880, 412
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
position_top = int(screen_height / 2 - window_height / 2)
position_right = int(screen_width / 2 - window_width / 2)
root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

# Crear un frame principal para dividir el espacio
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Crear un label para el video
video_frame = tk.Frame(main_frame, width=640, height=480)
video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
video_label = tk.Label(video_frame)
video_label.pack(fill=tk.BOTH, expand=True)

# Frame para las imágenes y cuadros de texto en la parte blanca
side_frame = tk.Frame(main_frame, width=320, bg="white")
side_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=10, pady=10)

# Subtítulo "Elegir Teclas"
subtitle_label = tk.Label(side_frame, text="Elegir Teclas", font=("Arial", 16, "bold"), bg="white", fg="black")
subtitle_label.pack(pady=10)

# Lista de emociones e imágenes
emotions = ["Felicidad", "Tristeza", "Enojo", "Sorpresa", "Neutral"]
emoji_paths = ["assets/happy.png", "assets/sad.png", "assets/angry.png", "assets/surprise.png", "assets/neutral.png"]
key_entries = {}

# Crear filas para cada emoción
for emotion, emoji_path in zip(emotions, emoji_paths):
    row_frame = tk.Frame(side_frame, bg="white")
    row_frame.pack(fill=tk.X, pady=5, padx=5)

    # Imagen emoji
    emoji_image = Image.open(emoji_path).resize((30, 30))
    emoji_photo = ImageTk.PhotoImage(emoji_image)
    emoji_label = tk.Label(row_frame, image=emoji_photo, bg="white")
    emoji_label.image = emoji_photo  # Necesario para evitar que la imagen sea recolectada por el GC
    emoji_label.pack(side=tk.LEFT, padx=5)

    # Etiqueta de emoción
    emotion_label = tk.Label(row_frame, text=emotion, font=("Arial", 14), bg="white")
    emotion_label.pack(side=tk.LEFT, padx=10)

    # Cuadro de texto para la tecla
    key_entry = tk.Entry(row_frame, font=("Arial", 14), width=10)
    key_entry.pack(side=tk.RIGHT, padx=10)
    key_entries[emotion] = key_entry

# Función para reiniciar teclas
def reset_keys():
    for emotion in emotions:
        key_entries[emotion].configure(state="normal")  # Habilita el cuadro de texto
        key_entries[emotion].delete(0, tk.END)          # Borra el contenido
        emotion_keys[emotion] = None                   # Reinicia la tecla en el diccionario

# Función para guardar teclas
def save_keys():
    for emotion in emotions:
        key = key_entries[emotion].get()
        if key:
            emotion_keys[emotion] = key
        else:
            emotion_keys[emotion] = None
        key_entries[emotion].configure(state="disabled")
    print("Teclas guardadas:", emotion_keys)

# Botones para reiniciar y guardar teclas
button_frame = tk.Frame(side_frame, bg="white")
button_frame.pack(fill=tk.X, pady=20)

reset_button = tk.Button(button_frame, text="Reiniciar", font=("Arial", 14), command=reset_keys, bg="red", fg="white")
reset_button.pack(side=tk.LEFT, padx=10, pady=5)

save_button = tk.Button(button_frame, text="Guardar", font=("Arial", 14), command=save_keys, bg="green", fg="white")
save_button.pack(side=tk.RIGHT, padx=10, pady=5)

# Función para actualizar el video
def update_frame():
    ret, bgr_image = video_capture.read()
    if not ret:
        return

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

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        emotion_mapping = {
            "happy": "Felicidad",
            "sad": "Tristeza",
            "angry": "Enojo",
            "surprise": "Sorpresa",
            "neutral": "Neutral"
        }

        # Traducir el texto de la emoción detectada
        mapped_emotion = emotion_mapping.get(emotion_text, None)

        if mapped_emotion in emotion_keys and emotion_keys[mapped_emotion]:
            print(f"Presionando tecla asignada: {emotion_keys[mapped_emotion]} para emoción {mapped_emotion}")
            press_repeatedly(emotion_keys[mapped_emotion])

        draw_bounding_box(face_coordinates, rgb_image, (0, 255, 0))
        draw_text(face_coordinates, rgb_image, emotion_mode, (0, 255, 0), 0, -45, 1, 1)

    # Convertir el frame a formato ImageTk
    combined_image = Image.fromarray(rgb_image)
    imgtk = ImageTk.PhotoImage(image=combined_image)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(10, update_frame)

# Inicializar captura de video
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Iniciar la actualización del video
update_frame()

# Ejecutar la interfaz de Tkinter
root.mainloop()

# Liberar recursos
video_capture.release()
cv2.destroyAllWindows()