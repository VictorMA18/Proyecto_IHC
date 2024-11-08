import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import cv2
from statistics import mode
from keras.models import load_model
import numpy as np
import time

from utils.datasets import get_labels
from utils.inference import detect_faces, draw_text, draw_bounding_box, apply_offsets, load_detection_model
from utils.preprocessor import preprocess_input

# Parameters for loading data and images
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# Hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

emotion = [0, 0, 0, 0, 0] # Agregado Bryan


# Loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# Getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# Starting lists for calculating modes
emotion_window = []

# Initialize Tkinter window
root = tk.Tk()
root.title("Emotion Detector")

# Create a frame for video stream
video_frame = tk.Frame(root)
video_frame.pack(side="left")

# Create a label to show video stream
video_label = Label(video_frame)
video_label.pack()

time_label = tk.Label(root, text="Tiempo de ejecución: 0 segundos", font=("Arial", 16))
time_label.pack(pady=20)

# Create a label to display "Emotions"
text_label0 = Label(root, text="Angry = ", font=("Arial", 24))
text_label0.pack()
text_label1 = Label(root, text="Sad = ", font=("Arial", 24))
text_label1.pack()
text_label2 = Label(root, text="Happy = ", font=("Arial", 24))
text_label2.pack()
text_label3 = Label(root, text="Surprise = ", font=("Arial", 24))
text_label3.pack()
text_label4 = Label(root, text="Neutral = ", font=("Arial", 24))
text_label4.pack()
total_emotions = 0

# Tiempo de inicio
start_time = time.time()

# Función para actualizar el tiempo de ejecución
def update_time():
    elapsed_time = int(time.time() - start_time)
    time_label.config(text=f"Tiempo de ejecución: {elapsed_time} segundos")
    root.after(1000, update_time)

def update_emotion_labels():
    global total_emotions
    if total_emotions > 0:
        text_label0.config(text=f"Angry = {emotion[0] / total_emotions * 100:.2f}%")
        text_label1.config(text=f"Sad = {emotion[1] / total_emotions * 100:.2f}%")
        text_label2.config(text=f"Happy = {emotion[2] / total_emotions * 100:.2f}%")
        text_label3.config(text=f"Surprise = {emotion[3] / total_emotions * 100:.2f}%")
        text_label4.config(text=f"Neutral = {emotion[4] / total_emotions * 100:.2f}%")
    root.after(1000, update_emotion_labels)

# Function to update the video frame
def update_frame():
    ret, bgr_image = video_capture.read()
    if not ret:
        return
    
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # Convert to RGB before display
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
        
        # Agregado por Bryan
        total_emotions = np.sum(emotion)
        
        # print("Angry    = ", emotion[0] / total_emotions * 100) 
        # print("Sad      = ", emotion[1] / total_emotions * 100)
        # print("Happy    = ", emotion[2] / total_emotions * 100)
        # print("Surprise = ", emotion[3] / total_emotions * 100)
        # print("Neutral  = ", emotion[4] / total_emotions * 100)
        
        update_time()
        text_label0.config(text=f"Angry = {emotion[0] / total_emotions * 100:.2f}%")
        text_label1.config(text=f"Sad = {emotion[1] / total_emotions * 100:.2f}%")
        text_label2.config(text=f"Happy = {emotion[2] / total_emotions * 100:.2f}%")
        text_label3.config(text=f"Surprise = {emotion[3] / total_emotions * 100:.2f}%")
        text_label4.config(text=f"Neutral = {emotion[4] / total_emotions * 100:.2f}%")
        root.after(1000, update_emotion_labels)
        
        # Fin de agregado por Bryan

        color = color.astype(int).tolist()
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode, color, 0, -45, 1, 1)

    # Convert RGB image to ImageTk format
    img = Image.fromarray(rgb_image)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(10, update_frame)

# Starting video streaming
video_capture = cv2.VideoCapture(0)
update_frame()
update_emotion_labels()
root.mainloop()

# Release the video capture after closing
video_capture.release()
