import cv2
import mediapipe as mp
from pynput.keyboard import Controller, Key
import tkinter as tk
from PIL import Image, ImageTk
import time
import gestures2 as gs

# Configuración de opciones
option = 3  # 1: Presionar una sola vez, 2: Repetir con delay, 3: Mantener presionada
delay_time = 0.5  # Tiempo de delay en segundos entre presionar y soltar la tecla (solo para opción 2)
# keys_pressed = {"w": False, "a": False, "s": False, "d": False}  # Estado de las teclas
manos_gesto = {"pulgarArriba_Derecha": False, "pulgarArriba_Izquierda": False,
               "pulgarAbajo_Derecha": False, "pulgarAbajo_Izquierda": False,
               "manoAbierta_Derecha": False, "manoAbierta_Izquierda": False,
               "manoCerrada_Derecha": False, "manoCerrada_Izquierda": False,
               "amorYPaz_Derecha": False, "amorYPaz_Izquierda": False,
               "okay_Derecha": False, "okay_Izquierda": False,
               "rockOn_Derecha": False, "rockOn_Izquierda": False,
               "letraL_Derecha": False, "letraL_Izquierda": False}
lado_derecho = ""
lado_izquierdo = ""
key_right = ""
key_left = ""

# Variables para determinar si se está detectando la mano derecha o izquierda
right_hand_detected = False
left_hand_detected = False

detect_gestures = {
    "pulgarArriba": lambda hand_label, width, height: gs.is_pulgar_arriba(hand_label, width, height),
    "pulgarAbajo": lambda hand_label, width, height: gs.is_pulgar_abajo(hand_label, width, height),
    "manoAbierta": lambda hand_label, width, height: gs.is_mano_abierta(hand_label, width, height),
    "manoCerrada": lambda hand_label, width, height: gs.is_mano_cerrada(hand_label, width, height),
    "amorYPaz": lambda hand_label, width, height: gs.is_amor_y_paz(hand_label, width, height),
    "okay": lambda hand_label, width, height: gs.is_okay(hand_label, width, height),
    "rockOn": lambda hand_label, width, height: gs.is_rock_on(hand_label, width, height),
    "letraL": lambda hand_label, width, height: gs.is_letra_l(hand_label, width, height)
}


def gesture_detector(gesture_right_list, gesture_left_list):
    print(gesture_right_list)
    print(gesture_left_list)
    # Inicializar MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    # Inicializar controlador de teclado
    keyboard = Controller()


    def press_repeatedly(key, delay):
        """Presiona y suelta repetidamente la tecla con un delay."""
        keyboard.press(key)
        time.sleep(delay)
        keyboard.release(key)
    

    def process_frame():
        global right_hand_detected, left_hand_detected, lado_derecho, lado_izquierdo, manos_gesto, keys_pressed, key_right, key_left, option
        ret, frame = cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        right_hand_detected = False  # Reset detection status
        left_hand_detected = False   # Reset detection status

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_label = results.multi_handedness[idx].classification[0].label
                height, width, _ = frame.shape

                # Detecting right hand
                if hand_label == "Right":
                    right_hand_detected = True
                    gesture_detected = False
                    gs.update_points(hand_landmarks.landmark, mp_hands)

                    # Check for thumbs up gesture on right hand
                    # if gs.is_thumbs_up(hand_landmarks.landmark, hand_label, width, height, mp_hands):
                    for gesture_data in gesture_right_list:
                        side = gesture_data['side']
                        gesture = gesture_data['gesture']
                        key = gesture_data['key']
                        gestureSide = gesture+"_"+side

                        if detect_gestures[gesture](hand_label, width, height):
                            gesture_detected = True

                            if lado_derecho != gestureSide and lado_derecho != "":
                                manos_gesto[lado_derecho] = False
                                # lado_derecho = ""
                            lado_derecho = gestureSide
                            if option == 1:
                                if not manos_gesto[gestureSide]:  # Solo presionar si el gesto no fue hecho anteriormente
                                    manos_gesto[gestureSide] = True
                                    keyboard.press(key)
                                    keyboard.release(key)
                            elif option == 2:
                                press_repeatedly(key, delay_time)
                            elif option == 3:
                                if key_right != key and key_right != "":
                                    keyboard.release(key_right)
                                # if not manos_gesto["PulgarDerecho"]:
                                #     manos_gesto["PulgarDerecho"] = True
                                key_right = key
                                #     # keys_pressed[key_right] = True
                                keyboard.press(key_right)
                            print("Mano derecha detectada: " + gestureSide)

                    if not gesture_detected:
                        if lado_derecho != "":
                            manos_gesto[lado_derecho] = False
                            lado_derecho = ""
                            if option == 3:
                                keyboard.release(key_right)
                                key_right = ""
                        print("Mano derecha detectada, pero no se detecta ningún gesto específico.")

                # Detecting left hand
                elif hand_label == "Left":
                    left_hand_detected = True
                    gesture_detected = False
                    gs.update_points(hand_landmarks.landmark, mp_hands)

                    # Check for thumbs up gesture on left hand
                    for gesture_data in gesture_left_list:
                        side = gesture_data['side']
                        gesture = gesture_data['gesture']
                        key = gesture_data['key']
                        gestureSide = gesture+"_"+side

                        if detect_gestures[gesture](hand_label, width, height):
                            gesture_detected = True

                            if lado_izquierdo != gestureSide and lado_izquierdo != "":
                                manos_gesto[lado_izquierdo] = False
                                # lado_derecho = ""
                            lado_izquierdo = gestureSide
                            if option == 1:
                                if not manos_gesto[gestureSide]:  # Solo presionar si el gesto no fue hecho anteriormente
                                    manos_gesto[gestureSide] = True
                                    keyboard.press(key)
                                    keyboard.release(key)
                            elif option == 2:
                                press_repeatedly(key, delay_time)
                            elif option == 3:
                                if key_left != key and key_left != "":
                                    keyboard.release(key_left)
                                key_left = key
                                keyboard.press(key_left)
                            print("Mano izquierda detectada: " + gestureSide)
                    
                    if not gesture_detected:
                        if lado_izquierdo != "":
                            manos_gesto[lado_izquierdo] = False
                            lado_izquierdo = ""
                            if option == 3:
                                keyboard.release(key_left)
                                key_left = ""
                        print("Mano izquierda detectada, pero no se detecta ningún gesto específico.")

        # Imprimir si no se detecta ninguna mano
        if not right_hand_detected:
            if lado_derecho != "":
                manos_gesto[lado_derecho] = False
                lado_derecho = ""
                if option == 3:
                    keyboard.release(key_right)
                    key_right = ""
            print("No se detecta mano derecha.")
        if not left_hand_detected:
            if lado_izquierdo != "":
                manos_gesto[lado_izquierdo] = False
                lado_izquierdo = ""
                if option == 3:
                    keyboard.release(key_left)
                    key_left = ""
            print("No se detecta mano izquierda.")

        # Mostrar el frame en la ventana de tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        label.after(10, process_frame)

    # Configuración de la ventana de tkinter
    root = tk.Tk()
    root.title("Reconocimiento de Gestos")
    root.geometry("640x480")
    root.attributes('-topmost', 1)  # Mantener siempre al frente

    label = tk.Label(root)
    label.pack()

    cap = cv2.VideoCapture(0)

    process_frame()

    root.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), root.destroy()))
    root.mainloop()
