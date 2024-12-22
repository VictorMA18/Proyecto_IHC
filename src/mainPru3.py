import cv2
import mediapipe as mp
from pynput.keyboard import Controller, Key
import tkinter as tk
from PIL import Image, ImageTk
import time

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Inicializar controlador de teclado
keyboard = Controller()

# Configuración de opciones
option = 2  # 1: Presionar una sola vez, 2: Repetir con delay, 3: Mantener presionada
delay_time = 0.5  # Tiempo de delay en segundos entre presionar y soltar la tecla (solo para opción 2)
keys_pressed = {"w": False, "a": False, "s": False, "d": False}  # Estado de las teclas
manos_gesto = {"PulgarDerecho": False, "PunoDerecho": False, "PulgarIzquierdo": False, "PunoIzquierdo": False}
lado_derecho = ""
lado_izquierdo = ""
key_right = ""
key_left = ""

# Variables para determinar si se está detectando la mano derecha o izquierda
right_hand_detected = False
left_hand_detected = False

def is_thumbs_up(landmarks, hand_label, width, height):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    thumb_tip_y = thumb_tip.y * height
    thumb_mcp_y = thumb_mcp.y * height
    index_tip_y = index_tip.y * height
    middle_tip_y = middle_tip.y * height
    ring_tip_y = ring_tip.y * height
    pinky_tip_y = pinky_tip.y * height

    is_thumb_up = thumb_tip_y < thumb_mcp_y
    are_fingers_bent = (index_tip_y > thumb_mcp_y and
                        middle_tip_y > thumb_mcp_y and
                        ring_tip_y > thumb_mcp_y and
                        pinky_tip_y > thumb_mcp_y)

    return is_thumb_up and are_fingers_bent

def is_closed_fist(landmarks, hand_label, width, height):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]

    thumb_tip_y = thumb_tip.y * height
    thumb_mcp_y = thumb_mcp.y * height
    index_tip_y = index_tip.y * height
    index_mcp_y = index_mcp.y * height
    middle_tip_y = middle_tip.y * height
    middle_mcp_y = middle_mcp.y * height
    ring_tip_y = ring_tip.y * height
    ring_mcp_y = ring_mcp.y * height
    pinky_tip_y = pinky_tip.y * height
    pinky_mcp_y = pinky_mcp.y * height

    are_fingers_bent = (index_tip_y > index_mcp_y and
                        middle_tip_y > middle_mcp_y and
                        ring_tip_y > ring_mcp_y and
                        pinky_tip_y > pinky_mcp_y)

    return are_fingers_bent

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

                # Check for thumbs up gesture on right hand
                if is_thumbs_up(hand_landmarks.landmark, hand_label, width, height):
                    if lado_derecho != "PulgarDerecho" and lado_derecho != "":
                        manos_gesto[lado_derecho] = False
                        # lado_derecho = ""
                    lado_derecho = "PulgarDerecho"
                    if option == 1:
                        if not manos_gesto["PulgarDerecho"]:  # Solo presionar si el gesto no fue hecho anteriormente
                            manos_gesto["PulgarDerecho"] = True
                            keyboard.press("w")
                            keyboard.release("w")
                    elif option == 2:
                        press_repeatedly("w", delay_time)
                    elif option == 3:
                        if key_right != "w" and key_right != "":
                            keyboard.release(key_right)
                        # if not manos_gesto["PulgarDerecho"]:
                        #     manos_gesto["PulgarDerecho"] = True
                        key_right = "w"
                        #     # keys_pressed[key_right] = True
                        keyboard.press(key_right)
                    print("Mano derecha detectada: Pulgar arriba")
                
                # Check for closed fist gesture on right hand
                elif is_closed_fist(hand_landmarks.landmark, hand_label, width, height):
                    if lado_derecho != "PunoDerecho" and lado_derecho != "":
                        manos_gesto[lado_derecho] = False
                        # lado_derecho = ""
                    lado_derecho = "PunoDerecho"
                    if option == 1:
                        if not manos_gesto["PunoDerecho"]:  # Solo presionar si el gesto no fue hecho anteriormente
                            manos_gesto["PunoDerecho"] = True
                            keyboard.press("s")
                            keyboard.release("s")
                    elif option == 2:
                        press_repeatedly("s", delay_time)
                    elif option == 3:
                        if key_right != "s" and key_right != "":
                            keyboard.release(key_right)
                        # if not manos_gesto["PunoDerecho"]:
                        #     manos_gesto["PunoDerecho"] = True
                        key_right = "s"
                        #     # keys_pressed[key_right] = True
                        keyboard.press(key_right)
                    print("Mano derecha detectada: Puño cerrado")
                else:
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

                # Check for thumbs up gesture on left hand
                if is_thumbs_up(hand_landmarks.landmark, hand_label, width, height):
                    if lado_izquierdo != "PulgarIzquierdo" and lado_izquierdo != "":
                        manos_gesto[lado_izquierdo] = False
                        # lado_derecho = ""
                    lado_izquierdo = "PulgarIzquierdo"
                    if option == 1:
                        if not manos_gesto["PulgarIzquierdo"]:  # Solo presionar si el gesto no fue hecho anteriormente
                            manos_gesto["PulgarIzquierdo"] = True
                            keyboard.press("a")
                            keyboard.release("a")
                    elif option == 2:
                        press_repeatedly("a", delay_time)
                    elif option == 3:
                        if key_left != "a" and key_left != "":
                            keyboard.release(key_left)
                        key_left = "a"
                        keyboard.press(key_left)
                    print("Mano izquierda detectada: Pulgar arriba")
                
                # Check for closed fist gesture on left hand
                elif is_closed_fist(hand_landmarks.landmark, hand_label, width, height):
                    if lado_izquierdo != "PunoIzquierdo" and lado_izquierdo != "":
                        manos_gesto[lado_izquierdo] = False
                        # lado_derecho = ""
                    lado_izquierdo = "PunoIzquierdo"
                    if option == 1:
                        if not manos_gesto["PunoIzquierdo"]:  # Solo presionar si el gesto no fue hecho anteriormente
                            manos_gesto["PunoIzquierdo"] = True
                            keyboard.press("d")
                            keyboard.release("d")
                    elif option == 2:
                        press_repeatedly("d", delay_time)
                    elif option == 3:
                        if key_left != "d" and key_left != "":
                            keyboard.release(key_left)
                        key_left = "d"
                        keyboard.press(key_left)
                    print("Mano izquierda detectada: Puño cerrado")
                else:
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
