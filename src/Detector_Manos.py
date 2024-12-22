import cv2
import mediapipe as mp
import numpy as np

def press_repeatedly(key, delay):
    """Presiona y suelta repetidamente la tecla con un delay."""
    keyboard.press(key)
    time.sleep(delay)
    keyboard.release(key)

def distancia_euclidiana(p1, p2):
    d = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return d

def draw_bounding_box(image, hand_landmarks):
    image_height, image_width, _ = image.shape
    x_min, y_min = image_width, image_height
    x_max, y_max = 0, 0
    
    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * image_width), int(landmark.y * image_height)
        if x < x_min: x_min = x
        if y < y_min: y_min = y
        if x > x_max: x_max = x
        if y > y_max: y_max = y
    
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

# Función para gestionar el desplazamiento cíclico
def update_offset(offset, max_height, step):
    offset -= step
    if offset < -max_height:
        offset = 0
    return offset
    
# Función para gestionar el desplazamiento cíclico
def update_offset(offset, max_height, step):
    offset -= step
    if offset < -max_height:
        offset = 0
    return offset

# Cargar las imágenes de las emociones
images = {
    "Amor y Paz": cv2.imread("assets/amor_y_paz.png"),
    "Pulgar Abajo": cv2.imread("assets/pulgar_abajo.png"),
    "Bueno": cv2.imread("assets/bueno.png"),
    "OK": cv2.imread("assets/ok.png"),
    "Mano Abierta": cv2.imread("assets/mano_abierta.png"),
    "Cerrado": cv2.imread("assets/cerrado.png"),
    "Rock On": cv2.imread("assets/rock_on.png"),
    "Loser (L)": cv2.imread("assets/loser.png")
}

# Redimensionar imágenes
for key in images.keys():
    images[key] = cv2.resize(images[key], (60, 60))

# Variables para animación
scroll_offset = 0
scroll_step = 2
emotion_height = 80
base_offset = 230
emotion = ""
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Crear el fondo del dashboard con color blanco
        dashboard = np.ones((image.shape[0], 400, 3), dtype=np.uint8) * 255  # Fondo blanco
        cv2.rectangle(dashboard, (0, 0), (dashboard.shape[1], 50), (169, 169, 169), -1)  # Gris

        # Añadir el texto "Detección de Gestos" en el cuadro gris
        text = "DETECCION DE GESTOS"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_x = (dashboard.shape[1] - text_width) // 2  # Centrar el texto horizontalmente
        cv2.putText(dashboard, text, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                draw_bounding_box(image, hand_landmarks)
                # Coordenadas de puntos clave
                thumb_tip = (int(hand_landmarks.landmark[4].x * image.shape[1]), 
                             int(hand_landmarks.landmark[4].y * image.shape[0]))
                thumb_pip = (int(hand_landmarks.landmark[3].x * image.shape[1]), 
                             int(hand_landmarks.landmark[3].y * image.shape[0]))

                index_finger_tip = (int(hand_landmarks.landmark[8].x * image.shape[1]), 
                                    int(hand_landmarks.landmark[8].y * image.shape[0]))
                index_finger_pip = (int(hand_landmarks.landmark[6].x * image.shape[1]), 
                                    int(hand_landmarks.landmark[6].y * image.shape[0]))

                middle_finger_tip = (int(hand_landmarks.landmark[12].x * image.shape[1]), 
                                     int(hand_landmarks.landmark[12].y * image.shape[0]))
                middle_finger_pip = (int(hand_landmarks.landmark[10].x * image.shape[1]), 
                                     int(hand_landmarks.landmark[10].y * image.shape[0]))

                ring_finger_tip = (int(hand_landmarks.landmark[16].x * image.shape[1]), 
                                   int(hand_landmarks.landmark[16].y * image.shape[0]))
                ring_finger_pip = (int(hand_landmarks.landmark[14].x * image.shape[1]), 
                                   int(hand_landmarks.landmark[14].y * image.shape[0]))

                pinky_tip = (int(hand_landmarks.landmark[20].x * image.shape[1]), 
                             int(hand_landmarks.landmark[20].y * image.shape[0]))
                pinky_pip = (int(hand_landmarks.landmark[18].x * image.shape[1]), 
                             int(hand_landmarks.landmark[18].y * image.shape[0]))

                # Inicializar emoción
                emotion = "N/A"

                # Detectar gestos
                if (index_finger_tip[1] < index_finger_pip[1] and  
                    middle_finger_tip[1] < middle_finger_pip[1] and 
                    ring_finger_tip[1] > ring_finger_pip[1] and    
                    pinky_tip[1] > pinky_pip[1] and                 
                    abs(thumb_tip[0] - index_finger_tip[0]) > 30 and 
                    abs(middle_finger_tip[0] - index_finger_tip[0]) > 20):
                    emotion = "Amor y Paz"
                
                elif (thumb_tip[1] > thumb_pip[1] and                   
                      index_finger_tip[1] > index_finger_pip[1] and    
                      middle_finger_tip[1] > middle_finger_pip[1] and   
                      ring_finger_tip[1] > ring_finger_pip[1] and      
                      pinky_tip[1] > pinky_pip[1]):
                    emotion = "Pulgar Abajo"
                
                elif abs(index_finger_tip[1] - thumb_tip[1]) < 360 and \
                     index_finger_tip[1] - middle_finger_pip[1] < 0 and \
                     index_finger_tip[1] - middle_finger_tip[1] < 0 and \
                     index_finger_tip[1] - index_finger_pip[1] > 0:
                    emotion = "Bueno"
                
                elif (abs(thumb_tip[0] - index_finger_tip[0]) < 20 and  
                      middle_finger_tip[1] < middle_finger_pip[1] and   
                      ring_finger_tip[1] < ring_finger_pip[1] and      
                      pinky_tip[1] < pinky_pip[1]):
                    emotion = "OK"
                
                elif (index_finger_tip[1] < index_finger_pip[1] and    
                      middle_finger_tip[1] < middle_finger_pip[1] and   
                      ring_finger_tip[1] < ring_finger_pip[1] and       
                      pinky_tip[1] < pinky_pip[1] and                   
                      thumb_tip[0] < index_finger_tip[0]):
                    emotion = "Mano Abierta"
                
                elif (index_finger_tip[1] > index_finger_pip[1] and    
                      middle_finger_tip[1] > middle_finger_pip[1] and  
                      ring_finger_tip[1] > ring_finger_pip[1] and      
                      pinky_tip[1] > pinky_pip[1] and                   
                      thumb_tip[0] < index_finger_pip[0]):
                    emotion = "Cerrado"
    
                elif (index_finger_tip[1] < index_finger_pip[1] and  
                      pinky_tip[1] < pinky_pip[1] and  
                      abs(thumb_tip[0] - index_finger_tip[0]) < 20 and  
                      middle_finger_tip[1] > middle_finger_pip[1] and 
                      ring_finger_tip[1] > ring_finger_pip[1]): 
                    emotion = "Rock On"
                    
                elif (index_finger_tip[1] < index_finger_pip[1] and  
                      thumb_tip[1] < thumb_pip[1] and  
                      middle_finger_tip[1] > middle_finger_pip[1] and  
                      ring_finger_tip[1] > ring_finger_pip[1] and  
                      pinky_tip[1] > pinky_pip[1]): 
                    emotion = "Loser (L)"
            


                # Mostrar emoción en el dashboard
                cv2.putText(dashboard, f"Emotion: {emotion}", (10, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(dashboard, "Mano Detectada", (10, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        else:
            cv2.putText(dashboard, "Mano sin detectar", (10, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Actualizar desplazamiento
        scroll_offset = update_offset(scroll_offset, emotion_height * len(images), scroll_step)

        # Dibujar las emociones con estilos
        for i, (emotion_name, emotion_image) in enumerate(images.items()):
            y_offset = base_offset + (scroll_offset + i * emotion_height) % (emotion_height * len(images)) - emotion_height
            x_offset = 10

            # Validar que las coordenadas estén dentro de los límites del dashboard
            if base_offset <= y_offset < dashboard.shape[0] - 60:
                # Fondo circular blanco detrás de la imagen
                center_x = x_offset + 30
                center_y = y_offset + 30
                cv2.circle(dashboard, (center_x, center_y), 40, (255, 255, 255), -1)  # Cambiado a blanco
                cv2.rectangle(dashboard, (x_offset + 5, y_offset + 5), (x_offset + 65, y_offset + 65), (200, 200, 200), -1)
                
                # Dibujar la imagen
                dashboard[y_offset:y_offset+60, x_offset:x_offset+60] = emotion_image
                
                # Resaltar si es la emoción detectada
                if emotion_name == emotion:  # Aquí usamos la emoción detectada
                    cv2.rectangle(dashboard, (x_offset - 5, y_offset - 5), (x_offset + 65, y_offset + 65), (0, 255, 255), 3)
                
                
                # Mostrar el nombre de la emoción
                cv2.putText(dashboard, emotion_name, (x_offset + 70, y_offset + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


        combined_frame = np.hstack((image, dashboard))
        combined_resized = cv2.resize(combined_frame, (720, 360))
        cv2.imshow('MediaPipe Hands with Dashboard', combined_resized)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()