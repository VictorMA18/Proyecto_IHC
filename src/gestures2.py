# detecta mano abierta inclinada hacia la derecha
# detecta mejor el okay de costado
# detecta L inclinada hacia afuera
# detecta rock on inclinada hacia dentro
wrist = 0
thumb_cmc = 0
thumb_mcp = 0
thumb_ip = 0
thumb_tip = 0
index_finger_mcp = 0
index_finger_pip = 0
index_finger_dip = 0
index_finger_tip = 0
middle_finger_mcp = 0
middle_finger_pip = 0
middle_finger_dip = 0
middle_finger_tip = 0
ring_finger_mcp = 0
ring_finger_pip = 0
ring_finger_dip = 0
ring_finger_tip = 0
pinky_mcp = 0
pinky_pip = 0
pinky_dip = 0
pinky_tip = 0

def update_points(landmarks, mp_hands):
    global wrist, thumb_cmc, thumb_mcp, thumb_ip, thumb_tip, index_finger_mcp, index_finger_pip, index_finger_dip, index_finger_tip, middle_finger_mcp, middle_finger_pip, middle_finger_dip, middle_finger_tip, ring_finger_mcp, ring_finger_pip, ring_finger_dip, ring_finger_tip, pinky_mcp, pinky_pip, pinky_dip, pinky_tip
    
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    thumb_cmc = landmarks[mp_hands.HandLandmark.THUMB_CMC]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    index_finger_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_finger_dip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    middle_finger_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    middle_finger_dip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    middle_finger_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
    ring_finger_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]
    ring_finger_dip = landmarks[mp_hands.HandLandmark.RING_FINGER_DIP]
    ring_finger_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]
    pinky_pip = landmarks[mp_hands.HandLandmark.PINKY_PIP]
    pinky_dip = landmarks[mp_hands.HandLandmark.PINKY_DIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

def is_pulgar_arriba(hand_label, width, height):
    is_thumb_up = thumb_tip.y * height < thumb_mcp.y * height
    are_fingers_bent = (index_finger_tip.y * height > thumb_mcp.y * height and
                        middle_finger_tip.y * height > thumb_mcp.y * height and
                        ring_finger_tip.y * height > thumb_mcp.y * height and
                        pinky_tip.y * height > thumb_mcp.y * height)

    return is_thumb_up and are_fingers_bent

def is_pulgar_abajo(hand_label, width, height):
    return (
        thumb_tip.y * height > thumb_ip.y * height and
        index_finger_tip.y * height > index_finger_pip.y * height and
        middle_finger_tip.y * height > middle_finger_pip.y * height and
        ring_finger_tip.y * height > ring_finger_pip.y * height and
        pinky_tip.y * height > pinky_pip.y * height
    )

def is_mano_abierta(hand_label, width, height):
    return (
        index_finger_tip.y * height < index_finger_pip.y * height and
        middle_finger_tip.y * height < middle_finger_pip.y * height and
        ring_finger_tip.y * height < ring_finger_pip.y * height and
        pinky_tip.y * height < pinky_pip.y * height and
        thumb_tip.x * width < index_finger_tip.x * width
    )

def is_mano_cerrada(hand_label, width, height):
    are_fingers_bent = (index_finger_tip.y * height > index_finger_mcp.y * height and
                        middle_finger_tip.y * height > middle_finger_mcp.y * height and
                        ring_finger_tip.y * height > ring_finger_mcp.y * height and
                        pinky_tip.y * height > pinky_mcp.y * height)

    return are_fingers_bent

def is_amor_y_paz(hand_label, width, height):
    return (
        index_finger_tip.y * height < index_finger_pip.y * height and
        middle_finger_tip.y * height < middle_finger_pip.y * height and
        ring_finger_tip.y * height > ring_finger_pip.y * height and
        pinky_tip.y * height > pinky_pip.y * height and
        abs(thumb_tip.x * width - index_finger_tip.x * width) > 30 and
        abs(middle_finger_tip.x * width - index_finger_tip.x * width) > 20
    )



def is_okay(hand_label, width, height):
    return (
        abs(thumb_tip.x * width - index_finger_tip.x * width) < 20 and
        middle_finger_tip.y * height < middle_finger_pip.y * height and
        ring_finger_tip.y * height < ring_finger_pip.y * height and
        pinky_tip.y * height < pinky_pip.y * height
    )

# def is_pulgar_arriba(hand_label, width, height):
#     return (
#         abs(index_finger_tip.y * height - thumb_tip.y * height) < 360 and
#         index_finger_tip.y * height < middle_finger_pip.y * height and
#         index_finger_tip.y * height < middle_finger_tip.y * height and
#         index_finger_tip.y * height > index_finger_pip.y * height
#     )



def is_rock_on(hand_label, width, height):
    return (
        index_finger_tip.y * height < index_finger_pip.y * height and
        pinky_tip.y * height < pinky_pip.y * height and
        abs(thumb_tip.x * width - index_finger_tip.x * width) < 20 and
        middle_finger_tip.y * height > middle_finger_pip.y * height and
        ring_finger_tip.y * height > ring_finger_pip.y * height
    )

def is_letra_l(hand_label, width, height):
    return (
        index_finger_tip.y * height < index_finger_pip.y * height and
        thumb_tip.y * height < thumb_ip.y * height and
        middle_finger_tip.y * height > middle_finger_pip.y * height and
        ring_finger_tip.y * height > ring_finger_pip.y * height and
        pinky_tip.y * height > pinky_pip.y * height
    )