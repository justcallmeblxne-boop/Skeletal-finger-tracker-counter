import cv2
import mediapipe as mp
import numpy as np

# config
MAX_HANDS = 2
DETECT_CONF = 0.9
TRACK_CONF = 0.9
WINDOW = "Finger Counter"
HAND_COLORS = [(0, 200, 0), (0, 140, 255)]

# setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=DETECT_CONF,
    min_tracking_confidence=TRACK_CONF,
)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# ------------
# finger name map
# ------------
FINGER_NAME_BY_ID = {
    0: "palm",
    1: "thumb_1", 2: "thumb_2", 3: "thumb_3", 4: "thumb_4",
    5: "index_1", 6: "index_2", 7: "index_3", 8: "index_4",
    9: "middle_1", 10: "middle_2", 11: "middle_3", 12: "middle_4",
    13: "ring_1", 14: "ring_2", 15: "ring_3", 16: "ring_4",
    17: "pinky_1", 18: "pinky_2", 19: "pinky_3", 20: "pinky_4",
}

# ------------
# count fingers
# ------------
def count_fingers(hand_landmarks, handedness_label):
    tip_ids = [4, 8, 12, 16, 20]
    pip_ids = [2, 6, 10, 14, 18]
    up = []
    if handedness_label == "Right":
        up.append(1 if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[pip_ids[0]].x else 0)
    else:
        up.append(1 if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[pip_ids[0]].x else 0)
    for tip, pip in zip(tip_ids[1:], pip_ids[1:]):
        up.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y else 0)
    return sum(up)

# ------------
# draw rounded rect
# ------------
def draw_rounded_rect(img, top_left, bottom_right, color, radius=10, alpha=0.6):
    overlay = img.copy()
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

# ------------
# draw bar
# ------------
def draw_bar(img, top_left, width, height, frac, color):
    x, y = top_left
    cv2.rectangle(img, (x, y), (x + width, y + height), (40, 40, 40), -1)
    fill = int(width * np.clip(frac, 0.0, 1.0))
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + height), color, -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (200, 200, 200), 1)

# ------------
# main loop
# ------------
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    # draw per-hand boxes on right 
    if res.multi_hand_landmarks:
        for i in range(min(len(res.multi_hand_landmarks), MAX_HANDS)):
            lm = res.multi_hand_landmarks[i]
            handed = res.multi_handedness[i].classification[0]
            label = handed.label
            score = handed.score
            color = HAND_COLORS[i % len(HAND_COLORS)]
            ctext = (int(color[0]), int(color[1]), int(color[2]))

            mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=color, thickness=2))

            p_w = 380
            p_h = 80
            p_x1 = w - p_w - 12
            p_y1 = 8 + i * (p_h + 8)
            draw_rounded_rect(frame, (p_x1, p_y1), (p_x1 + p_w, p_y1 + p_h), (8,8,8), radius=8, alpha=0.65)

            fingers_up = count_fingers(lm, label)
            cv2.putText(frame, f'Hand {i+1}: {label} ({fingers_up})', (p_x1 + 12, p_y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ctext, 2)
            cv2.putText(frame, 'Detect:', (p_x1 + 12, p_y1 + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)
            draw_bar(frame, (p_x1 + 70, p_y1 + 46), 120, 12, score, color)
            cv2.putText(frame, 'Track:', (p_x1 + 200, p_y1 + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)
            draw_bar(frame, (p_x1 + 250, p_y1 + 46), 110, 12, TRACK_CONF, (100,100,220))

            
            for lm_id, l in enumerate(lm.landmark):
                cx, cy = int(l.x * w), int(l.y * h)
                text = f'({cx},{cy})'
                tx_w, tx_h = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
                tx = cx + 8
                ty = cy - 8
                if tx + tx_w > w - 10:
                    tx = cx - tx_w - 12
                cv2.rectangle(frame, (tx - 2, ty - tx_h - 2), (tx + tx_w + 2, ty + 2), (12,12,12), -1)
                cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ctext, 1)

    # version box bottom right
    ver_w, ver_h = 220, 40
    ver_x1 = w - ver_w - 12
    ver_y1 = h - ver_h - 12
    draw_rounded_rect(frame, (ver_x1, ver_y1), (ver_x1 + ver_w, ver_y1 + ver_h), (10,10,10), radius=8, alpha=0.7)
    cv2.putText(frame, 'Version 1.2', (ver_x1 + 12, ver_y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 2)

    # hands detected box bottom left
    info_w, info_h = 320, 72
    info_x1 = 12
    info_y1 = h - info_h - 12
    draw_rounded_rect(frame, (info_x1, info_y1), (info_x1 + info_w, info_y1 + info_h), (10,10,10), radius=8, alpha=0.6)
    num_hands = 0
    if res.multi_hand_landmarks:
        num_hands = min(len(res.multi_hand_landmarks), MAX_HANDS)
    cv2.putText(frame, f'Hands Detected: {num_hands}', (info_x1 + 12, info_y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240,240,240), 2)
    cv2.putText(frame, f'Model DetectConf: {DETECT_CONF:.2f}', (info_x1 + 12, info_y1 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    
    cv2.imshow(WINDOW, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
