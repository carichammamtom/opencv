import cv2 as cv
import mediapipe as mp

# Khởi tạo mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7
)

cap = cv.VideoCapture(0)

canvas = None
draw_mode = True  # True: vẽ, False: xóa

def is_finger_folded(landmarks, tip_id, pip_id, h):
    """Kiểm tra 1 ngón có cụp không: đầu ngón nằm dưới khớp giữa"""
    return landmarks[tip_id].y * h > landmarks[pip_id].y * h

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)

    h, w, _ = frame.shape
    if canvas is None:
        canvas = frame.copy() * 0

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark

            # Kiểm tra số lượng ngón tay đang cụp
            folded = 0
            fingers = [(8, 6), (12, 10), (16, 14), (20, 18)]  # trỏ, giữa, áp út, út
            for tip, pip in fingers:
                if is_finger_folded(landmarks, tip, pip, h):
                    folded += 1

            # Cập nhật chế độ vẽ/xóa
            if folded >= 3:
                draw_mode = False  # đang nắm tay
                mode_text = "Erase mode"
            else:
                draw_mode = True   # mở tay
                mode_text = "Draw mode"

            # Lấy tọa độ đầu ngón trỏ
            x = int(landmarks[8].x * w)
            y = int(landmarks[8].y * h)

            # Vẽ/xóa trên canvas
            if draw_mode:
                cv.circle(canvas, (x, y), 6, (50, 50, 50), -1)  # xám đậm
            else:
                cv.circle(canvas, (x, y), 20, (0, 0, 0), -1)  # xóa = màu nền

            # Vẽ bàn tay lên frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Hiển thị chế độ hiện tại
            cv.putText(frame, mode_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Gộp nét vẽ với ảnh camera
    frame = cv.add(frame, canvas)

    cv.imshow("Paint with Hand", frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = frame.copy() * 0  # clear canvas

cap.release()
cv.destroyAllWindows()
