import cv2
import mediapipe as mp
import pyautogui
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import time
import numpy as np

try:
    from webdriver_manager.chrome import ChromeDriverManager
except ModuleNotFoundError:
    import os
    os.system('pip install webdriver-manager')
    from webdriver_manager.chrome import ChromeDriverManager

def open_youtube_shorts():
    options = webdriver.ChromeOptions()
    options.add_experimental_option("detach", True)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get("https://www.youtube.com/shorts")
    return driver

def draw_animated_icon(frame, center, frame_count, label, color):
    x, y = center
    pulse = int(10 * np.sin(frame_count / 5.0))
    radius = 40 + pulse
    thickness = 3 + pulse // 5

    cv2.circle(frame, (x, y), radius, color, thickness)
    cv2.line(frame, (x - 20, y - 20), (x + 20, y + 20), color, 2)
    cv2.line(frame, (x + 20, y - 20), (x - 20, y + 20), color, 2)
    cv2.putText(frame, label, (x - 35, y + 60), cv2.FONT_HERSHEY_DUPLEX, 1.0 + pulse / 20, color, 2)

def detect_pinch_gesture():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.8)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    cooldown = 0
    anim_timer = 0
    anim_label = ""
    anim_center = (0, 0)
    anim_color = (255, 255, 255)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                pinch_index_dist = np.linalg.norm(
                    np.array([thumb_tip.x, thumb_tip.y]) -
                    np.array([index_tip.x, index_tip.y])
                )
                pinch_middle_dist = np.linalg.norm(
                    np.array([thumb_tip.x, thumb_tip.y]) -
                    np.array([middle_tip.x, middle_tip.y])
                )

                h, w, _ = frame.shape
                cx = int((thumb_tip.x + index_tip.x) / 2 * w)
                cy = int((thumb_tip.y + index_tip.y) / 2 * h)

                if cooldown == 0:
                    if pinch_index_dist < 0.04:
                        pyautogui.press('down')
                        anim_center = (cx, cy)
                        anim_label = "NEXT"
                        anim_color = (0, 255, 0)
                        anim_timer = 15
                        cooldown = 5
                    elif pinch_middle_dist < 0.04:
                        pyautogui.press('up')
                        anim_center = (cx, cy)
                        anim_label = "PREV"
                        anim_color = (0, 0, 255)
                        anim_timer = 15
                        cooldown = 5

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if cooldown > 0:
            cooldown -= 1

        if anim_timer > 0:
            draw_animated_icon(frame, anim_center, anim_timer, anim_label, anim_color)
            anim_timer -= 1

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    driver = open_youtube_shorts()
    time.sleep(5)
    detect_pinch_gesture()
