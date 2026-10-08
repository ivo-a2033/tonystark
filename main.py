import os
import time
import urllib.request

import cv2
import mediapipe as mp

MODEL = "hand_landmarker.task"
URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
if not os.path.exists(MODEL):
    urllib.request.urlretrieve(URL, MODEL)

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,
)

cap = cv2.VideoCapture(0)
t0 = time.time()

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int((time.time() - t0) * 1000))

        h, w = frame.shape[:2]
        for hand in result.hand_landmarks:  # 21 normalized landmarks per hand
            for lm in hand:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)

        cv2.imshow("Hand tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Esc to quit
            break

cap.release()
cv2.destroyAllWindows()