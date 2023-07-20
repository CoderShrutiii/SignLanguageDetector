import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
  try:
      # try except to ignore any file except .jpg
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        if not img_path.endswith('.jpg'):
            continue
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # converting each bgr to rgb

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            # node points for hand
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    # code to find landmarks on hand to recognise the pattern formed
                    # mp_drawing.draw_landmarks(
                    #     img_rgb,
                    #     hand_landmarks,
                    #     mp_hands.HAND_CONNECTIONS,
                    #     mp_drawing_styles.get_default_hand_landmarks_style(),
                    #     mp_drawing_styles.get_default_hand_connections_style()
                    # )


                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y


                    data_aux.append(x )
                    data_aux.append(y )
            # plt.figure()
            # plt.imshow(img_rgb)
            data.append(data_aux)
            labels.append(dir_)
    # plt.show()
  except NotADirectoryError:
    # Handle the case when the file is not a directory (e.g., .gitignore)
    pass

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
