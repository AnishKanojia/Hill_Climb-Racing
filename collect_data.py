import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

gesture_labels = {0: "ACCELERATE", 1: "BRAKE", 2: "NONE"}
data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    landmark_data = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmark_data.append(lm.x)
                landmark_data.append(lm.y)
                landmark_data.append(lm.z)  # Add Z-coordinates (63 features)

    # Ensure consistent feature length (63 features)
    while len(landmark_data) < 63:
        landmark_data.append(0.0)  # Add zeros for missing values

    cv2.imshow("Collect Data", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('a'):  # ACCELERATE
        data.append(landmark_data + [0])
    elif key == ord('b'):  # BRAKE
        data.append(landmark_data + [1])
    elif key == ord('n'):  # NONE
        data.append(landmark_data + [2])
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save Data
df = pd.DataFrame(data)
df.to_csv("gesture_data.csv", index=False)
print("Data saved with 63 features.")