import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import pyautogui
import time

# Load trained model & scaler
model = joblib.load("gesture_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Function to send keystrokes based on detected gestures
def control_game(prediction):
    if prediction == 0:  # Example: "Accelerate"
        pyautogui.keyDown("right")  
        pyautogui.keyUp("left")  # Ensure brake is not pressed
        print("Accelerating (Right Arrow)")
    elif prediction == 1:  # Example: "Brake"
        pyautogui.keyDown("left")
        pyautogui.keyUp("right")  # Ensure accelerate is not pressed
        print("Braking (Left Arrow)")
    else:  # No valid gesture, release keys
        pyautogui.keyUp("right")
        pyautogui.keyUp("left")
        print("🖐️ No gesture detected, releasing keys")

# Wait for game to be focused
print("Open Hill Climb Racing and make sure it's the active window!")
time.sleep(5)  # Gives you time to open the game

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for natural mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    landmark_data = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                landmark_data.append(lm.x)
                landmark_data.append(lm.y)
                landmark_data.append(lm.z)

    if len(landmark_data) == 63:  # Ensure correct number of features
        # Convert to DataFrame and apply scaling
        X_test = pd.DataFrame([landmark_data], columns=scaler.feature_names_in_)
        X_test_scaled = scaler.transform(X_test)

        # Predict gesture
        prediction = model.predict(X_test_scaled)[0]

        # Control the game
        control_game(prediction)

    # Show the webcam feed
    cv2.imshow("Gesture Control", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pyautogui.keyUp("right")  # Ensure keys are released on exit
pyautogui.keyUp("left")
print("Exiting gesture control!")