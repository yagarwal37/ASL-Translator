import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Initialize Mediapipe hand detection and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Create a Mediapipe Hands object for static images
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'  # Directory containing image data

data = []   # List to store landmark data for all images
labels = [] # List to store corresponding class labels

# Loop through each class directory in the data folder
for dir in os.listdir(DATA_DIR):
    # Loop through images in each class directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir)):
        data_aux = []  # Temporary list to store landmarks for this image
        # Read the image using OpenCV
        img = cv2.imread(os.path.join(DATA_DIR, dir, img_path))
        # Convert the image from BGR (OpenCV default) to RGB (Mediapipe expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to detect hands and landmarks
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            # For each detected hand in the image
            for hand_landmarks in results.multi_hand_landmarks:
                # Loop through all landmarks (21 per hand)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x  # Normalized x coordinate
                    y = hand_landmarks.landmark[i].y  # Normalized y coordinate
                    data_aux.append(x)
                    data_aux.append(y)
            # Append the landmarks and label to the main lists
            data.append(data_aux)
            labels.append(dir)

# Save the collected data and labels to a pickle file
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()