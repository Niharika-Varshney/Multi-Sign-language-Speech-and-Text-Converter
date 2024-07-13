import mediapipe as mp  # Importing the MediaPipe library
import cv2 as cv  # Importing the OpenCV library
import os  # Importing the os library for directory operations
import pickle  # Importing the pickle library for data serialization

# Initializing MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the hands object with static_image_mode=True for processing images
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Initialize lists to store data and labels
data = []
labels = []

# Directory containing the dataset
datadir = 'Data_ASL'

# Loop through each directory in the dataset directory
for dir_ in os.listdir(datadir):
    # Loop through each image file in the current directory
    for img_path in os.listdir(os.path.join(datadir, dir_)):
        aux = []  # Initialize an auxiliary list to store hand landmarks
        img = cv.imread(os.path.join(datadir, dir_, img_path))  # Read the image using OpenCV
        # Convert image to RGB so that we can send it to MediaPipe
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # Process the image to detect hand landmarks
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:  # Check if any hand landmarks are detected
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    # Extract x and y coordinates of each hand landmark
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    aux.append(x)  # Append x coordinate to auxiliary list
                    aux.append(y)  # Append y coordinate to auxiliary list
            data.append(aux)  # Append auxiliary list to data list
            labels.append(dir_)  # Append directory name (label) to labels list

# Open a file in write-binary mode to save the data and labels
f = open('data_ASL.pickle', 'wb')
# Dump the data and labels into the file using pickle
pickle.dump({'data': data, 'labels': labels}, f)
# Close the file
f.close()
