import cv2 as cv  # Importing OpenCV for video capture and processing
import mediapipe as mp  # Importing MediaPipe for hand tracking
import pickle  # Importing pickle for loading the trained model
import numpy as np  # Importing NumPy for numerical operations
import warnings  # Importing warnings to handle warnings

import concurrent.futures  # Importing concurrent.futures for future use (if needed)

# Suppress specific UserWarning from google.protobuf.symbol_database
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

# Load the trained model from the pickle file
model_dict = pickle.load(open('model_ISL.pkl', 'rb'))
model = model_dict['model']

# Initialize video capture from the default camera (usually the first one)
cap = cv.VideoCapture(0)

# Initialize MediaPipe hands for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Define gesture labels for prediction output
labels_dict = {
    "A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "F", "G": "G", "H": "H", "I": "I", "J": "J",
    "K": "K", "L": "L", "M": "M", "N": "N", "O": "O", "P": "P", "Q": "Q", "R": "R", "S": "S", "T": "T",
    "U": "U", "V": "V", "W": "W", "X": "X", "Y": "Y", "Z": "Z"
}

while True:
    data_aux = []  # Auxiliary list to store hand landmark data
    x_ = []  # List to store x-coordinates of hand landmarks
    y_ = []  # List to store y-coordinates of hand landmarks
    ret, frame = cap.read()  # Read a frame from the video capture
    if not ret:
        break  # Exit loop if there is an issue with video capture

    H, W, _ = frame.shape  # Get the height and width of the frame

    # Convert frame to RGB (required by MediaPipe)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame,
                                      hand_landmarks,
                                      mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())
            # Collect hand landmark data
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        # Calculate bounding box for the detected hand
        x1 = int(min(x_) * W)
        y1 = int(min(y_) * H)
        x2 = int(max(x_) * W)
        y2 = int(max(y_) * H)

        # Ensure data length consistency for the model
        if len(data_aux) == 42:  # Only one hand detected
            data_aux.extend([0] * 42)  # Pad with zeros for the second hand
        elif len(data_aux) > 84:  # More than two hands detected, truncate to two hands
            data_aux = data_aux[:84]

        # Make prediction using the trained model
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[str(prediction[0])]

        # Draw bounding box and predicted text on the frame
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness=4)
        cv.putText(frame, predicted_character, (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv.LINE_AA)

    # Display the frame
    cv.imshow('frame', frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break  # Exit loop if 'q' key is pressed

# Release video capture and close display windows
cap.release()
cv.destroyAllWindows()
