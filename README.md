# Sign-Language-To-Speech-and-Text-Converter


## Overview
This project is designed as a tool for **communication between visually impaired individuals and speech impaired individuals**, converting **American Sign Language (ASL), British Sign Language (BSL), Spanish Sign Language (SSL), and Indian Sign Language (ISL)** gestures into spoken language and text. It employs computer vision and machine learning techniques, leveraging MediaPipe for accurate hand landmark detection, OpenCV for efficient image processing, OS for seamless file handling, Pickle for data serialization, scikit-learn for robust machine learning models, and NumPy for essential numerical computations. By translating gestures into both audible speech and displayed text, this tool aims to facilitate better communication accessibility for diverse user needs.

## Features
- **Hand Gesture Recognition:** Utilizes MediaPipe and OpenCV to detect and track hand landmarks, identifying gestures in real-time.
- **Multi-language Support:** Recognizes gestures from American Sign Language (ASL), British Sign Language (BSL), Spanish Sign Language (SSL), and Indian Sign Language (ISL) converting them into words and speaking them.
- **Speech Synthesis:** Converts recognized gestures into spoken language for auditory feedback.
- **Text Display:** Displays recognized gestures as text on the interface for visual feedback.

## Libraries Used

- **MediaPipe**: Provides tools and machine learning models for real-time hand landmark detection and tracking.
- **OpenCV**: Used for image and video processing tasks, including capturing video streams from webcams and processing frames for hand gesture recognition.
- **OS**: Provides functionality for interacting with the operating system, used here for managing directories and files containing training data and models.
- **Pickle**: Used for serializing and deserializing Python objects, enabling efficient storage and retrieval of trained machine learning models.
- **scikit-learn**: Provides a wide range of machine learning algorithms and tools for data preprocessing, model selection, and evaluation, used here for training and predicting hand gesture classifications.
- **NumPy**: Essential for scientific computing in Python, used extensively for numerical operations and handling multi-dimensional arrays, crucial in preprocessing data and performing computations within the machine learning pipeline.


## Datasets Used

- **American Sign Language (ASL)**: Utilized the [ASL Alphabet Dataset on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet?select=asl_alphabet_train) containing images of hand gestures representing letters in ASL.
  
- **Indian Sign Language (ISL)**: Employed the [Indian Sign Language Dataset on Kaggle](https://www.kaggle.com/datasets/vaishnaviasonawane/indian-sign-language-dataset) comprising images of hand gestures used in ISL.

- **Spanish Sign Language (SSL)**: Incorporated the [Spanish Sign Language Alphabet Dataset on Kaggle](https://www.kaggle.com/datasets/kirlelea/spanish-sign-language-alphabet-static) featuring images of hand gestures corresponding to letters in SSL.

- **British Sign Language (BSL)**: Referenced the [BSL Numbers and Alphabet Hand Position Dataset on Kaggle](https://www.kaggle.com/datasets/erentatepe/bsl-numbers-and-alphabet-hand-position-for-mediapipe?select=2_HAND_DATASET) containing hand position data for BSL gestures, compatible with MediaPipe.

These datasets were used to train and validate machine learning models for gesture recognition and translation into speech and text across multiple sign languages.

## How to Use This Project

### Step 1: Download Datasets

1. **American Sign Language (ASL)**
   - Download the [ASL Alphabet Dataset from Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet?select=asl_alphabet_train).
   - Save the dataset into a directory named `Data_ASL`.

2. **Indian Sign Language (ISL)**
   - Download the [Indian Sign Language Dataset from Kaggle](https://www.kaggle.com/datasets/vaishnaviasonawane/indian-sign-language-dataset).
   - Save the dataset into a directory named `Data_ISL`.

3. **Spanish Sign Language (SSL)**
   - Download the [Spanish Sign Language Alphabet Dataset from Kaggle](https://www.kaggle.com/datasets/kirlelea/spanish-sign-language-alphabet-static).
   - Save the dataset into a directory named `Data_SSL`.

4. **British Sign Language (BSL)**
   - Download the [BSL Numbers and Alphabet Hand Position Dataset from Kaggle](https://www.kaggle.com/datasets/erentatepe/bsl-numbers-and-alphabet-hand-position-for-mediapipe?select=2_HAND_DATASET).
   - Save the dataset into a directory named `Data_BSL`.

### Step 2: Data Augmentation

- If needed, run `dataaugmentation.py` to increase the dataset size.
  ```sh
  python dataaugmentation.py
