import cv2  # Importing OpenCV for image manipulation
import numpy as np  # Importing NumPy for numerical operations
import Augmentor  # Importing Augmentor for image augmentation
import os  # Importing os for file operations


def augment_images_for_alphabet(input_base_directory, num_samples=1000):
    """
    Augments images for each letter directory from A to Z in the input base directory.

    Args:
    - input_base_directory (str): Base directory containing subdirectories for each letter.
    - num_samples (int): Number of augmented samples to generate for each letter (default: 1000).
    """
    # Loop over each letter directory from A to Z
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        input_directory = os.path.join(input_base_directory, letter)
        output_directory = input_directory  # Output in the same directory

        # Ensure the input directory exists
        if not os.path.exists(input_directory):
            print(f"Directory {input_directory} does not exist, skipping.")
            continue

        # Initialize Augmentor pipeline for the current letter directory
        p = Augmentor.Pipeline(input_directory)

        # Define augmentation operations
        p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
        p.flip_left_right(probability=0.5)
        p.random_contrast(probability=0.5, min_factor=0.7, max_factor=1.3)

        # Generate augmented images
        p.sample(num_samples)

        # Augmentor saves images in a subdirectory of the input directory by default
        augmented_subdir = os.path.join(input_directory, 'output')
        augmented_images = os.listdir(augmented_subdir)

        # Move augmented images to the output directory
        for img_name in augmented_images:
            src = os.path.join(augmented_subdir, img_name)
            dst = os.path.join(output_directory, img_name)
            os.rename(src, dst)

        # Clean up the temporary directory created by Augmentor
        os.rmdir(augmented_subdir)
        print(f"Augmented images for {letter} saved in {output_directory}")


# Example usage
input_base_dir = 'Data_BSL/'  # Base directory containing letter subdirectories
augment_images_for_alphabet(input_base_dir, num_samples=1000)  # Augment images for each letter
