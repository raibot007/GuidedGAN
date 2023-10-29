import cv2
import os
import random
import numpy as np

# Path to the folder containing your original images
input_folder = "path_to_input_folder"

# Path to the folder where augmented images will be saved
output_folder = "path_to_output_folder"

# Desired number of images in the output folder (10,000 in your case)
desired_count = 10000

# List of augmentation functions
augmentation_functions = [
    # You can add more augmentation functions here
    lambda img: cv2.flip(img, 1),  # Horizontal flip
    lambda img: cv2.flip(img, 0),
    lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),  # Rotate 90 degrees clockwise
    lambda img: cv2.rotate(img, random.randint(0, 2)),
    lambda img: img[:random.randint(0, img.shape[0]), :random.randint(0, img.shape[1])],
    lambda img: cv2.resize(img, (int(img.shape[1]*0.75), int(img.shape[0]*0.75))),
    lambda img: cv2.convertScaleAbs(img, alpha=0.5, beta=0),
    lambda img: cv2.convertScaleAbs(img, alpha=1.5, beta=0)
]

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all the files in the input folder
input_files = os.listdir(input_folder)

# Calculate how many times each image needs to be augmented to reach the desired count
num_images_to_generate = (desired_count // len(input_files)) + 1

# Loop through each image in the input folder
for filename in input_files:
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)

    # Apply random augmentations to create additional images
    for i in range(num_images_to_generate):
        augmentation_function = random.choice(augmentation_functions)
        augmented_img = augmentation_function(img)

        if 0 not in augmented_img.shape:
            # Save the augmented image to the output folder
            output_filename = f"{os.path.splitext(filename)[0]}_{i}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, augmented_img)

print(f"{desired_count} images have been generated in {output_folder}.")