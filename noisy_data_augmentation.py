# necessary imports
import torch
from torchvision import datasets
from torchvision import transforms

import os
import cv2
from PIL import Image
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import random
#import helper

%matplotlib inline


# Define a function to add noise to an image
def add_noise(image, noise_level=0.1):
    noise = torch.randn_like(image) * noise_level
    noisy_image = image + noise
    return noisy_image

# Create a GPU device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a data transform to convert PIL images to tensors
transform = transforms.Compose([transforms.ToTensor()])


input_files = os.listdir(input_folder)

# Loop through the input images and process each one
for input_image_name in input_files:
    # Load the input image from the CPU
    input_image_path = os.path.join('clean_dataset/myfotos', input_image_name)
    input_image = Image.open(input_image_path).convert('RGB')

    # Apply the transform to convert the image to a tensor
    input_image = transform(input_image)

    # Transfer the input image to the GPU
    input_image = input_image.to(device)

    # Add noise to the image on the GPU
    noisy_image = add_noise(input_image,noise_level=random.random())

    # Transfer the noisy image back to the CPU
    noisy_image = noisy_image.to('cpu')

    # Save the noisy image to the output folder on the CPU
    noisy_image = noisy_image.clamp(0, 1)  # Ensure pixel values are in the [0, 1] range
    noisy_image = (noisy_image * 255).byte()  # Convert to byte format for saving as an image
    noisy_image = transforms.ToPILImage()(noisy_image)  # Convert to PIL image
    noisy_image.save(os.path.join(output_folder, input_image_name))
 
 
 