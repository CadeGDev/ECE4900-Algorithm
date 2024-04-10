import torch
import torchvision.transforms.v2 as transforms  # For data preprocessing
from torch.utils.data import DataLoader, TensorDataset, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split # To install run "pip install scikit-learn" in terminal
import numpy as np
import os
import pandas as pd
from algorithm_model import config, Algorithm_v0_1 # Import hyperparameter values

import numpy as np
import matplotlib.pyplot as plt

import torch
from PIL import Image
import torchvision.transforms.functional as TF
from algorithm_model import Algorithm_v0_1

def load_image(image_path):
    crop_box=(100, 50, 700, 585)
    resize_dims=(600, 585)
    threshold=0.5
    # Load an image and apply preprocessing transformations
    image = Image.open(image_path)
    image = TF.to_tensor(image)
    image = apply_binary_mask(image, threshold)
    image = TF.resize(image, resize_dims)
    image = TF.rgb_to_grayscale(image)
    return image

def apply_binary_mask(tensor, threshold):
        return torch.where(tensor > threshold, torch.ones_like(tensor), torch.zeros_like(tensor))

def test_network(image_path, model):
    # Load and preprocess the image
    image = load_image(image_path)
    
    model.load_state_dict(torch.load('ECE4900-Algorithm\project_root\models\TrainedModelV1.pth'))
    # Run the image through the network
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output = model(image)

    # Print or process the output
    print("Network output:", output)

# Path to a sample spectrogram image
sample_image_path = 'ECE4900-Algorithm\project_root\data\test_image.png'

# Initialize the neural network
input_size = 1427, 858
hidden_size = 128
output_size = 10 
num_hidden_layers = 2 

model = Algorithm_v0_1(input_size, hidden_size, output_size, num_hidden_layers)
# Test the network with a sample image
test_network(sample_image_path, model)
