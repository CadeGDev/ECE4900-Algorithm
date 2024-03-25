# Import statements
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as transforms  # For data preprocessing
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from sklearn.model_selection import train_test_split # To install run "pip install scikit-learn" in terminal

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Define a custom transform function
class ResizeAndGrayscaleTransform:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def __call__(self, img):
        # Resize the image
        img = img.resize(self.target_size)

        # Convert to grayscale
        img = img.convert("L")

        return img
    
# Split dataset if the dataset is not already split
def split_dataset(data, labels):
    # Use train_test_split to split the dataset into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
    return train_data, val_data, train_labels, val_labels

def show_spectrogram_images(train_loader, n_images=4):
    """
    Displays a batch of spectrogram training images from the DataLoader.

    Args:
    - train_loader: DataLoader containing the spectrogram dataset.
    - n_images: Number of images to display (default is 4).
    """
    # Get a batch of training data
    images, _ = next(iter(train_loader))

    # Make a grid from the batch
    img_grid = torchvision.utils.make_grid(images[:n_images])

    plt.figure(figsize=(10, 10))
    # Convert the tensor to a format suitable for Matplotlib
    np_img = img_grid.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)), interpolation='nearest')
    plt.title("Sample Spectrogram Images")
    plt.axis('off')
    plt.show()

# Define The Transformation pipeline
transform = transforms.Compose([
    ResizeAndGrayscaleTransform(),  # Resize and convert to grayscale
    transforms.ToTensor(),          # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize (adjust mean and std as needed)
])

# Define the path to your spectrogram image
# TODO: define path to spectrogram images/dataset
image_path = "/Users/cadeglauser/VSCose2Projects/ECE4900-Algorithm/project_root/spectrogram2.tiff"
# Ex: image_path = "path/to/your/spectrogram_image.png"

# Initialize size variables
# TODO: Set default sizes for NN inputs based on spectrogram resolution

# Load and preprocess the image
image = Image.open(image_path)
preprocessed_image = transform(image)

# Add a batch dimension to the image if needed
# preprocessed_image = preprocessed_image.unsqueeze(0)

