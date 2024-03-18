# Import statements
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as transforms  # For data preprocessing
import torchvision.transforms.functional as F
import torchvision
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split # To install run "pip install scikit-learn" in terminal
import numpy as np
import os
import pandas as pd

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF

def load_image_as_tensor(image_path):
    """Load a PNG image and convert it to a PyTorch tensor."""
    img = Image.open(image_path)
    tensor = transforms.ToTensor()(img)
    return tensor

# Define custom transformations as callable classes (if needed)
class BinaryMaskTransform:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def __call__(self, tensor):
        mask = tensor > self.threshold
        return torch.where(mask, tensor, torch.zeros_like(tensor))

class ResizeAndGrayscaleTransform:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def __call__(self, tensor):
        resized_tensor = TF.resize(tensor, self.target_size)
        grayscale_tensor = TF.rgb_to_grayscale(resized_tensor, num_output_channels=1)
        return grayscale_tensor

def apply_binary_mask(tensor, threshold=0.5):
    """Apply a binary mask to a tensor based on a threshold."""
    mask = tensor > threshold
    masked_tensor = torch.where(mask, tensor, torch.zeros_like(tensor))
    return masked_tensor

def resize_and_grayscale(tensor, target_size=(224, 224)):
    """Resize and convert a tensor to grayscale."""
    resized_tensor = TF.resize(tensor, target_size)
    grayscale_tensor = TF.rgb_to_grayscale(resized_tensor, num_output_channels=1)
    return grayscale_tensor

def preprocess_image(image_path, threshold=0.5, target_size=(224, 224)):
    """
    Complete preprocessing pipeline for an image file.
    Includes loading, binary masking, and transformations.
    """
    # Load the image and convert to tensor
    tensor = load_image_as_tensor(image_path)
    
    # Apply binary mask
    masked_tensor = apply_binary_mask(tensor, threshold=threshold)
    
    # Apply resizing and grayscaling
    final_tensor = resize_and_grayscale(masked_tensor, target_size=target_size)
    
    return final_tensor



class SpectrogramDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            root_dir (string): Directory with all the spectrograms.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.spectrogram_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.spectrogram_frame)

    def __getitem__(self, idx):
        # Get the image file path from the 'Filename' column
        
        img_name = os.path.join(self.root_dir, self.spectrogram_frame.iloc[idx]['Filename'])
        image = Image.open(img_name)

        # Get the labels from the other columns
        labels = self.spectrogram_frame.iloc[idx][1:].to_dict()  # Excludes the 'Filename' column

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Will probably convert the labels to a tensor, ex:
        # labels = torch.tensor(list(labels.values()))

        return image, labels


batch_size = 32
shuffle_dataset = True
num_workers = 4


# Combine into a single transform pipeline
preprocessing_transforms = transforms.Compose([
    transforms.Lambda(lambda image_path: load_image_as_tensor(image_path)),
    BinaryMaskTransform(threshold=0.5),
    ResizeAndGrayscaleTransform(target_size=(224, 224)),
    # Add any other transformations here
])

spectrogram_dataset = SpectrogramDataset(
    csv_file='project_root/Continuous/spectrogram_labels.csv',
    root_dir='Continuous',  # Replace with the correct path to your images directory
    transform=preprocessing_transforms
)

from torch.utils.data import DataLoader

dataloader = DataLoader(spectrogram_dataset, batch_size, shuffle=True)

    
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


# Define the path to spectrogram image
# TODO: define path to spectrogram images/dataset
image_path = "/Users/cadeglauser/VSCose2Projects/ECE4900-Algorithm/project_root/Continuous"

# Initialize size variables
# TODO: Set default sizes for NN inputs based on spectrogram resolution


# Add a batch dimension to the image if needed
# preprocessed_image = preprocessed_image.unsqueeze(0)
