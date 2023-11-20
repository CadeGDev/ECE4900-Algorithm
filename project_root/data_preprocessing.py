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

# This is a example spectrogram generation

# # Load audio file
# y, sr = librosa.load('sir_duke_fast.ogg')

# # Compute spectrogram
# spec = librosa.feature.melspectrogram(y=y, sr=sr)

# # Convert power to decibels
# spec_db = librosa.power_to_db(spec, ref=np.max)

# # Plot spectrogram
# fig, ax = plt.subplots(nrows = 1, ncols = 1)
# img = librosa.display.specshow(spec_db, x_axis='time', y_axis='mel', ax = ax)
# fig.colorbar(img, ax = ax, format='%+2.0f dB')
# ax.set_title('Spectrogram')
# fig.show()

# # Save the figure as a TIFF file
# plt.savefig('spectrogram.tiff', format='tiff')


# Define the path to your spectrogram image
# TODO: define path to spectrogram images/dataset
image_path = 0
# Ex: image_path = "path/to/your/spectrogram_image.png"

# Initialize size variables
# TODO: Set default sizes for NN inputs based on spectrogram resolution
new_height = 256 # Placeholder height CHANGE AS NEEDED to spectrogram height
new_width = 256 # Placeholder width CHANGE AS NEEDED to spectrogram width

# Define the transformation pipeline
transform = transforms.Compose([
    # Random data augmentation (not sure if needed, example augments commented out for now)
    # transforms.RandomRotation(degrees=15),      # Data augmentation: random rotation
    # transforms.RandomHorizontalFlip(),         # Data augmentation: random horizontal flip
    transforms.Resize((new_height, new_width)), # Resize to a specific size
    transforms.Grayscale(num_output_channels=1),# Convert to grayscale if needed
    transforms.ToTensor(),                       # Convert the image to a PyTorch tensor
    # transforms.Normalize((0, 0, 0), (1, 1, 1))  # Normalize the image to have values in the range [0, 1]

])

# Load and preprocess the image
image = Image.open(image_path)
preprocessed_image = transform(image)
# Find values and normalize image between 0,1 (Probable does not work this way)
mean_value = preprocessed_image.mean()
std_value = preprocessed_image.std()
transforms.Normalize((mean_value,), (std_value,)) # Normalization if needed (Maybe put inside pipeline) (This also probably isnt correct)

# Add a batch dimension to the image if needed
preprocessed_image = preprocessed_image.unsqueeze(0)

# Now, preprocessed_image is a PyTorch tensor ready to be input into your neural network

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

