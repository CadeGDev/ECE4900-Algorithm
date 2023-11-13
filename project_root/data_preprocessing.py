# Import statements
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as transforms  # For data preprocessing
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from sklearn.model_selection import train_test_split # To install run "pip install scikit-learn" in terminal

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


