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
from algorithm_model import config, Algorithm_v0_1 # Import hyperparameter values

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF

# Extract hyperparameters
input_size = config["input_size"]
hidden_size = config["hidden_size"]
output_size = config["output_size"]
num_hidden_layers = config["num_hidden_layers"]
learning_rate = config["learning_rate"]
num_epochs = config["num_epochs"]
batch_size = config["batch_size"]


class SpectrogramDataset(Dataset):
    def __init__(self, csv_file, root_dir, resize_dims=(600, 585), threshold=0.5, subset='train', test_size=0.2, random_state=42):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            root_dir (string): Directory with all the spectrograms.
            subset (string): 'train' or 'test' to specify which subset to load.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random state for reproducible splits.
        """
        self.labels_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.subset = subset
        
        # Split the indices into training and testing subsets
        indices = range(len(self.labels_frame))
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )

        # Select the indices for the requested subset
        if subset == 'train':
            self.indices = train_indices
        elif subset == 'test':
            self.indices = test_indices
        else:
            raise ValueError("subset must be 'train' or 'test'")
        
        # Transformation Pipeline
        self.transformation_pipeline = TransformationPipeline(resize_dims=resize_dims, threshold=threshold)

    def __len__(self):
        return len(self.labels_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        true_idx = self.indices[idx]
        img_name = os.path.join(self.root_dir, self.labels_frame.iloc[idx, 0])
        label = self.labels_frame.iloc[true_idx, 1]
        # Use the transformation_pipeline on image
        image = self.transformation_pipeline.transform(img_name)
        sample = {'image': image, 'label': label}

        return sample

class TransformationPipeline:
    def __init__(self, crop_box=(100, 50, 700, 585), resize_dims=(600, 585), threshold=0.5):
        """
        Initialize the transformation pipeline with cropping, resizing, and thresholding.
        
        :param crop_box: The coordinates (left, upper, right, lower) for the crop or None if no cropping is needed.
        :param resize_dims: Tuple of (width, height) for the resize dimensions.
        :param threshold: Float value for the binary mask threshold.
        """
        self.crop_box = crop_box
        self.resize_dims = resize_dims
        self.threshold = threshold
        # The order of transforms is cropping -> to tensor -> mask -> resize and grayscale
        # Only add the crop transform if a crop_box has been specified
        transforms_list = []
        if crop_box is not None:
            transforms_list.append(transforms.Lambda(lambda img: TF.crop(img, *crop_box)))
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: self.apply_binary_mask(x, self.threshold)),
            transforms.Lambda(lambda x: self.resize_and_greyscale(x, self.resize_dims))
        ])
        self.transform = transforms.Compose(transforms_list)

    @staticmethod
    def load_image_as_tensor(image_path):
        """Load a PNG image and convert it to a PyTorch tensor."""
        image = Image.open(image_path).convert('RGB')
        return transforms.ToTensor()(image)

    @staticmethod
    def apply_binary_mask(tensor, threshold=0.5):
        """Apply a binary mask to a tensor based on a threshold value."""
        return torch.where(tensor > threshold, torch.ones_like(tensor), torch.zeros_like(tensor))

    def resize_and_greyscale(self, tensor, resize_dims):
        """
        Resize the tensor to the specified dimensions and convert it to greyscale.
        This method assumes a certain functionality based on its name.
        """
        tensor_resized = F.resize(tensor, resize_dims)
        return F.rgb_to_grayscale(tensor_resized)


# batch_size = 32
# shuffle_dataset = True
# num_workers = 4

# # Split dataset if the dataset is not already split
# def split_dataset(data, labels):
#     # Use train_test_split to split the dataset into training and validation sets
#     train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
#     return train_data, val_data, train_labels, val_labels

# def show_spectrogram_images(train_loader, n_images=4):
#     """
#     Displays a batch of spectrogram training images from the DataLoader.

#     Args:
#     - train_loader: DataLoader containing the spectrogram dataset.
#     - n_images: Number of images to display (default is 4).
#     """
#     # Get a batch of training data
#     images, _ = next(iter(train_loader))

#     # Make a grid from the batch
#     img_grid = torchvision.utils.make_grid(images[:n_images])

#     plt.figure(figsize=(10, 10))
#     # Convert the tensor to a format suitable for Matplotlib
#     np_img = img_grid.numpy()
#     plt.imshow(np.transpose(np_img, (1, 2, 0)), interpolation='nearest')
#     plt.title("Sample Spectrogram Images")
#     plt.axis('off')
#     plt.show()


# Define the path to spectrogram image
# TODO: define path to spectrogram images/dataset
# image_path = "/Users/cadeglauser/VSCose2Projects/ECE4900-Algorithm/project_root/Continuous"

# Initialize size variables
# TODO: Set default sizes for NN inputs based on spectrogram resolution

# Add a batch dimension to the image if needed
# preprocessed_image = preprocessed_image.unsqueeze(0)
