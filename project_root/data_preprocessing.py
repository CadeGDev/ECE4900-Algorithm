# Import statements
import torch
import torchvision.transforms.v2 as transforms  # For data preprocessing
from torch.utils.data import DataLoader, TensorDataset, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split # To install run "pip install scikit-learn" in terminal
import numpy as np
import os
import pandas as pd
from algorithm_model import config, Algorithm # Import hyperparameter values

import numpy as np
import matplotlib.pyplot as plt

import torch
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


class TransformationPipeline:
    """
    A class for applying a series of transformations to an image, tailored for preparing data for neural network input.
    
    Attributes:
        crop_box (tuple): Coordinates for cropping the image.
        resize_dims (tuple): Dimensions to resize the image to after cropping.
        threshold (float): Threshold value for applying a binary mask.
    
    Methods:
        transform(img): Applies cropping, conversion to tensor, resizing, and normalization to an image.
        apply_binary_mask(tensor, threshold): Applies a threshold-based binary mask to the tensor.
    """
    def __init__(self, crop_box=(100, 50, 700, 585), resize_dims=(600, 535), threshold=0.5):
        self.crop_box = crop_box
        self.resize_dims = resize_dims
        self.threshold = threshold

    def transform(self, img):
        """
        Transforms an image using predefined settings including cropping, tensor conversion, resizing, and normalization.
        
        Args:
            img (PIL.Image): The image to transform.
        
        Returns:
            torch.Tensor: The transformed image as a tensor.
        """
        if self.crop_box is not None:
            # img = TF.crop(img, *self.crop_box)
            img = TF.crop(img, 50, 100, 535, 600)
        img = TF.to_tensor(img)
        # img = self.apply_binary_mask(img, self.threshold)
        img = TF.resize(img, self.resize_dims)
        # img = TF.resize(img, (600,535))
        img = TF.rgb_to_grayscale(img)
        img = TF.normalize(img, 0, 1)
        return img

    @staticmethod
    def apply_binary_mask(tensor, threshold):
        """
        Applies a binary mask to a tensor based on a threshold, setting values above the threshold to 1 and others to 0.
        
        Args:
            tensor (torch.Tensor): The input tensor.
            threshold (float): The threshold value.
        
        Returns:
            torch.Tensor: The masked tensor.
        """
        return torch.where(tensor > threshold, torch.ones_like(tensor), torch.zeros_like(tensor))

class DatasetPreprocessor:
    """
    A class to handle the preprocessing of a dataset defined by a CSV file, which includes splitting, processing images,
    and saving the processed data for training or testing.
    
    Attributes:
        csv_file (str): Path to the CSV file containing image filenames and labels.
        root_dir (str): Root directory path where the images are stored.
        output_file (str): Path where the processed dataset will be saved.
        subset (str): Specifies whether to process 'train' or 'test' subset.
        test_size (float): Fraction of the dataset to be reserved as test set.
        random_state (int): Seed for random operations to ensure reproducibility.
        crop_box, resize_dims, threshold: Parameters for image transformation.
    
    Methods:
        _split_dataset(): Splits the dataset into training and testing subsets.
        process_and_save(): Processes the images and saves them along with labels in a dataset format.
    """
    def __init__(self, csv_file, root_dir, output_file, subset='train', test_size=0.2, random_state=42, crop_box=(100, 50, 700, 585), resize_dims=(600, 535), threshold=0.5):
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.output_file = output_file
        self.subset = subset
        self.test_size = test_size
        self.random_state = random_state
        self.crop_box = crop_box
        self.resize_dims = resize_dims
        self.threshold = threshold
        self.transformation_pipeline = TransformationPipeline(crop_box, resize_dims, threshold)
        self.labels_frame = pd.read_csv(csv_file)
        self._split_dataset()

    def _split_dataset(self):
        """
        Splits the dataset into training and testing indices based on the test_size and random_state attributes.
        """
        indices = range(len(self.labels_frame))
        train_indices, test_indices = train_test_split(indices, test_size=self.test_size, random_state=self.random_state)

        if self.subset == 'train':
            self.indices = train_indices
        elif self.subset == 'test':
            self.indices = test_indices
        else:
            raise ValueError("subset must be 'train' or 'test'")

    def process_and_save(self):
        """
        Processes each image specified in the subset of the dataset, applies transformations, and saves the data in a file.
        
        The method loads images according to the indices determined by the subset, applies transformations,
        and packages the images and their labels into a TensorDataset which is then saved to disk.
        """
        processed_images = []
        label_indices = []  # This will store the class indices instead of one-hot vectors
        
        num_bands = 50
        for i in self.indices:
            row = self.labels_frame.iloc[i]
            img_path = os.path.join(self.root_dir, row['Filename'])
            # Load and process image here, then append to processed_images and label_indices
            center_frequency = int(row['Frequency(MHz)']) - 1  # Adjusted for 0 indexing

            with Image.open(img_path) as img:
                transformed_img = self.transformation_pipeline.transform(img)
                processed_images.append(transformed_img)
            label_indices.append(center_frequency)  # Append the class index directly

        # Convert lists to tensors
        images_tensor = torch.stack(processed_images)

        # Convert label indices to a tensor directly
        labels_tensor = torch.tensor(label_indices, dtype=torch.long)  # Ensure the dtype is long for indices
        dataset = TensorDataset(images_tensor, labels_tensor)
        
        # Save the dataset
        torch.save(dataset, self.output_file)
        print(f"Dataset has been processed and saved to {self.output_file}.")