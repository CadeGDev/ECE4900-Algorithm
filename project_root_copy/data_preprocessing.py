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
    def __init__(self, crop_box=(100, 50, 700, 585), resize_dims=(600, 535), threshold=0.5):
        self.crop_box = crop_box
        self.resize_dims = resize_dims
        self.threshold = threshold

    def transform(self, img):
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
        return torch.where(tensor > threshold, torch.ones_like(tensor), torch.zeros_like(tensor))

class DatasetPreprocessor:
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
        indices = range(len(self.labels_frame))
        train_indices, test_indices = train_test_split(indices, test_size=self.test_size, random_state=self.random_state)

        if self.subset == 'train':
            self.indices = train_indices
        elif self.subset == 'test':
            self.indices = test_indices
        else:
            raise ValueError("subset must be 'train' or 'test'")

    def preprocess_and_save(self):
        labels_frame = pd.read_csv(self.csv_file)
        processed_images = []
        label_vectors = []  # Store  one-hot encoded labels

        num_bands = 50  # Number of separate bands
        for _, row in labels_frame.iterrows():
            img_path = os.path.join(self.root_dir, row['Filename'])
            center_frequency = int(row['Frequency(MHz)']) - 1  # Adjusted for 0 indexing

            # Initialize the label vector with ones (assuming whitespace)
            one_hot_label = np.zeros(num_bands)

            # Mark the center and adjacent bands as signal (not whitespace)
            one_hot_label[center_frequency] = 1

            with Image.open(img_path) as img:
                transformed_img = self.transformation_pipeline.transform(img)
                processed_images.append(transformed_img)
            label_vectors.append(one_hot_label)  # Append the one-hot encoded label vector

        # Convert lists to tensors
        images_tensor = torch.stack(processed_images)
        labels_tensor = torch.tensor(label_vectors, dtype=torch.float)

        # Create a TensorDataset
        dataset = TensorDataset(images_tensor, labels_tensor)

        # Convert lists to tensors
        images_tensor = torch.stack(processed_images)
        labels_tensor = torch.tensor(label_vectors, dtype=torch.float)

        # Check the shape of the labels tensor
        print(labels_tensor.shape)  # This should output something like torch.Size([num_samples, 50])

        # If the shape is correct, then proceed to save
        dataset = TensorDataset(images_tensor, labels_tensor)
        torch.save(dataset, self.output_file)

        # # Serialize the dataset to a file
        # torch.save(dataset, self.output_file)
        print(f"Dataset has been processed and saved to {self.output_file}.")

