# Import statements
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms  # For data preprocessing
from torch.utils.data import DataLoader, TensorDataset
from algorithm_model import config, Algorithm_v0_1 # Import hyperparameter values
from data_preprocessing import SpectrogramDataset
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split # To install run "pip install scikit-learn" in terminal


# Define the path to dataset and labels
image_folder = 'project_root/Continuous'
labels_csv = 'project_root/Continuous/spectrogram_labels.csv'

# Extract hyperparameters
input_size = config["input_size"]
hidden_size = config["hidden_size"]
output_size = config["output_size"]
num_hidden_layers = config["num_hidden_layers"]
learning_rate = config["learning_rate"]
num_epochs = config["num_epochs"]
batch_size = config["batch_size"]

num_workers = 4  # Number of subprocesses for data loading

# Initialize the dataset and data loader
train_dataset = SpectrogramDataset(csv_file='path/to/labels.csv', root_dir='path/to/images', subset='train')
test_dataset = SpectrogramDataset(csv_file='path/to/labels.csv', root_dir='path/to/images', subset='test')

# dataset = SpectrogramDataset(csv_file=labels_csv, root_dir=image_folder)
dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# Create instance of algorithm model
model = Algorithm_v0_1(input_size, hidden_size, output_size, num_hidden_layers)

# Apply the custom weight initialization function to the model
model.init_weights()

# Define loss function and optimizer ### EDIT LATER
criterion = nn.MSELoss()  # Example loss function for regression
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training

# Training Loop ### JUST EXAMPLE, EDIT LATER
for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader_train):
        images, labels = batch['image'], batch['label']
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader_train)}], Loss: {loss.item():.4f}')

# Add validation/testing loop