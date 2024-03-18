# Import statements
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms  # For data preprocessing
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


# Define neural network architecture
class Algorithm_v0_1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super(Algorithm_v0_1, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.pool = nn.MaxPool2d(2) # Max pooling over (2,2) window
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)

        # Dropout layer after convolutional layers
        # self.dropout_conv = nn.Dropout(0.5)

        # Dummy forward pass to determine size of convolutional output
        dummy_input = torch.autograd.Variable(torch.zeros(1, 1, input_size, input_size))
        output = self.conv_forward(dummy_input)
        adjusted_output_size = output.view(-1).shape[0]

        self.fc1 = nn.Linear(adjusted_output_size, 128)  # Adjust the input size as needed

        # Dropout layer before the fully connected layer
        # self.dropout_fc = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, output_size) # Num classes (30 now but likely changing to 50)

        self.hidden_layers = nn.ModuleList()

        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_layers.append(nn.ReLU())

        # Output layer
        self.output = nn.Linear(hidden_size, output_size)

    def conv_forward(self, x):
        # Forward pass through convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.dropout_conv(x)
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.dropout_conv(x)
        return x

    def forward(self, x):
        x = self.conv_forward(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x)) # Apply sigmoid activation function at the end

        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.dropout_conv(x)
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.dropout_conv(x)
        
        # # Flatten x for the fully connected layers
        # x = x.view(-1, self.num_flat_features(x))
        
        # x = F.relu(self.fc1(x))
        # x = self.dropout_fc(x)
        # x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # All dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    # Initialize weights
    def init_weights(self):
        for m in self.modules:
        # Using He initialization for now (Good for ReLU activations)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

# Define hyperparameters
input_size = 10000 # = num of samples per time step * num of samples (Fs * num samples)
output_size = 30 # if classificatio: = num of classifications, or if regression: = 1 (maybe multi-dimensional output) 
hidden_size = 100
num_hidden_layers = 5

# Create config variables inherited by other classes
config = {
    "input_size": 10000,
    "hidden_size": 100,
    "output_size": 30,
    "num_hidden_layers": 5,
    "learning_rate": 0.001,
    "num_epochs": 50,
    "batch_size": 32
}

# Extract hyperparameters
input_size = config["input_size"]
hidden_size = config["hidden_size"]
output_size = config["output_size"]
num_hidden_layers = config["num_hidden_layers"]
learning_rate = config["learning_rate"]
num_epochs = config["num_epochs"]
batch_size = config["batch_size"]


#  LATER! - Spectral flux density layer
# class SpectralFluxLayer(nn.Module):
#     def __init__(self):
#         super(SpectralFluxLayer, self).__init__()

#     def forward(self, spectrogram):
#         """
#         Compute the spectral flux as part of the forward pass.

#         Args:
#         - spectrogram: A PyTorch tensor of shape (batch_size, channels, freq_bins, time_steps).

#         Returns:
#         - spectral_flux: A PyTorch tensor of shape (batch_size, channels, 1, time_steps) representing the spectral flux for each time step.
#         """
#         # Calculate squared difference between adjacent time steps
#         flux = torch.sum((torch.diff(spectrogram, dim=-1) ** 2), dim=2, keepdim=True)
        
#         # Take the square root to get the final spectral flux
#         spectral_flux = torch.sqrt(flux)
        
#         return spectral_flux
