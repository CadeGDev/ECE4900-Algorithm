# Import statements
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms  # For data preprocessing
from torch.utils.data import DataLoader, TensorDataset



# Define neural network architecture
class Algorithm_v0_1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super(Algorithm_v0_1, self).__init__()

        # Define layers here
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, output_size)

        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_layers.append(nn.ReLU())

    # Define the forward pass
    def forward(self, x):
        x = self.input_layer(x)
        
        for layer in self.hidden_layers:
            x = layer(x)
        
        x = self.output_layer(x)
        return x

    # Initialize weights
    def init_weights(self):
        for m in self.modules:
        # Using He initialization for now (Good for ReLU activations)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

# Define hyperparameters
input_size = 10000 # = num of samples per time step * num of samples (Fs * num samples)
output_size = 1 # if classificatio: = num of classifications, or if regression: = 1 (maybe multi-dimensional output) 
hidden_size = 100
num_hidden_layers = 5

# Create config variables inherited by other classes
config = {
    "input_size": 10000,
    "hidden_size": 100,
    "output_size": 1,
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



