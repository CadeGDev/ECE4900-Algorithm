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
def init_weights(m):
    # Using He initialization for now (Good for ReLU activations)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

# Define hyperparameters
input_size = 10000 # = num of samples per time step * num of samples (Fs * num samples)
output_size = 1 # if classificatio: = num of classifications, or if regression: = 1 (maybe multi-dimensional output) 
hidden_size = 100
num_hidden_layers = 5

# Create instance of model
model = Algorithm_v0_1(input_size, hidden_size, output_size, num_hidden_layers)

# Apply weights initialization
model.apply(init_weights)

# Define loss function and optimizer ### EDIT LATER
criterion = nn.MSELoss()  # Example loss function for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load and preprocess your RF spectrum data
# Split it into training, validation, and test sets

# Training
# Variable initializations
num_epochs = 50  # Adjust

# Dataset initialization
train_inputs = 0
train_labels = 0

# Assuming you have your data as tensors
train_data = TensorDataset(train_inputs, train_labels)
batch_size = 32  # Adjust the batch size as needed

# Training loop
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
for epoch in range(num_epochs):
    for data in train_loader:  # Iterate through training data
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    # Validation and monitoring during training

# Evaluation on a test set

# Save or serialize your trained model
