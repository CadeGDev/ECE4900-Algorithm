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
        output_size = 50  # There are 50 separate bands to classify

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)  # Max pooling over a (2,2) window
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Dynamically compute the flattened size after convolutions
        self._to_linear = None
        self._compute_flat_size(input_size)

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)])
        self.output = nn.Linear(hidden_size, output_size)

    def _compute_flat_size(self, input_size):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_size[0], input_size[1])  # use the actual image size here
            output = self._conv_output_size(dummy_input)
            self._to_linear = output.numel()  # compute the total number of resulting features

    def _conv_output_size(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

    def forward(self, x):
        x = self._conv_output_size(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output(x)
        return torch.sigmoid(x)  # Use sigmoid for multi-label classification

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.01)

# Update the config
config = {
    "input_size": (600, 585),  # Images are 600x585 pixels
    "hidden_size": 100,
    "output_size": 50,  # 50 separate bands to classify
    "num_hidden_layers": 5,
    "learning_rate": 0.01,
    "num_epochs": 5,
    "batch_size": 20,
    "num_workers": 4
}


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