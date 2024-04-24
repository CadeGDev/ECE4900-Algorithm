# Import statements
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms  # For data preprocessing
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


# Define neural network architecture

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B X N X C
        key = self.key_conv(x).view(batch_size, -1, height * width)  # B X C X N
        value = self.value_conv(x).view(batch_size, -1, height * width)  # B X C X N

        attention = self.softmax(torch.bmm(query, key))  # B X N X N
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)

        return out


class Algorithm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super(Algorithm, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=1)
        self.attention = SelfAttention(32)  # Adding the self-attention layer

        # Rest of the model architecture remains the same
        self._to_linear = None
        self._compute_flat_size(input_size)
        self.fc1 = nn.Linear(self._to_linear, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)])
        self.output = nn.Linear(hidden_size, output_size)

    def _compute_flat_size(self, input_size):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_size[0], input_size[1])
            output = self._conv_output_size(dummy_input)
            self._to_linear = output.numel()

    def _conv_output_size(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.attention(x)  # Apply attention
        return x

    def forward(self, x):
        # Apply the convolutional and attention layers
        x = self._conv_output_size(x)  # Includes convolution and attention layers
        x = x.view(-1, self._to_linear)  # Flatten the output for the fully connected layer
        
        # Apply the first fully connected layer and activation
        x = F.relu(self.fc1(x))
        
        # Process through the hidden layers
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        # Final output layer
        x = self.output(x)
        return torch.sigmoid(x)  # For multi-label classification

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
    "num_workers": 0
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
