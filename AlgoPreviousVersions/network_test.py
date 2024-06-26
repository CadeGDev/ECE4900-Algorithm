import torch
from PIL import Image
from data_preprocessing import preprocessing_transforms as transform
from algorithm_model import Algorithm_v0_1

def load_image(image_path):
    # Load an image and apply preprocessing transformations
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def test_network(image_path, model):
    # Load and preprocess the image
    image = load_image(image_path)

    # Run the image through the network
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output = model(image)

    # Print or process the output
    print("Network output:", output)

# Path to a sample spectrogram image
sample_image_path = "project_root/spectrogram2.tiff"

# Initialize the neural network
input_size = 224
hidden_size = 128
output_size = 10 
num_hidden_layers = 2 

model = Algorithm_v0_1(input_size, hidden_size, output_size, num_hidden_layers)

# Test the network with a sample image
test_network(sample_image_path, model)
