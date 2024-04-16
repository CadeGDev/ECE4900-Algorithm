import torch
from torchvision import transforms
from algorithm_model import Algorithm, config # replace with your actual model class
from PIL import Image
import torchvision.transforms.functional as TF

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Model will use: {device}")

# Step 1: Load the trained model
model = Algorithm(config['input_size'], config['hidden_size'], config['output_size'], config['num_hidden_layers']) # initialize model architecture

# Load the model weights onto the CPU (only once and with the correct path)
model.load_state_dict(torch.load("/home/l3harris-clinic/Algorithm/TrainingModelV1.pth", map_location=device))

# Set the model to evaluation mode
model.eval()

#Transformations
class TransformationPipeline:
    def __init__(self, crop_box=(100, 50, 700, 585), resize_dims=(600, 585), threshold=0.5):
        self.crop_box = crop_box
        self.resize_dims = resize_dims
        self.threshold = threshold

    def transform(self, img):
        if self.crop_box is not None:
            img = TF.crop(img, *self.crop_box)
        img = TF.to_tensor(img)
        img = self.apply_binary_mask(img, self.threshold)
        img = TF.resize(img, self.resize_dims)
        img = TF.rgb_to_grayscale(img)
        return img

    @staticmethod
    def apply_binary_mask(tensor, threshold):
        return torch.where(tensor > threshold, torch.ones_like(tensor), torch.zeros_like(tensor))

# Initialize the transformation pipeline with desired parameters
transformation_pipeline = TransformationPipeline()

# Load your spectrogram image file
spectrogram_image_path = '/home/l3harris-clinic/Algorithm/spectrogram4993.png'
spectrogram_image = Image.open(spectrogram_image_path)

# Apply transformations
transformed_image = transformation_pipeline.transform(spectrogram_image)

# Remove the singleton dimension from the tensor
input_tensor = transformed_image.squeeze(0)

input_batch = input_tensor.unsqueeze(0) # add a batch dimension

# If working with GPU, move your model and input batch to GPU

# Step 3: Perform inference
with torch.no_grad():
    output = model(input_batch)

# Step 4: Process the output
# For a classification model, you may want to apply a softmax to the output
# and then use the argmax to get the most likely class label
probabilities = torch.nn.functional.softmax(output, dim=1)
predicted_label = torch.argmax(probabilities).item()
# If you have a mapping of class indices to class labels, you would use it here to get the label

print(f'Predicted label: {probabilities}') # or print the actual label if you have the mapping
