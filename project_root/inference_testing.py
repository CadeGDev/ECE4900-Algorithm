import torch
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF
from algorithm_model import Algorithm, config

# Define the preprocessing transformation
class TransformationPipeline:
    def __init__(self, crop_box=(100, 50, 700, 585), resize_dims=(600, 535), threshold=0.5):
        self.crop_box = crop_box
        self.resize_dims = resize_dims
        self.threshold = threshold

    def transform(self, img):
        if self.crop_box is not None:
            img = TF.crop(img, 50, 100, 535, 600)
        img = TF.to_tensor(img)
        img = self.apply_binary_mask(img, self.threshold)
        img = TF.resize(img, self.resize_dims)
        img = TF.rgb_to_grayscale(img)
        return img

    @staticmethod
    def apply_binary_mask(tensor, threshold):
        return torch.where(tensor > threshold, torch.ones_like(tensor), torch.zeros_like(tensor)) 

device = torch.device('cuda')
# Load the trained model
model = Algorithm(config['input_size'], config['hidden_size'], config['output_size'], config['num_hidden_layers'])
## model.load_state_dict(torch.load("/Users/cadeglauser/VSCose2Projects/ECE4900-Algorithm/TrainingModelV1.pth", map_location=torch.device('cpu')))
model.load_state_dict(torch.load("/home/l3harris-clinic/Algorithm/TrainingModelV2Cont.pt", map_location=device))
model.eval()
##model = torch.load("./TrainingModelV2Cont.pt")
##model.eval()
##model.to(device)
# Initialize the transformation pipeline
transformation_pipeline = TransformationPipeline()

# Load the spectrogram image file
spectrogram_image_path = '/home/l3harris-clinic/Algorithm/spectrogram11.png'
spectrogram_image = Image.open(spectrogram_image_path)
#spectrogram_image = spectrogram_image.to(device)
#spectrogram_image = Image.open(spectrogram_image_path).cuda()

# Apply transformations
transformed_image = transformation_pipeline.transform(spectrogram_image)

# Make sure to remove the extra leading dimension if present (for grayscale images)
if transformed_image.dim() > 3 and transformed_image.shape[0] == 1:
    transformed_image = transformed_image.squeeze(0)

# Add a batch dimension
input_tensor = transformed_image.unsqueeze(0)
#input_tensor = input_tensor.to(device)
# Perform inference
with torch.no_grad():
    output = model(input_tensor)

# Assuming a single class prediction for each input
predicted_class = torch.argmax(output, dim=1).item()

print(f"Predicted class: {predicted_class}")
