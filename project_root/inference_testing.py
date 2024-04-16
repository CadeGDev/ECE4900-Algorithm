import torch
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF
from algorithm_model import Algorithm, config

class InferenceTester:
    def __init__(self, model_path, image_path):
        self.model_path = model_path
        self.image_path = image_path
        self.model = self.load_model()
        self.transformation_pipeline = TransformationPipeline()

    def load_model(self):
        model = Algorithm(config['input_size'], config['hidden_size'], config['output_size'], config['num_hidden_layers'])
        model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        model.eval()
        return model

    def perform_inference(self):
        image = Image.open(self.image_path)
        transformed_image = self.transformation_pipeline.transform(image)
        if transformed_image.dim() > 3 and transformed_image.shape[0] == 1:
            transformed_image = transformed_image.squeeze(0)
        input_tensor = transformed_image.unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        return predicted_class

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

# Example usage
model_file_path = "./TrainingModelV1.pt"
spectrogram_image_path = '/path/to/spectrogram_image.png'
tester = InferenceTester(model_file_path, spectrogram_image_path)
predicted_class = tester.perform_inference()
print(f'Predicted class: {predicted_class}')




# # Define the preprocessing transformation
# class TransformationPipeline:
#     def __init__(self, crop_box=(100, 50, 700, 585), resize_dims=(600, 535), threshold=0.5):
#         self.crop_box = crop_box
#         self.resize_dims = resize_dims
#         self.threshold = threshold

#     def transform(self, img):
#         if self.crop_box is not None:
#             img = TF.crop(img, 50, 100, 535, 600)
#         img = TF.to_tensor(img)
#         img = self.apply_binary_mask(img, self.threshold)
#         img = TF.resize(img, self.resize_dims)
#         img = TF.rgb_to_grayscale(img)
#         return img

#     @staticmethod
#     def apply_binary_mask(tensor, threshold):
#         return torch.where(tensor > threshold, torch.ones_like(tensor), torch.zeros_like(tensor)) 

# # Load the trained model
# model = Algorithm(config['input_size'], config['hidden_size'], config['output_size'], config['num_hidden_layers'])
# # model.load_state_dict(torch.load("/Users/cadeglauser/VSCose2Projects/ECE4900-Algorithm/TrainingModelV1.pth", map_location=torch.device('cpu')))
# model.load_state_dict(torch.load("./TrainingModelV1.pt"))
# model.eval()

# # Initialize the transformation pipeline
# transformation_pipeline = TransformationPipeline()

# # Load the spectrogram image file
# spectrogram_image_path = '/Users/cadeglauser/VSCose2Projects/ECE4900-Algorithm/project_root/Continuous/spectrogram4873.png'
# spectrogram_image = Image.open(spectrogram_image_path)

# # Apply transformations
# transformed_image = transformation_pipeline.transform(spectrogram_image)

# # Make sure to remove the extra leading dimension if present (for grayscale images)
# if transformed_image.dim() > 3 and transformed_image.shape[0] == 1:
#     transformed_image = transformed_image.squeeze(0)

# # Add a batch dimension
# input_tensor = transformed_image.unsqueeze(0)

# # Perform inference
# with torch.no_grad():
#     output = model(input_tensor)

# # Assuming a single class prediction for each input
# predicted_class = torch.argmax(output, dim=1).item()

# print(f'Predicted class: {predicted_class}')
