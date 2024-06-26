import torch
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF
from algorithm_model import Algorithm, config
import pandas as pd

class InferenceTester:
    """
    A class to handle the loading of a neural network model and perform inference on given images.

    Attributes:
        model_path (str): Path to the saved model file.
        model (torch.nn.Module): Loaded neural network model.

    Methods:
        load_model(): Loads the neural network model from the specified file.
        perform_inference(image_path): Processes an image and performs inference using the loaded model.
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()
        self.transformation_pipeline = TransformationPipeline()

    def load_model(self):
        """Loads the model from the specified path and prepares it for inference."""
        model = Algorithm(config['input_size'], config['hidden_size'], config['output_size'], config['num_hidden_layers'])
        model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        model.eval()
        return model

    def perform_inference(self, image_path):
        """
        Opens an image, applies necessary transformations, and performs inference.

        Args:
            image_path (str): Path to the image file.

        Returns:
            int: Predicted class index as an integer.
        """
        image = Image.open(image_path)
        transformed_image = self.transformation_pipeline.transform(image)
        if transformed_image.dim() > 3 and transformed_image.shape[0] == 1:
            transformed_image = transformed_image.squeeze(0)
        input_tensor = transformed_image.unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        return predicted_class

class TransformationPipeline:
    """
    A class to handle image transformations for preprocessing before model inference.

    Attributes:
        crop_box (tuple): Coordinates for cropping the image.
        resize_dims (tuple): Dimensions to resize the image to after cropping.
        threshold (float): Threshold for binary mask application.

    Methods:
        transform(img): Applies the transformation pipeline to an image.
        apply_binary_mask(tensor, threshold): Applies a binary mask based on the given threshold.
    """
    def __init__(self, crop_box=(100, 50, 700, 585), resize_dims=(600, 535), threshold=0.5):
        self.crop_box = crop_box
        self.resize_dims = resize_dims
        self.threshold = threshold

    def transform(self, img):
        """
        Transforms an image according to the defined pipeline settings.

        Args:
            img (PIL.Image.Image): Image to be transformed.

        Returns:
            torch.Tensor: Transformed image as a tensor.
        """
        if self.crop_box is not None:
            img = TF.crop(img, 50, 100, 535, 600)
        img = TF.to_tensor(img)
        # img = self.apply_binary_mask(img, self.threshold)
        img = TF.resize(img, self.resize_dims)
        img = TF.rgb_to_grayscale(img)
        img = TF.normalize(img, 0, 1)
        return img

    @staticmethod
    def apply_binary_mask(tensor, threshold):
        return torch.where(tensor > threshold, torch.ones_like(tensor), torch.zeros_like(tensor))

def evaluate_model_accuracy(model_path, csv_file_path, num_samples=None):
    """
    Evaluates the accuracy of a model based on predictions for images specified in a CSV file.

    Args:
        model_path (str): Path to the model file.
        csv_file_path (str): Path to the CSV file containing filenames and correct labels.
        base_image_folder (str): Base directory where images are stored.
        num_samples (int, optional): Number of samples to evaluate. If None, evaluates all.

    Returns:
        float: The accuracy percentage of the model predictions.
    """
    tester = InferenceTester(model_path)
    data = pd.read_csv(csv_file_path)
    if num_samples is not None:
        data = data.sample(n=num_samples, random_state=42)
    
    correct_predictions = 0
    for _, row in data.iterrows():
        image_path = f"{base_image_folder}/{row['Filename']}"
        predicted_frequency = tester.perform_inference(image_path)
        if predicted_frequency == row['Frequency(MHz)']:
            correct_predictions += 1

    accuracy = correct_predictions / len(data) * 100
    return accuracy

# Run inference testing
model_file_path = "./TrainingModelV1.pt"
csv_file_path = 'project_root/Continuous/spectrogram_labels.csv'
base_image_folder = 'project_root/Continuous'
accuracy = evaluate_model_accuracy(model_file_path, csv_file_path, num_samples=100)
print(f'Model accuracy: {accuracy}%')
