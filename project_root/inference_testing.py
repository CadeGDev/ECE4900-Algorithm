import torch
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF
from algorithm_model import Algorithm, config
import argparse
import os


class InferenceTester:
   def __init__(self, model_path, image_path):
       self.model_path = model_path
       self.image_path = image_path
       self.model = self.load_model()
       self.transformation_pipeline = TransformationPipeline()


   def load_model(self):
       model = Algorithm(config['input_size'], config['hidden_size'], config['output_size'], config['num_hidden_layers'])
       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       print(f"Model will use: {device}")
       model.load_state_dict(torch.load(self.model_path, map_location=device))
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
       #img = self.apply_binary_mask(img, self.threshold)
       img = TF.resize(img, self.resize_dims)
       img = TF.rgb_to_grayscale(img)
       img = TF.normalize(img, 0, 1)
       return img


   @staticmethod
   def apply_binary_mask(tensor, threshold):
       return torch.where(tensor > threshold, torch.ones_like(tensor), torch.zeros_like(tensor))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spectrogram Inference')
  
    parser.add_argument('--model', action="store", required = True, help = "Pre-trained model path")
    parser.add_argument('--image', action="store", required = True, help = "Image to be processed by script")
    args = parser.parse_args()


    tester = InferenceTester(args.model, args.image)
    predicted_class = tester.perform_inference()
    print(f'Predicted class: {predicted_class}')
    # Specify the name of the new text file
    file_name = "inference.txt"

    # Create the "Output" folder if it doesn't exist
    output_folder = "/home/l3harris-clinic/Desktop/Algorithm/Output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define the full path to the file within the "Output" folder
    file_path = os.path.join(output_folder, file_name)

    # Open the new text file in write mode ('w')
    with open(file_path, "w") as file:
        # Write the string to the file
        file.write(str(predicted_class))

# Example usage 
"""
model_file_path = "./TrainingModelV1.pt"
spectrogram_image_path = '/path/to/spectrogram_image.png'
tester = InferenceTester(model_file_path, spectrogram_image_path)
predicted_class = tester.perform_inference()
print(f'Predicted class: {predicted_class}')
python3 /home/l3harris-clinic/Desktop/Algorithm/ECE4900-Algorithm/project_root/inference_testing.py --model /home/l3harris-clinic/Desktop/Algorithm/Trained_Models/TrainingModelV2Cont.pt --image /home/l3harris-clinic/Desktop/Algorithm/Spectrograms/spectrogram11.png
"""

