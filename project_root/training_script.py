# Import statements
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from algorithm_model import config, Algorithm # Import hyperparameter values
from data_preprocessing import DatasetPreprocessor

import time
import torch.cuda
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np

def show_images(images, labels, num_images=5):
    """
    Displays a batch of images with their corresponding labels.

    Args:
        images (torch.Tensor): The batch of images to display.
        labels (torch.Tensor): The batch of labels corresponding to the images.
        num_images (int): Number of images to display.
    """
    images = images.to('cpu').numpy()  # Convert images to NumPy arrays for visualization
    labels = labels.to('cpu').numpy()
    
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        ax = axes[i]
        img = np.transpose(images[i], (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1] for display
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Label: {labels[i]}')
    plt.show()

def save_model(model, path="./TrainingModelV1.pt"):
    """
    Saves the model state to a specified path.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): File path where the model state will be saved.
    """
    torch.save(model.state_dict(), path)

def test_accuracy(model, dataloader, device):
    """
    Evaluates the model's accuracy on the provided dataloader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader with test/validation data.
        device (torch.device): The device tensors will be transferred to.

    Returns:
        float: The accuracy percentage.
    """
    model.eval()  # Set model to evaluation mode
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)  # Get the indices of the max logit which are the predicted classes
            total_correct += (preds == labels).sum().item()  # Compare predictions with true labels
            total_samples += labels.size(0)  # Update total number of samples processed
    
    accuracy = 100 * total_correct / total_samples
    model.train()  # Set model back to training mode
    return accuracy

def train(model, num_epochs, train_dataloader, test_dataloader, criterion, optimizer, device):
    """
    Trains the model using the provided training and validation sets.

    Args:
        model (torch.nn.Module): The model to train.
        num_epochs (int): Number of epochs to train for.
        train_dataloader (DataLoader): DataLoader for training data.
        test_dataloader (DataLoader): DataLoader for test data.
        criterion (loss function): The loss function to use for training.
        optimizer (torch.optim.Optimizer): The optimizer for updating model weights.
        device (torch.device): The device model and data are transferred to.
    """
    best_accuracy = 0.0
    print("Begin training ...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # if i == 0:  # Visualize the first batch of each epoch
            #     show_images(images, labels)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss_value = running_loss / len(train_dataloader)
        accuracy = test_accuracy(model, test_dataloader, device)
        
        print(f'Epoch {epoch}, Loss: {train_loss_value}, Accuracy: {accuracy}%')
        
        if accuracy > best_accuracy:
            save_model(model)
            best_accuracy = accuracy

def main():
    """
    Main function to execute the training process, including setting up data, model, and training parameters.
    """
    start = time.perf_counter()

    # Extract hyperparameters
    input_size = config["input_size"]
    hidden_size = config["hidden_size"]
    output_size = config["output_size"]
    num_hidden_layers = config["num_hidden_layers"]
    learning_rate = config["learning_rate"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]

    # Dataset file paths
    csv_file = "/Users/cadeglauser/VSCose2Projects/Datasets/Continuous/spectrogram_labels.csv"
    root_dir = "/Users/cadeglauser/VSCose2Projects/Datasets/Continuous"
    train_output_file = "/Users/cadeglauser/VSCose2Projects/ECE4900-Algorithm/project_root/data/processed_TrainDataset100.pt"
    test_output_file = "/Users/cadeglauser/VSCose2Projects/ECE4900-Algorithm/project_root/data/processed_TestDataset100.pt"

    # Preprocess the dataset (Done only once)
    # COMMENT THIS OUT AFTER FIRST TIME PREPROCESSING
    # # Training dataset
    # preprocessorTrain = DatasetPreprocessor(csv_file,
    #                                 root_dir,
    #                                 train_output_file,
    #                                 subset='train',
    #                                 threshold=0.2)
    # preprocessorTrain.process_and_save()

    # # Testing dataset
    # preprocessorTest = DatasetPreprocessor(csv_file,
    #                                 root_dir,
    #                                 test_output_file,
    #                                 subset='test',
    #                                 threshold=0.2)
    # preprocessorTest.process_and_save()

    # Load the preprocessed datasets
    train_dataset = torch.load(train_output_file)
    test_dataset = torch.load(test_output_file)

    # Initialize the DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    # Define GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Model will use: {device}")

    # Create instance of algorithm model
    model = Algorithm(config['input_size'], config['hidden_size'], config['output_size'], config['num_hidden_layers'])
    model.to(device)
    model.init_weights()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Train the model
    train(model, config['num_epochs'], train_dataloader, test_dataloader, criterion, optimizer, device)

    print("Finished Training")
    end = time.perf_counter() - start
    print(f"Training Latency: {end}s for {config['num_epochs']} epochs")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()

