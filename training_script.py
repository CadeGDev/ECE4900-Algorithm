# Import statements
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from algorithm_model import config, Algorithm_v0_1 # Import hyperparameter values
from data_preprocessing import DatasetPreprocessor

import time
import torch.cuda
from torch.autograd import Variable

def save_model(model, path="./TrainingModelV1.pth"):
    torch.save(model.state_dict(), path)

def test_accuracy(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(outputs)
            # Convert probabilities to binary predictions
            preds = (probs > 0.5).float()
            # Update total_correct by comparing predictions with true labels
            total_correct += (preds == labels).sum().item()
            total_samples += labels.numel()  # Update total number of label comparisons
    
    accuracy = 100 * total_correct / total_samples
    model.train()  # Set model back to training mode
    return accuracy

def train(model, num_epochs, train_dataloader, test_dataloader, criterion, optimizer, device):
    best_accuracy = 0.0
    print("Begin training ...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for images, labels in train_dataloader:
            # print(images.shape, labels.shape)
            images = images.to(device)
            labels = labels.to(device)
            
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
    csv_file = "/Users/cadeglauser/VSCose2Projects/Datasets/Cont100/spectrogram_labels.csv"
    root_dir = "/Users/cadeglauser/VSCose2Projects/Datasets/Cont100"
    train_output_file = "/Users/cadeglauser/VSCose2Projects/ECE4900-Algorithm/project_root/data/processed_TrainDataset100.pt"
    test_output_file = "/Users/cadeglauser/VSCose2Projects/ECE4900-Algorithm/project_root/data/processed_TestDataset100.pt"

    # Preprocess the dataset (Done only once)
    # COMMENT THIS OUT AFTER FIRST TIME PREPROCESSING
    # Training dataset
    preprocessorTrain = DatasetPreprocessor(csv_file,
                                    root_dir,
                                    train_output_file,
                                    subset='train',
                                    threshold=0.5)
    preprocessorTrain.process_and_save()

    # Testing dataset
    preprocessorTest = DatasetPreprocessor(csv_file,
                                    root_dir,
                                    test_output_file,
                                    subset='test',
                                    threshold=0.5)
    preprocessorTest.process_and_save()

    # Load the preprocessed datasets
    train_dataset = torch.load(train_output_file)
    test_dataset = torch.load(test_output_file)

    # Check the first label tensor to make sure it's the correct shape
    # print(dataset.tensors[1][0].shape)  # Should be torch.Size([50])

    # Initialize the DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    # Define GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Model will use: {device}")

    # Create instance of algorithm model
    model = Algorithm_v0_1(config['input_size'], config['hidden_size'], config['output_size'], config['num_hidden_layers'])
    model.to(device)
    model.init_weights()

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Train the model
    train(model, config['num_epochs'], train_dataloader, test_dataloader, criterion, optimizer, device)

    print("Finished Training")
    end = time.perf_counter() - start
    print(f"Training Latency: {end}s for {config['num_epochs']} epochs")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()

