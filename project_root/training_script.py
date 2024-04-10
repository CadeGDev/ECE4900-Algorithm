import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from algorithm_model import config, Algorithm_v0_1 # Import hyperparameter values
from data_preprocessing import DatasetPreprocessor

import sys
import time
import torch.cuda
from torch.autograd import Variable

def setup():
    """
    This function contains the setup and initialization of the training/testing data, and the algorithm model 

    Returns: 
        model: custom algorithm model extracted from algorithm_model.py
        train_dataloader: dataLoader containg training images and labels
        test_dataloader: dataLoader containg testing images and labels
        criterion: loss function used to train algorithm 
        optimizer: optimizer function used to optimize algorithm
    """

    # Extract hyperparameters
    input_size = config["input_size"]
    hidden_size = config["hidden_size"]
    output_size = config["output_size"]
    num_hidden_layers = config["num_hidden_layers"]
    learning_rate = config["learning_rate"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]

    train_output_file = "ECE4900-Algorithm/project_root/data/processed_TrainDataset.pt"
    test_output_file = "ECE4900-Algorithm/project_root/data/processed_TestDataset.pt"

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
    model = Algorithm_v0_1(config['input_size'], config['hidden_size'], config['output_size'], config['num_hidden_layers'])
    model.to(device)
    model.init_weights()

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    return(model, train_dataloader, test_dataloader, criterion, optimizer, device)

# Save training model
def saveModel(model):
    """
    This function saves the algorithm with the highest accuracy model 
    """
    path = "ECE4900-Algorithm\project_root\models\TrainedModelV1.pth"
    torch.save(model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy(model, test_dataloader, device):
    """
    This function returns the accuracy resulting from one epoch of training, tested over the testing dataset 

    Params:
        model: custom algorithm model extracted from algorithm_model.py
        test_dataloader: dataLoader containg testing images and labels

    Returns:
        accuracy: the percentage of accurate predictions made by the algorithm
    """
    model.eval()  # Set model to evaluation mode
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in test_dataloader:
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


# Training loop
def train(num_epochs, model, train_dataloader, test_dataloader, criterion, optimizer, device):
    """
    This function contains the training loop, accuracy and loss evaluation, and model saving during algorithm training 

    Params: 
        num_epochs: number of times the training data set is iterated through 
        model: custom algorithm model extracted from algorithm_model.py
        train_dataloader: dataLoader containg training images and labels
        test_dataloader: dataLoader containg testing images and labels
        criterion: loss function used to train algorithm 
        optimizer: optimizer function used to optimize algorithm
    """
    best_accuracy = 0.0
    # loop over data repeatedly
    for epoch in range(num_epochs):
        print(f"Beginning epoch {epoch} of {num_epochs}")
        # track loss/accuracy
        running_loss = 0.0
        accuracy = 0

        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            #FORWARD PASS
            optimizer.zero_grad() # zero parameter gradiants
            outputs = model(images) # predict using training data 
            loss = criterion(outputs, labels) # compute loss

            #BACKWARD AND OPTIMIZATION
            loss.backward() # back prop loss
            optimizer.step() # adjust parameters
            
            # Validation and monitoring during training
            # extract loss for loop iteration
            running_loss += loss.item()  # track the loss value 
        
        # Calculate training loss value 
        train_loss = running_loss/len(train_dataloader) 
        # Calculate average accuracy for this epoch
        accuracy = testAccuracy(model, test_dataloader, device)
        
        print(f"Completed training batch {epoch} || Loss: {train_loss}, Accuracy:{accuracy}")    

        # Save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel(model)
            best_accuracy = accuracy

# python project_root\training_script.py [num_epochs]
def main():
    if len(sys.argv) > 1:
        arg_epochs = sys.argv[1]
        try:
            num_epochs = int(arg_epochs)
        except ValueError:
            print("Argument is not an integer")
        print(f"Beginning training with {num_epochs} epochs...")

        # Log start time
        start = time.perf_counter()

        # Initialize algorithm and train/test data
        model, train_dataloader, test_dataloader, criterion, optimizer, device = setup()
        # Begin training algorithm
        train(num_epochs, model, train_dataloader, test_dataloader, criterion, optimizer, device) 
        print("Finished Training\n") 

        # Log end time
        end = time.perf_counter() - start
        print(f"Training Latency: {format(end)}s for {num_epochs} epochs")
    else:
        print("Scipt cancelled, specify number of epochs to run")
        sys.exit()
        
if __name__ == "__main__": 
    main()
