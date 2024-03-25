# Import statements
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms  # For data preprocessing
from torch.utils.data import DataLoader, TensorDataset
from algorithm_model import config, Algorithm_v0_1 # Import hyperparameter values
from data_preprocessing import SpectrogramDataset
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split # To install run "pip install scikit-learn" in terminal

import time
import torch.cuda
from torch.autograd import Variable

start = time.perf_counter()

# Define the path to dataset and labels
image_folder = 'project_root/Continuous'
labels_csv = 'project_root/Continuous/spectrogram_labels.csv'

# Extract hyperparameters
input_size = config["input_size"]
hidden_size = config["hidden_size"]
output_size = config["output_size"]
num_hidden_layers = config["num_hidden_layers"]
learning_rate = config["learning_rate"]
num_epochs = config["num_epochs"]
batch_size = config["batch_size"]

# define GPU device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Model will use: {device}")

# Create instance of algorithm model
model = Algorithm_v0_1(input_size, hidden_size, output_size, num_hidden_layers)

# Set model to use GPU 
model.to(device)
# Apply the custom weight initialization function to the model
model.init_weights()

num_workers = 4  # Number of subprocesses for data loading

# Initialize the dataset and data loader
train_dataset = SpectrogramDataset(csv_file=labels_csv, root_dir=image_folder, subset='train')
test_dataset = SpectrogramDataset(csv_file=labels_csv, root_dir=image_folder, subset='test')

dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Example loss function for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Variable initializations
num_epochs = 50  # Adjust

# Save training model
def saveModel():
    path = "./TrainingModelV1.pth"
    torch.save(model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():
    running_accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        model.eval()
        for batch in dataloader_test:
            images, labels = batch['image'], batch['label']
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            running_accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * running_accuracy / total)
    return(accuracy)


# Training loop
def train(num_epochs):
    best_accuracy = 0.0
    print("Begin training ...")
    # loop over data repeatedly
    for epoch in range(num_epochs):
        # track loss/accuracy
        running_loss = 0.0
        running_acc = 0.0
        
        for i,batch in enumerate(dataloader_train):
            # init inputs and set to use GPU
            images, labels = batch['image'], batch['label']
            #images = Variable(images.to(device))
            #labels = Variable(labels.to(device))
            
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
        train_loss_value = running_loss/len(dataloader_train) 

        # Calculate average accuracy for this epoch
        accuracy = testAccuracy()
        
        print(f'Completed training batch {epoch}, Accuracy is {accuracy}')    

        # Save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy

if __name__ == "__main__": 
    num_epochs = 10
    train(num_epochs) 
    print('Finished Training\n') 
    end = time.perf_counter() - start
    print(f"Training Latency: {format(end)}s for {num_epochs} epochs")
