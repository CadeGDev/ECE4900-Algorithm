# Import statements
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms  # For data preprocessing
import torch.cuda
from torch.utils.data import DataLoader, TensorDataset
from model_definition.algorithm_model import config, Algorithm_v0_1 # Import hyperparameter values

start = time.perf_counter()

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
print("model will use: " + device)


# Create instance of algorithm model
model = Algorithm_v0_1(input_size, hidden_size, output_size, num_hidden_layers)
# Set model to use GPU 
model.to(device)

# Apply the custom weight initialization function to the model
model.init_weights()

# Load and preprocess your RF spectrum data
# TODO

# Define loss function and optimizer ### EDIT LATER
criterion = nn.MSELoss()  # Example loss function for regression
#criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)
#optimizer = SGD(model.parameters(), lr = 0.001)

# Variable initializations
num_epochs = 50  # Adjust

# Dataset initialization
# TODO: data preprocessing (init, labeling, etc.)
train_inputs = 0
train_labels = 0

# Assuming you have your data as tensors
train_data = TensorDataset(train_inputs, train_labels)
batch_size = 32  # Adjust the batch size as needed

train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True) 

#TODO: define validation and test sets
validate_loader = DataLoader(validate_data, batch_size = 1) 
test_loader = DataLoader(test_data, batch_size = 1)

# Save training model
def saveModel():
    path = "./TrainingModelV1.pth"
    torch.save(model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():
    
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in train_loader:
            inputs, labels = data
            # run the model on the test set to predict labels
            outputs = model(inputs)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)



# Training loop
def train(num_epochs):
    best_accuracy = 0.0

    # loop over data repeatedly
    for epoch in range(num_epochs):
        # track loss/accuracy
        running_loss = 0.0
        running_acc = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # init inputs and set to use GPU
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            # zero parameter gradiants
            optimizer.zero_grad()
            # predict using training data 
            outputs = model(inputs)
            # compute loss
            loss = criterion(outputs, labels)
            # back prop loss
            loss.backward()
            # adjust parameters
            optimizer.step()
            
            # Validation and monitoring during training
            # extract loss for loop iteration
            running_loss += loss.item()    
            if i % 30 == 0:    
                # print every 30
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                # reset loss
                running_loss = 0.0

        # Compute average accuracy for this epoch
        accuracy = testAccuracy()
        
        print(f'Completed training batch {epoch}, Accuracy is {accuracy}')    

        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy
          
num_epochs = 100
train(num_epochs)

end = time.perf_counter() - start
print(f'Training Latency: {format(end)}s for {num_epochs} epochs')