# Import statements
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from algorithm_model import config, Algorithm_v0_1 # Import hyperparameter values
from data_preprocessing import DatasetPreprocessor

# Dataset file paths
csv_file = 'ECE4900-Algorithm/project_root\data\Continuous\spectrogram_labels.csv'
root_dir = 'ECE4900-Algorithm/project_root/data/Continuous'
train_output_file = "ECE4900-Algorithm/project_root/data/processed_TrainDataset.pt"
test_output_file = "ECE4900-Algorithm/project_root/data/processed_TestDataset.pt"

def main():
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

if __name__ == '__main__':
    main()