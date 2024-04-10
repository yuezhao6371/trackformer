import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd

def load_dataloader(config, device):
    data_dir = config['data']['data_dir']    
    train_dataset = load_data(data_dir, config['data']['train_file'])    
    val_dataset = load_data(data_dir, config['data']['val_file'])    
    test_dataset = load_data(data_dir, config['data']['test_file'])    

    batch_size = config['training']['batch_size']
    shuffle = config['training']['shuffle']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader

def load_data(data_dir, file_name):
    file_path = os.path.join(data_dir, file_name)
    return torch.load(file_path)

def load_truths(config):
    data_dir = config['data']['data_dir']
    test_truth_filename = config['data']['test_truthfile']
    test_truth_filepath = os.path.join(data_dir, test_truth_filename)
    return pd.read_csv(test_truth_filepath)
