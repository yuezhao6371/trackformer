import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd

def load_dataloader(config, device):
    data_dir = config['data']['data_dir']    
    train_input = load_data(data_dir, config['data']['train_inputfile'])    
    train_output = load_data(data_dir, config['data']['train_outputfile'])    
    val_input = load_data(data_dir, config['data']['val_inputfile'])    
    val_output = load_data(data_dir, config['data']['val_outputfile'])    
    test_input = load_data(data_dir, config['data']['test_inputfile'])    
    test_output = load_data(data_dir, config['data']['test_outputfile'])
    test_hit_ids = load_data(data_dir, config['data']['test_hit_ids_file'])
    test_event_ids = load_data(data_dir, config['data']['test_event_ids_file'])

    train_dataset = TensorDataset(train_input, train_output) 
    val_dataset = TensorDataset(val_input, val_output)
    test_dataset = TensorDataset(test_input, test_output, test_hit_ids, test_event_ids)

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
