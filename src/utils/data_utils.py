import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

def load_dataloader(config, device):
    data_dir = config['data']['data_dir']    
    train_dataset = load_data(data_dir, config['data']['train_file'])    
    val_dataset = load_data(data_dir, config['data']['val_file'])    
    test_dataset = load_data(data_dir, config['data']['test_file'])    
    test_helper_dataset = load_data(data_dir, config['data']['test_helperfile']) # hit_ids and event_ids 

    batch_size = config['training']['batch_size']
    shuffle = config['training']['shuffle']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_helper_loader = DataLoader(test_helper_dataset, batch_size=batch_size) # padding not necessary here
    return train_loader, val_loader, test_loader, test_helper_loader

def load_data(data_dir, file_name):
    file_path = os.path.join(data_dir, file_name)
    return torch.load(file_path)

def load_truths(config):
    data_dir = config['data']['data_dir']
    test_truth_filename = config['data']['test_truthfile']
    test_truth_filepath = os.path.join(data_dir, test_truth_filename)
    return pd.read_csv(test_truth_filepath)

# dynamically padding data based on the max seq_length of the batch
def collate_fn(batch):
    coords, labels = zip(*batch)
    coords_padded = pad_sequence(coords, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)
    return coords_padded, labels_padded 
