import torch
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import logging

def load_dataloader(config, device):
    data_dir = config['data']['data_dir']    
    train_dataset = ForwardPassDataset(data_dir, config['data']['train_file'])
    val_dataset = ForwardPassDataset(data_dir, config['data']['val_file'])
    test_dataset = ForwardPassDataset(data_dir, config['data']['test_file'])
    test_helper_dataset = ScoringHelperDataset(data_dir, config['data']['test_helperfile'])

    batch_size = config['training']['batch_size']
    shuffle = config['training']['shuffle']
    num_workers = config['data']['dataloader_num_workers']
    logging.info("Loading data with DataLoader")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
    test_helper_loader = DataLoader(test_helper_dataset, batch_size=batch_size, num_workers=num_workers) # padding not necessary here
    return train_loader, val_loader, test_loader, test_helper_loader

def load_truths(config):
    data_dir = config['data']['data_dir']
    test_truth_filename = config['data']['test_truthfile']
    test_truth_filepath = os.path.join(data_dir, test_truth_filename)
    return pd.read_csv(test_truth_filepath)

# dynamically padding data based on the max seq_length of the batch
def collate_fn(batch):
    coords, labels = zip(*batch)
    coords_padded = pad_sequence(coords, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    return coords_padded, labels_padded 

class ForwardPassDataset(Dataset):
    def __init__(self, data_dir, file_name):
        file_path = os.path.join(data_dir, file_name)
        self.coords, self.labels = torch.load(file_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        coords_tensor = torch.tensor(self.coords[idx], dtype=torch.float)
        labels_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return coords_tensor, labels_tensor

class ScoringHelperDataset(Dataset):
    def __init__(self, data_dir, file_name):
        file_path = os.path.join(data_dir, file_name)
        self.hit_ids, self.event_ids = torch.load(file_path)

    def __len__(self):
        return len(self.hit_ids)

    def __getitem__(self, idx):
        hit_ids_tensor = torch.tensor(self.hit_ids[idx], dtype=torch.long)
        event_ids_tensor = torch.tensor(self.event_ids[idx], dtype=torch.long)
        return hit_ids_tensor, event_ids_tensor
