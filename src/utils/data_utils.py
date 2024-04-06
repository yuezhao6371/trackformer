import torch
from torch.utils.data import DataLoader, TensorDataset
import os

def load_dataloader(config, device):
    data_dir = config['data']['data_dir']    
    train_input = load_data(data_dir, config['data']['train_inputfile'])    
    train_output = load_data(data_dir, config['data']['train_outputfile'])    
    val_input = load_data(data_dir, config['data']['val_inputfile'])    
    val_output = load_data(data_dir, config['data']['val_outputfile'])    
    test_input = load_data(data_dir, config['data']['test_inputfile'])    
    test_output = load_data(data_dir, config['data']['test_outputfile'])

    train_input_tensor = torch.tensor(train_input, dtype=torch.float32).to(device)
    train_labels_tensor = torch.tensor(train_output, dtype=torch.long).to(device) 
    train_dataset = TensorDataset(train_input_tensor, train_labels_tensor) 

    val_input_tensor = torch.tensor(val_input, dtype=torch.float32).to(device)
    val_labels_tensor = torch.tensor(val_output, dtype=torch.long).to(device)
    val_dataset = TensorDataset(val_input_tensor, val_labels_tensor)

    test_input_tensor = torch.tensor(test_input, dtype=torch.float32).to(device)
    test_labels_tensor = torch.tensor(test_output, dtype=torch.long).to(device)
    test_dataset = TensorDataset(test_input_tensor, test_labels_tensor)

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
