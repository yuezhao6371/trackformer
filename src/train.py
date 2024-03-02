import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import numpy as np
import pandas as pd
import random
import os
import toml
import logging
import wandb
import argparse
from model import TransformerClassifier
import utils.training_utils as training_utils
import utils.data_utils as data_utils
import utils.output_utils as output_utils
import utils.wandb_utils as wandb_utils

def load_config(config_path):
    """
    Load the TOML configuration file and return a dictionary.
    """
    with open(config_path, 'r') as config_file:
        config = toml.load(config_file)
    return config

def setup_logging(config, output_dir):
    level = getattr(logging, config["logging"]["level"].upper(), logging.INFO)
    log_file = os.path.join(output_dir, 'training.log')
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

def initialize_wandb(config, output_dir):
    wandb_logger = wandb_utils.WandbLogger(config=config["wandb"],
        output_dir=output_dir,
        job_type="training")
    return wandb_logger.initialize()

def setup_training(config):
    # model
    model = TransformerClassifier(
        inputfeature_dim = config['model']['inputfeature_dim'],
        num_classes = config['model']['num_classes'],
        num_heads=config['model']['num_heads'],
        embed_dim = config['model']['embed_dim'],
        num_layers = config['model']['num_layers'],
        dropout=config['model']['dropout']
    )

    # optimizer
    initial_lr = config['training']['scheduler']['initial_lr']
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr)

    # scheduler
    mode = config['training']['scheduler']['mode']
    factor = config['training']['scheduler']['factor']
    patience = config['training']['scheduler']['patience']
    verbose = config['training']['scheduler']['verbose']
    lr_scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose)

    # criterion
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, lr_scheduler, criterion

def train_epoch(model, trainloader, optimizer, criterion, device, config, epoch, wandb_logger):
    model.train()  # Set model to training mode
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        # inputs.shape = (batch_size, seq_len, num_feature)

        optimizer.zero_grad()

        outputs = model(inputs)

        # flatten outputs and labels
        outputs = outputs.view(-1, num_classes)
        labels = labels.view(-1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate training accuracy for this batch
        _, predicted = torch.max(outputs.data, 1) # getting predicted labels
        batch_correct = (predicted == labels).sum().item()
        epoch_correct += batch_correct
        epoch_total += labels.size(0)
        for c in range(0, num_classes): # this includes class 0 which is padding, but it's ignored later
            class_correct_counts[c] += ((predicted == c) & (labels == c)).sum().item()
            false_positive_counts[c] += ((predicted == c) & (labels != c)).sum().item()
            class_total_counts[c] += (labels == c).sum().item()

    epoch_accuracy = 100 * epoch_correct / epoch_total
    epoch_loss = total_loss / len(trainloader)
    
    classes_mask = class_total_counts > 0 # excluding empty classes
    classes_mask[0] = False  # Exclude the class with label 0, which is padding
    class_success_rates = np.where(classes_mask, class_correct_counts / class_total_counts, 0)
    false_positive_rates = np.where(classes_mask, false_positive_counts / class_total_counts, 0)
    successful_classes_mask = (class_success_rates > 0.5) & (false_positive_rates < 0.5) & classes_mask
    successful_classes = np.sum(successful_classes_mask) # already excluding padding class
    false_positives = np.sum(false_positive_rates)
    total_classes = np.sum(classes_mask)
    # checking for zero in case all the initial outputs have class 0,
    # in that case we'd be dividing by zero
    epoch_score = 0 if total_classes == 0 else 100 * successful_classes / total_classes
    false_pos_avg = 0 if total_classes == 0 else 100 * false_positives / total_classes

    if (epoch + 1) % config.epoch_log_interval == 0:
    #if (epoch + 1) % epoch_log_interval == 0 and i==len(trainloader)-1:
        logging.info(f'Epoch {epoch + 1}, Training loss: {epoch_loss}')
        logging.info(f'Training accuracy: {epoch_accuracy:.2f}%')
        logging.info(f'Training TrackML score: {epoch_score:.2f}%')
        logging.info(f'Training false positive rate: {false_pos_avg:.2f}%')

    logging.info(f"Train Epoch: {epoch}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    wandb_logger.log({"train_loss": epoch_loss, "train_accuracy": epoch_accuracy, "epoch": epoch})


def validate_epoch(model, valloader, criterion, device, config, epoch, wandb_logger):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs).view(-1, num_classes)
            labels = labels.view(-1)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
            val_loss += loss.item()
            for c in range(num_classes): # this includes class 0 which is padding, but it's ignored later
                val_class_correct_counts[c] += ((predicted == c) & (labels == c)).sum().item()
                val_false_positive_counts[c] += ((predicted == c) & (labels != c)).sum().item()
                val_class_total_counts[c] += (labels == c).sum().item()

    val_accuracy = 100 * val_correct / val_total
    val_loss /= len(valloader)
    
    train_losses.append(epoch_loss)
    val_losses.append(val_loss)
    
    classes_mask_val = val_class_total_counts > 0 # excluding empty classes
    classes_mask_val[0] = False  # Exclude the class with label 0, which is padding
    val_class_success_rates = np.where(classes_mask_val, val_class_correct_counts / val_class_total_counts, 0)
    val_false_positive_rates = np.where(classes_mask_val, val_false_positive_counts / val_class_total_counts, 0)
    successful_classes_mask_val = (val_class_success_rates > 0.5) & (val_false_positive_rates < 0.5) & classes_mask_val
    successful_classes_val = np.sum(successful_classes_mask_val) # already excluding padding class
    false_positives_val = np.sum(val_false_positive_rates)
    total_classes_val = np.sum(classes_mask_val)
    # checking for zero in case all the initial outputs have class 0,
    # in that case we'd be dividing by zero
    val_score = 0 if total_classes_val == 0 else 100 * successful_classes_val / total_classes_val
    val_false_pos_avg = 0 if total_classes_val == 0 else 100 * false_positives_val / total_classes_val
    
    if (epoch + 1) % epoch_print_interval == 0:
           #if (epoch + 1) % epoch_print_interval == 0 and i==len(trainloader)-1:
        logging.info(f'Epoch {epoch + 1}, Val loss: {val_loss}')
        logging.info(f'Val accuracy: {val_accuracy:.2f}%')
        logging.info(f'Val TrackML score: {val_score:.2f}%')
        logging.info(f'Val false positive rate: {val_false_pos_avg:.2f}%')
    
    logging.info(f"Val Epoch: {epoch}, Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
    wandb_logger.log({"val_loss": val_loss, "val_accuracy": val_accuracy, "epoch": epoch})

    return val_loss

 
def test(model, test_loader, device, num_classes, wandb_logger):
    model.eval()
    correct = 0
    total = 0
    test_class_correct_counts = np.zeros(num_classes)
    test_false_positive_counts = np.zeros(num_classes)
    test_class_total_counts = np.zeros(num_classes)
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs).view(-1, num_classes)
            labels = labels.view(-1)
            
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for c in range(num_classes):
                test_class_correct_counts[c] += ((predicted == c) & (labels == c)).sum().item()
                test_false_positive_counts[c] += ((predicted == c) & (labels != c)).sum().item()
                test_class_total_counts[c] += (labels == c).sum().item()
    
    accuracy = 100 * correct / total
    test_classes_mask = test_class_total_counts > 0 # excluding empty classes
    test_classes_mask[0] = False  # Exclude the class with label 0, which is padding
    zerovalues = np.where(test_classes_mask & (test_class_total_counts==0))
    
    test_class_success_rates = np.where(test_classes_mask, test_class_correct_counts / test_class_total_counts, 0)
    test_false_positive_rates = np.where(test_classes_mask, test_false_positive_counts / test_class_total_counts, 0)
    successful_test_classes_mask = (test_class_success_rates > 0.5) & (test_false_positive_rates < 0.5) & test_classes_mask # already excluded padding class
    successful_test_classes = np.sum(successful_test_classes_mask)
    test_false_positives = np.sum(test_false_positive_rates)
    test_total_classes = np.sum(classes_mask)
    test_score = 0 if test_total_classes == 0 else 100 * successful_test_classes / test_total_classes
    test_false_pos_avg = 0 if test_total_classes == 0 else 100 * test_false_positives / test_total_classes
    
    logging.info(f'Test accuracy: {accuracy:.2f}%')
    logging.info(f'Test TrackML score: {test_score:.2f}%')
    logging.info(f'Test false positive rate: {test_false_pos_avg:.2f}%')
    logging.info(f'Sanity check, this value should be zero: {zerovalues:.2f}%')
    
    model.train()

def main(config_path):
    config = load_config(config_path)
    output_dir = output_utils.unique_output_dir(config) # with time stamp
    output_utils.copy_config_to_output(config_path, output_dir)
    setup_logging(config, output_dir)
    wandb_logger = initialize_wandb(config, output_dir)
    early_stopper = training_utils.EarlyStopping(config['early_stopping'], output_dir, trace_func=logging.info)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    model, optimizer, scheduler, criterion = setup_training(config)
    train_loader, val_loader, test_loader = data_utils.load_dataloader(config, device)

    logging.info("Started training and validation")
    training_utils.log_memory_usage()
    for epoch in range(config['training']['num_epochs']):
        training_utils.adjust_learning_rate(optimizer, epoch, config)
        train_epoch(model, train_loader, optimizer, criterion, device, config, epoch, wandb_logger)

        val_loss = validate_epoch(model, val_loader, criterion, device, config, epoch, wandb_logger)
        # adjust learning rate based on validation loss
        lr_scheduler.step(val_loss) 
        # stop training and checkpoint the model if val loss stops improving
        early_stopper(val_loss, model)

    logging.info("Finished training and started testing")
    test(model, test_loader, device, config.num_classes, wandb_logger)
    logging.info("Finished testing")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total Trainable Parameters: {total_params}")
    wandb_logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with a given config file.")
    parser.add_argument('config_path', type=str, help='Path to the configuration TOML file.')
    
    # Parse arguments
    args = parser.parse_args()
    main(args.config_path)

