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
import utils.metrics_calculator as metrics_calculator
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
    wandb_logger.initialize()
    return wandb_logger

def setup_training(config, device):
    # model
    model = TransformerClassifier(
        inputfeature_dim = config['model']['inputfeature_dim'],
        num_classes = config['data']['num_classes'],
        num_heads=config['model']['num_heads'],
        embed_dim = config['model']['embed_dim'],
        num_layers = config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)

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

def train_epoch(model, trainloader, optimizer, criterion, device, config, epoch, metrics_calculator, wandb_logger):
    model.train()  # Set model to training mode

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        # inputs.shape = (batch_size, seq_len, num_feature)

        optimizer.zero_grad()

        outputs = model(inputs)

        # flatten outputs and labels
        outputs = outputs.view(-1, model.num_classes)
        labels = labels.view(-1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # update values used for calculating metrics
        metrics_calculator.update(outputs, labels, loss=loss.item())

    # calculate metrics
    epoch_accuracy = metrics_calculator.calculate_accuracy()
    epoch_loss = metrics_calculator.calculate_loss(len(trainloader))
    epoch_score = metrics_calculator.calculate_trackml_score()

    if (epoch + 1) % config['logging']['epoch_log_interval'] == 0:
    #if (epoch + 1) % epoch_log_interval == 0 and i==len(trainloader)-1:
        logging.info(f'Epoch {epoch + 1}, Training loss: {epoch_loss}')
        logging.info(f'Training accuracy: {epoch_accuracy:.2f}%')
        logging.info(f'Training TrackML score: {epoch_score:.2f}%')

    wandb_logger.log({"train_loss": epoch_loss, "train_accuracy": epoch_accuracy, "train_score": epoch_score, "epoch": epoch})


def validate_epoch(model, valloader, criterion, device, config, epoch, metrics_calculator, wandb_logger):
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs).view(-1, model.num_classes)
            labels = labels.view(-1)
            loss = criterion(outputs, labels)
        
            metrics_calculator.update(outputs, labels, loss=loss.item())
            
    epoch_accuracy = metrics_calculator.calculate_accuracy()
    epoch_loss = metrics_calculator.calculate_loss(len(valloader))
    epoch_score = metrics_calculator.calculate_trackml_score()
    
    if (epoch + 1) % config['logging']['epoch_log_interval'] == 0:
           #if (epoch + 1) % epoch_print_interval == 0 and i==len(trainloader)-1:
        logging.info(f'Epoch {epoch + 1}, Val loss: {epoch_loss}')
        logging.info(f'Val accuracy: {epoch_accuracy:.2f}%')
        logging.info(f'Val TrackML score: {epoch_score:.2f}%')
    
    wandb_logger.log({"val_loss": epoch_loss, "val_accuracy": epoch_accuracy, "val_scoore": epoch_score, "epoch": epoch})

    return epoch_loss

 
def test(model, test_loader, device, num_classes, wandb_logger):
    model.eval()
    metrics_calculator = MetricsCalculator(model.num_classes)
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs).view(-1, num_classes)
            labels = labels.view(-1)
            
            metrics_calculator.update(outputs, labels)           
 
    accuracy = metrics_calculator.calculate_accuracy()
    score = metrics_calculator.calculate_trackml_score()
    
    logging.info(f'Test accuracy: {accuracy:.2f}%')
    logging.info(f'Test TrackML score: {score:.2f}%')
    wandb_logger.log({"test_accuracy": accuracy, "test_scoore": score})
    

def main(config_path):
    config = load_config(config_path)
    output_dir = output_utils.unique_output_dir(config) # with time stamp
    output_utils.copy_config_to_output(config_path, output_dir)
    setup_logging(config, output_dir)
    logging.info(f'output_dir: {output_dir}%')
    wandb_logger = initialize_wandb(config, output_dir)
    early_stopper = training_utils.EarlyStopping(config['training']['early_stopping'], output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    model, optimizer, lr_scheduler, criterion = setup_training(config, device)
    train_loader, val_loader, test_loader = data_utils.load_dataloader(config, device)
    train_metrics_calculator = metrics_calculator.MetricsCalculator(model.num_classes)
    val_metrics_calculator = metrics_calculator.MetricsCalculator(model.num_classes)

    logging.info("Started training and validation")
    training_utils.log_memory_usage()
    for epoch in range(config['training']['num_epochs']):
        # resetting values used for calculating epoch metrics
        train_metrics_calculator.reset()
        val_metrics_calculator.reset()
        
        train_epoch(model, train_loader, optimizer, criterion, device, config, epoch, train_metrics_calculator, wandb_logger)

        val_loss = validate_epoch(model, val_loader, criterion, device, config, epoch, val_metrics_calculator, wandb_logger)
        # adjust learning rate based on validation loss
        lr_scheduler.step(val_loss) 
        # stop training and checkpoint the model if val loss stops improving
        early_stopper(val_loss)
        if early_stopper.should_stop():
            logging.info("Early stopping triggered. Saving checkpoint.")
            wandb_logger.save_model(output_dir)
            break
        # learning rate warm-up
        training_utils.adjust_learning_rate(optimizer, epoch, config)

    logging.info("Finished training and started testing")
    test(model, test_loader, device, config['data']['num_classes'], wandb_logger)
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

