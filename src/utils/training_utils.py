import torch
import numpy as np
import os
import logging

def log_memory_usage():
    logging.info(f"Memory Allocated: {torch.cuda.memory_allocated() / 1e9} GB")
    logging.info(f"Memory Reserved:  {torch.cuda.memory_reserved() / 1e9} GB")

def adjust_learning_rate(optimizer, epoch, config):
    """Adjusts learning rate for each epoch during the warm-up phase, based on TOML config."""
    warmup_epochs = config['training']['scheduler']['warmup_epochs']
    if epoch < warmup_epochs:
        initial_lr = config['training']['scheduler']['initial_lr']
        target_lr = config['training']['scheduler']['target_lr']
        lr = initial_lr + (target_lr - initial_lr) * (epoch / warmup_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, config, output_dir, trace_func=print):
        self.patience = config['patience']
        self.verbose = config['verbose']
        self.delta = config['delta']
        self.path = os.path.join(output_dir, "earlystop_checkpoint.pt")
        self.trace_func = trace_func
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
