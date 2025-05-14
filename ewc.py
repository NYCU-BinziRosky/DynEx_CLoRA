"""
ewc.py — Elastic Weight Consolidation (EWC) continual learning framework

This module defines the core EWC logic including:
- Fisher information computation from past data
- Parameter consolidation
- EWC penalty term for model regularization
- Generic training loop that applies EWC loss

Note: This is a task-agnostic framework. The user must provide their own model and data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import os
import time
import gc
import shutil


class EWC:
    def __init__(self, fisher: dict, params: dict):
        """
        Stores Fisher Information and model parameters from a previous task.
        """
        self.fisher = {k: v.cpu() for k, v in fisher.items()}
        self.params = {k: v.cpu() for k, v in params.items()}

    @staticmethod
    def compute_fisher_and_params(model, dataloader, criterion, device, sample_size=None):
        """
        Estimates Fisher Information and stores parameter values after training on a task.
        """
        model.train()
        fisher = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}
        params = {n: p.clone().detach().cpu() for n, p in model.named_parameters() if p.requires_grad}

        total_samples = 0
        for x, y in dataloader:
            if sample_size and total_samples >= sample_size:
                break
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += (p.grad ** 2) * x.size(0)
            total_samples += x.size(0)

        fisher = {n: f / total_samples for n, f in fisher.items()}
        return fisher, params

    def penalty(self, model):
        """
        Computes the EWC penalty loss for the current model parameters.
        """
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher:
                _loss = self.fisher[n].to(p.device) * (p - self.params[n].to(p.device)) ** 2
                loss += _loss.sum()
        return loss


def train_with_ewc(model, dataloader, val_loader, optimizer, criterion,
                   ewc=None, lambda_ewc=0.4, scheduler=None,
                   device='cuda', num_epochs=100, model_saving_folder=None):
    """
    Generic training loop with optional EWC regularization.
    
    Arguments:
        model (nn.Module): Target model for training.
        dataloader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        optimizer (Optimizer): Optimizer instance.
        criterion (Loss): Loss function.
        ewc (EWC, optional): EWC object containing Fisher info and reference parameters.
        lambda_ewc (float): Scaling factor for EWC loss.
        scheduler (optional): Learning rate scheduler.
        device (str): Device for training.
        num_epochs (int): Number of training epochs.
        model_saving_folder (str, optional): If provided, best model will be saved to this path.
    """
    model.to(device)
    best_val_acc = 0.0

    if model_saving_folder:
        os.makedirs(model_saving_folder, exist_ok=True)
        best_model_path = os.path.join(model_saving_folder, "best_model.pth")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            if ewc:
                loss += (lambda_ewc / 2) * ewc.penalty(model)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)

        avg_loss = total_loss / len(dataloader.dataset)

        # === Validation ===
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs = model(x_val)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == y_val).sum().item()
                val_total += y_val.size(0)

        val_acc = val_correct / val_total
        print(f"[Epoch {epoch+1:03d}/{num_epochs}] Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2%}")

        # === Save Best Model ===
        if model_saving_folder and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        # === Scheduler Step ===
        if scheduler:
            scheduler.step(val_acc)

    gc.collect()
    torch.cuda.empty_cache()


"""
Example: Applying EWC in a Continual Learning Pipeline
-------------------------------------------------------

This example demonstrates how to protect knowledge from Period 2 while training on Period 3.

# ==== Period 2: After training ====
# Compute Fisher information and store parameter snapshot
from ewc import EWC

# Assumes you have a trained model and DataLoader from Period 2
fisher_dict, params_dict = EWC.compute_fisher_and_params(
    model=model,
    dataloader=period2_loader,     # DataLoader built from X_train_p2, y_train_p2
    criterion=nn.CrossEntropyLoss(),
    device=device
)

# Store EWC state
ewc = EWC(fisher=fisher_dict, params=params_dict)

# ==== Period 3: Train on new data with EWC protection ====

train_with_ewc(
    model=model,                          # Model reused or reloaded from Period 2
    dataloader=period3_loader,            # DataLoader for Period 3 training
    val_loader=period3_val_loader,        # Validation DataLoader
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    criterion=nn.CrossEntropyLoss(),
    ewc=ewc,                              # EWC state from Period 2
    lambda_ewc=1.0,                       # Regularization strength
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min'),
    device=device,
    num_epochs=100,
    model_saving_folder="./Trained_models/EWC/Period_3"
)
"""

"""
Tips for Integrating EWC Across Periods
---------------------------------------

1. After training each period, store Fisher & parameter snapshot:
   - Recommended: compute EWC using training data from that period (or a subset).
   - Save them if you need to resume later.

2. When moving to the next period:
   - Load model weights from the previous period.
   - Reconstruct EWC state (`EWC(...)`).
   - Start training with EWC penalty.

3. λ (lambda_ewc):
   - Higher values preserve old knowledge more strongly but may hinder plasticity.
   - Typical range: 0.1 ~ 10.0 depending on task complexity.

4. Supported model types:
   - Any PyTorch model with `.named_parameters()` and `.state_dict()` support.

5. Validation:
   - EWC does not require validation labels from past tasks, only parameter alignment.
   - EWC is compatible with single-head or multi-head output settings.

6. Storage:
   - EWC penalty only uses one snapshot of parameters and their corresponding Fisher matrix.
   - You do not need to store entire past datasets.
"""
