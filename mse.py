"""
mse.py — Continual Learning with MSE Distillation

This module implements a continual learning strategy using knowledge distillation.
- Supports MSE loss between student and frozen teacher for selected classes
- Balances CrossEntropy and MSE loss via a tunable alpha
- Task-agnostic structure suitable for any classification model

Note: Users must supply their own model, data, and class stability logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
import gc


def train_with_mse(model, dataloader, val_loader, optimizer, criterion,
                   teacher_model=None, stable_classes=None, alpha=0.1,
                   scheduler=None, device='cuda', num_epochs=100, model_saving_folder=None):
    """
    Generic training loop with CrossEntropy + MSE distillation loss.

    Arguments:
        model (nn.Module): The student model for training.
        dataloader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        optimizer (Optimizer): Optimizer instance.
        criterion (Loss): CrossEntropy loss function.
        teacher_model (nn.Module, optional): Frozen teacher model.
        stable_classes (list[int], optional): Subset of class indices for distillation.
        alpha (float): Distillation strength (0.0 ~ 1.0).
        scheduler (optional): Learning rate scheduler.
        device (str): Device for training.
        num_epochs (int): Number of training epochs.
        model_saving_folder (str, optional): If provided, best model will be saved to this path.
    """
    model.to(device)
    if teacher_model:
        teacher_model.to(device)
        teacher_model.eval()

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
            student_logits = model(x_batch)
            ce_loss = criterion(student_logits, y_batch)

            if teacher_model and stable_classes:
                with torch.no_grad():
                    teacher_logits = teacher_model(x_batch)

                idx = torch.tensor(stable_classes, device=student_logits.device)
                s_stable = student_logits.index_select(dim=1, index=idx)
                t_stable = teacher_logits.index_select(dim=1, index=idx)
                distill_loss = F.mse_loss(s_stable, t_stable)
                loss = alpha * distill_loss + (1 - alpha) * ce_loss
            else:
                loss = ce_loss

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
Example: Applying MSE Distillation in Continual Learning
--------------------------------------------------------

# ==== Period 2: Load frozen teacher model ====
teacher_model = YourModel(...)
teacher_model.load_state_dict(torch.load(".../Period_1/best_model.pth"))
teacher_model.eval()

# ==== Period 2: Initialize student model ====
student_model = YourModel(...)

# ==== Period 2: Training ====

train_with_mse(
    model=student_model,
    dataloader=train_loader,
    val_loader=val_loader,
    optimizer=torch.optim.Adam(student_model.parameters(), lr=1e-3),
    criterion=nn.CrossEntropyLoss(),
    teacher_model=teacher_model,
    stable_classes=[0, 1, 2],  # indices of classes to distill
    alpha=0.2,
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min'),
    device='cuda',
    num_epochs=100,
    model_saving_folder="./Trained_models/MSE/Period_2"
)
"""

"""
Tips for MSE Distillation Across Periods
----------------------------------------

1. Distillation Focus:
   - Use `stable_classes` to focus on reliable targets from previous tasks.
   - Set to `None` if you do not want to distill in certain periods.

2. alpha (float):
   - Controls tradeoff between MSE distillation and CE loss.
   - Suggested range: 0.05 ~ 0.3 depending on confidence in teacher signal.

3. Teacher Model:
   - Should be frozen and share compatible output structure with the student.
   - You may skip loading incompatible FC layers.

4. Storage:
   - Only previous model weights and class selection (`stable_classes`) are needed.
"""
