"""
mse.py — Continual Learning with MSE Distillation

This module defines a generic training function that incorporates
Mean Squared Error (MSE) loss between a student model and a teacher model
for distillation in continual learning scenarios.

Features:
- Optional teacher model distillation for selected "stable classes"
- CrossEntropy + MSE combined loss using a tunable alpha
- Generic training loop with validation and best model saving

Note: This is a task-agnostic framework. Users should supply their own model and dataset.
"""

import os
import time
import shutil
import gc
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def train_with_mse(student_model, output_size, criterion, optimizer,
                   X_train, y_train, X_val, y_val,
                   stable_classes=None, teacher_model=None, alpha=0.1,
                   scheduler=None, num_epochs=10, batch_size=64,
                   model_saving_folder=None, model_name=None,
                   stop_signal_file=None, device=None):
    """
    Generic training loop with CrossEntropy + MSE distillation loss.
    """
    print("\n🚀 'train_with_mse' started.")
    start_time = time.time()
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = model_name or 'model'
    model_saving_folder = model_saving_folder or './saved_models'
    if os.path.exists(model_saving_folder):
        shutil.rmtree(model_saving_folder)
        print(f"✅ Removed existing folder: {model_saving_folder}")
    os.makedirs(model_saving_folder, exist_ok=True)

    student_model.to(device)
    if teacher_model:
        teacher_model.to(device)
        teacher_model.eval()

    # Wrap input into torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    best_val_acc = 0.0
    best_model_path = os.path.join(model_saving_folder, f"{model_name}_best.pth")

    for epoch in range(num_epochs):
        if stop_signal_file and os.path.exists(stop_signal_file):
            print("🛑 Stop signal detected.")
            break

        student_model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            student_logits = student_model(X_batch)
            ce_loss = criterion(student_logits, y_batch)

            if teacher_model and stable_classes:
                with torch.no_grad():
                    teacher_logits = teacher_model(X_batch)

                indices = torch.tensor(stable_classes, device=student_logits.device)
                s_stable = student_logits.index_select(dim=1, index=indices)
                t_stable = teacher_logits.index_select(dim=1, index=indices)
                mse_loss = F.mse_loss(s_stable, t_stable)
                loss = alpha * mse_loss + (1 - alpha) * ce_loss
            else:
                loss = ce_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        # Validation
        student_model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = student_model(X_batch)
                preds = torch.argmax(outputs, dim=-1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{num_epochs} — Train Loss: {avg_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student_model.state_dict(), best_model_path)

        if scheduler:
            scheduler.step(val_acc)

    training_time = time.time() - start_time
    print(f"\n🏁 Finished training. Best Val Acc: {best_val_acc*100:.2f}%")
    print(f"🕒 Training time: {training_time:.2f} seconds")

    gc.collect()
    torch.cuda.empty_cache()


"""
Example: MSE Distillation in Continual Learning
-----------------------------------------------

# === Period 2: Load teacher model from Period 1 ===
teacher_model = YourModel(...)
teacher_model.load_state_dict(torch.load(".../Period_1/best_model.pth"))
teacher_model.eval()

# === Period 2: Initialize student model ===
student_model = YourModel(...)

# === Period 2: Train with MSE Distillation ===
from mse import train_with_mse

train_with_mse(
    student_model=student_model,
    output_size=output_size,
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(student_model.parameters(), lr=1e-3),
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    stable_classes=[0, 2, 3],               # Define which old classes to distill
    teacher_model=teacher_model,
    alpha=0.1,                              # Distillation strength
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.9
    ),
    num_epochs=100,
    batch_size=64,
    model_saving_folder="./Trained_models/MSE/Period_2",
    model_name="ResNet18_1D",
    stop_signal_file="./stop_training.txt",
    device=torch.device('cuda')
)
"""

"""
Tips for MSE-based Knowledge Distillation
-----------------------------------------

1. alpha (float):
   - Controls the tradeoff between CE loss and distillation loss.
   - Typical values: 0.05 ~ 0.3. Use higher alpha to emphasize teacher guidance.

2. stable_classes:
   - Only a subset of classes from the previous period may be stable and reliable for distillation.
   - Set to None to disable teacher loss.

3. teacher_model:
   - Must have compatible architecture (output dimensions) with student model.
   - FC layer mismatch is allowed if you selectively load only matching layers.

4. Model structure:
   - You can use the same architecture for teacher and student or slightly updated versions.

5. Performance:
   - MSE distillation can help prevent forgetting of previous knowledge,
     especially for stable / frequently-seen classes.

6. Storage:
   - Only the previous period's best model and stable class list are needed.
"""
