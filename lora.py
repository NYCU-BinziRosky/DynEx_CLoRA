"""
lora.py — Standard Low-Rank Adaptation (LoRA) Framework for Continual Learning

This module defines a task-agnostic training loop for models equipped
with LoRA adapters. It assumes the model implements:
- `init_lora()`: to initialize LoRA adapters in specific layers.
- `get_trainable_parameters()`: to return only LoRA and classifier parameters for optimization.

Typical Workflow:
- Initialize model for the current period.
- Call `init_lora()` to prepare adapters.
- Selectively load weights from the previous period (excluding FC and LoRA).
- Train with LoRA adapters only.

Note: This file does not define the model architecture. See `ResNet18_1D_LoRA` or others.
"""

import os
import time
import gc
import re
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def train_with_lora(model, output_size, criterion, optimizer,
                    X_train, y_train, X_val, y_val,
                    scheduler=None, num_epochs=100, batch_size=64,
                    model_saving_folder=None, model_name=None,
                    stop_signal_file=None, device=None):
    """
    Standard LoRA training loop — only LoRA + FC layers are trainable.
    """
    print("\n🚀 'train_with_lora' started.")
    start_time = time.time()
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = model_name or 'lora_model'
    model_saving_folder = model_saving_folder or './saved_models'
    if os.path.exists(model_saving_folder):
        shutil.rmtree(model_saving_folder)
        print(f"✅ Removed existing folder: {model_saving_folder}")
    os.makedirs(model_saving_folder, exist_ok=True)

    model.to(device)

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

        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = torch.argmax(model(X_batch), dim=-1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{num_epochs} — Train Loss: {avg_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        if scheduler:
            scheduler.step(val_acc)

    training_time = time.time() - start_time
    print(f"\n🏁 Finished training. Best Val Acc: {best_val_acc*100:.2f}%")
    print(f"🕒 Training time: {training_time:.2f} seconds")

    gc.collect()
    torch.cuda.empty_cache()


"""
Example: Standard LoRA Training Across Periods
----------------------------------------------

# === Period 2: Reconstruct Period 1 model FIRST ===
model = ResNet18_1D_LoRA(input_channels=12, output_size=2, lora_rank=4)
model.init_lora()  # Init BEFORE loading weights

# Load previous weights (excluding FC and LoRA)
prev_checkpoint = torch.load(".../Period_1/best_model.pth")
prev_state = prev_checkpoint["model_state_dict"]
model_dict = model.state_dict()
filtered = {
    k: v for k, v in prev_state.items()
    if k in model_dict and model_dict[k].shape == v.shape and not (k.startswith("fc") or "lora_adapter" in k)
}
model.load_state_dict(filtered, strict=False)

# === Period 3+: Init LoRA FIRST, then load weights ===
model = ResNet18_1D_LoRA(...)
model.init_lora()
# Load only base weights
...

# === Optimizer and training ===
optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

from lora import train_with_lora

train_with_lora(
    model=model,
    output_size=output_size,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optimizer,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    scheduler=scheduler,
    num_epochs=100,
    batch_size=64,
    model_saving_folder="./Trained_models/Standard_LoRA/Period_3",
    model_name="ResNet18_1D_LoRA",
    stop_signal_file="./stop_training.txt",
    device=torch.device('cuda')
)
"""

"""
LoRA Training Notes
--------------------

1. init_lora():
   - Must be called once per model (starting Period 2).
   - Creates LoRAConv1d modules on specific internal layers (e.g., conv2 in BasicBlock).

2. Load weights carefully:
   - Avoid loading `fc.*` and `*.lora_adapter.*` parameters.
   - Filter based on `key not in fc` and `not in lora_adapter`.

3. get_trainable_parameters():
   - Should only return LoRA adapter weights and FC layer.
   - Helps optimizer focus on the efficient subset of parameters.

4. Model structure:
   - Use shared architecture (`ResNet18_1D_LoRA`) across periods.
   - The base model remains frozen; only LoRA and FC are updated.

5. Parameter Efficiency:
   - Training cost is low (few parameters).
   - Good for constrained environments or long sequences.

6. Period Setup Summary:
   - Period 1: Standard model training, no LoRA.
   - Period 2: Rebuild model, call `init_lora()`, load base weights.
   - Period ≥3: Call `init_lora()`, then load previous weights (same logic).
"""
