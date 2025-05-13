"""
pnn.py — Progressive Neural Network (PNN) for Continual Learning

This module defines a progressive architecture where new task-specific
columns are added while freezing previous knowledge, with lateral
connections to encourage feature reuse.

Includes:
- Column definition (`PNNColumn`)
- Wrapper for full PNN (`ProgressiveNN`)
- Training loop (`train_with_pnn`)

Note: This is a task-agnostic framework. The user is responsible for assembling the correct model structure.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import time
import shutil
import gc
import re


class PNNColumn(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim=512, dropout=0.2):
        super(PNNColumn, self).__init__()
        self.adapter = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, lateral_features: torch.Tensor) -> torch.Tensor:
        fused = lateral_features + self.adapter(x)
        fused = self.relu(fused)
        fused = self.dropout(fused)
        return self.classifier(fused)


class ProgressiveNN(nn.Module):
    def __init__(self, base_model: nn.Module, new_column: PNNColumn):
        super(ProgressiveNN, self).__init__()
        self.base_model = base_model
        self.new_column = new_column
        for p in self.base_model.parameters():
            p.requires_grad = False

    def forward_features(self, x):
        return self.base_model.forward_features(x) if hasattr(self.base_model, 'forward_features') else self.base_model(x)

    def get_base_logits(self, x):
        with torch.no_grad():
            return self.base_model(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_features = self.forward_features(x)
        base_logits = self.get_base_logits(x)
        new_logits = self.new_column(x, lateral_features=base_features)
        return torch.cat([base_logits, new_logits], dim=-1)


def train_with_pnn(model, output_size, criterion, optimizer,
                   X_train, y_train, X_val, y_val,
                   scheduler=None, num_epochs=100, batch_size=64,
                   model_saving_folder=None, model_name=None,
                   stop_signal_file=None, period=None,
                   device=None):
    """
    Generic PNN training loop. Only the new column is trained.
    """
    print("\n🚀 'train_with_pnn' started.")
    start_time = time.time()
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model_name = model_name or 'model'
    model_saving_folder = model_saving_folder or './saved_models'
    if os.path.exists(model_saving_folder):
        shutil.rmtree(model_saving_folder)
        print(f"✅ Removed existing folder: {model_saving_folder}")
    os.makedirs(model_saving_folder, exist_ok=True)

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
Example: Using PNN in Period 3
------------------------------

# Step 1: Load frozen PNN from Period 2
base_model = ProgressiveNN(...)
base_model.load_state_dict(torch.load(".../Period_2/best_model.pth"))
for p in base_model.parameters():
    p.requires_grad = False
base_model.eval()

# Step 2: Create new column for new classes
new_column = PNNColumn(input_dim=1024, output_dim=2)

# Step 3: Wrap into new PNN
model = ProgressiveNN(base_model=base_model, new_column=new_column)

# Step 4: Train
from pnn import train_with_pnn

train_with_pnn(
    model=model,
    output_size=total_classes_up_to_now,
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.new_column.parameters(), lr=1e-3),
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.9
    ),
    num_epochs=100,
    batch_size=64,
    model_saving_folder="./Trained_models/PNN/Period_3",
    model_name="ResNet18_PNN",
    stop_signal_file="./stop_training.txt",
    period=3,
    device=torch.device('cuda')
)
"""

"""
Tips for Progressive Neural Networks
------------------------------------

1. PNNColumn:
   - Must support lateral fusion: it takes both the input and lateral features.
   - Add dropout + relu for nonlinearity and regularization.

2. base_model:
   - Can be a standard model (Period 1) or another ProgressiveNN (Period ≥2).
   - Recursively wraps earlier knowledge while freezing parameters.

3. forward_features():
   - Must be defined in your base model (e.g., ResNet18_1D).
   - This should output a high-level feature vector for lateral reuse.

4. Partial loading:
   - It's fine if some keys (like FC layers) mismatch between periods.
   - Only require consistency in shared feature structure.

5. Training:
   - Only train the `new_column`; freeze everything else.
   - `model.new_column.parameters()` is the correct target for your optimizer.

6. Memory cost:
   - Increases linearly with number of tasks.
   - Each new period adds a full set of layers in `new_column`.
"""