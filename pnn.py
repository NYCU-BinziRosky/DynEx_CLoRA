"""
pnn.py — Progressive Neural Network (PNN) continual learning framework

This module defines a PNN-based architecture where new task-specific columns are
added sequentially. Lateral connections allow knowledge reuse from frozen columns.

This framework includes:
- Column module (`PNNColumn`) with lateral fusion
- Wrapper (`ProgressiveNN`) for combining frozen base and new columns
- Training function that updates only the new column

Note: This is a task-agnostic framework. Users must define their own backbone structure.
"""

import os
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class PNNColumn(nn.Module):
    """
    Defines a new column in PNN for the current task.
    Receives lateral features from the base model and fuses them with its own transformation.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.adapter = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, lateral_features: torch.Tensor) -> torch.Tensor:
        """
        Performs lateral fusion of base features and new input.
        """
        fused = self.adapter(x) + lateral_features
        fused = self.relu(fused)
        fused = self.dropout(fused)
        return self.classifier(fused)


class ProgressiveNN(nn.Module):
    """
    Wraps a frozen base model and a trainable new column for the current task.
    Supports lateral feature transfer and logits fusion.
    """
    def __init__(self, base_model: nn.Module, new_column: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.new_column = new_column
        for p in self.base_model.parameters():
            p.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts base features for lateral fusion.
        """
        return self.base_model.forward_features(x) if hasattr(self.base_model, 'forward_features') else self.base_model(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs inference using both base and new columns, and merges predictions.
        """
        base_feats = self.forward_features(x)
        with torch.no_grad():
            base_logits = self.base_model(x)
        new_logits = self.new_column(x, base_feats)
        return torch.cat([base_logits, new_logits], dim=-1)


def train_with_pnn(model, dataloader, val_loader, optimizer, criterion,
                   scheduler=None, num_epochs=100, device='cuda',
                   model_saving_folder=None):
    """
    Training loop for Progressive Neural Networks. Only the new column is updated.

    Arguments:
        model (nn.Module): ProgressiveNN instance (base model frozen, new column trainable).
        dataloader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        optimizer (Optimizer): Optimizer instance.
        criterion (Loss): Classification loss (e.g., CrossEntropyLoss).
        scheduler (optional): Learning rate scheduler.
        num_epochs (int): Total number of training epochs.
        device (str): CUDA or CPU.
        model_saving_folder (str, optional): Where to save the best model (new column only).
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
                preds = model(x_val).argmax(dim=1)
                val_correct += (preds == y_val).sum().item()
                val_total += y_val.size(0)

        val_acc = val_correct / val_total
        print(f"[Epoch {epoch+1:03d}/{num_epochs}] Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2%}")

        if model_saving_folder and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        if scheduler:
            scheduler.step(val_acc)

    gc.collect()
    torch.cuda.empty_cache()


"""
Example: PNN Training Across Tasks
----------------------------------

# === Load frozen model from previous period ===
prev_model = ProgressiveNN(...)  # contains frozen base model
prev_model.load_state_dict(torch.load(".../Period_2/best_model.pth"))
for p in prev_model.parameters():
    p.requires_grad = False

# === Create new column ===
new_column = PNNColumn(input_dim=128, output_dim=2)

# === Assemble full PNN ===
model = ProgressiveNN(base_model=prev_model, new_column=new_column)

# === Train new column ===

train_with_pnn(
    model=model,
    dataloader=period3_loader,
    val_loader=period3_val_loader,
    optimizer=torch.optim.Adam(model.new_column.parameters(), lr=1e-3),
    criterion=nn.CrossEntropyLoss(),
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min'),
    num_epochs=100,
    device='cuda',
    model_saving_folder="./Trained_models/PNN/Period_3"
)
"""

"""
Tips for PNN Usage
------------------

1. Only train the new column. Freeze all previous parameters.
2. Use `model.new_column.parameters()` when constructing the optimizer.
3. Ensure base model implements `forward_features()` for lateral reuse.
4. Expansion is recursive: PNNs grow a new column every task.
5. Output dimensions of base + new column must match full class set.
6. This approach increases memory and model size linearly over tasks.
7. This PNNColumn assumes the input and lateral features have the same dimensionality. Modify `adapter` as needed.
"""
