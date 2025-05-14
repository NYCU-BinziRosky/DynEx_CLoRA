"""
lora.py — Standard Low-Rank Adaptation (LoRA) continual learning framework

This module defines the training logic for models equipped with LoRA adapters.
It assumes the model implements:
- `init_lora()`: to initialize LoRA modules
- `get_trainable_parameters()`: to return LoRA + classifier parameters for optimization

Note: This is a task-agnostic implementation. The user must supply compatible model and data.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import gc


def train_with_lora(model, dataloader, val_loader, optimizer, criterion,
                    scheduler=None, num_epochs=100, device='cuda',
                    model_saving_folder=None):
    """
    LoRA training loop — updates only LoRA and classifier layers.
    
    Arguments:
        model (nn.Module): LoRA-compatible model.
        dataloader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        optimizer (Optimizer): Optimizer instance.
        criterion (Loss): Loss function (e.g., CrossEntropyLoss).
        scheduler (optional): Learning rate scheduler.
        num_epochs (int): Total number of training epochs.
        device (str): Computation device.
        model_saving_folder (str, optional): Save path for best checkpoint.
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
Example: Standard LoRA Training Across Periods
----------------------------------------------

# === Period 2: Rebuild and initialize adapters ===
model = YourLoRAModel(...)
model.init_lora()  # Call before loading weights

# Load weights excluding LoRA and classifier
prev_state = torch.load(".../Period_1/best_model.pth")
model_dict = model.state_dict()
filtered = {
    k: v for k, v in prev_state.items()
    if k in model_dict and model_dict[k].shape == v.shape and
       not (k.startswith("fc") or "lora_adapter" in k)
}
model.load_state_dict(filtered, strict=False)

# === Optimizer and Training ===
optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

train_with_lora(
    model=model,
    dataloader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=nn.CrossEntropyLoss(),
    scheduler=scheduler,
    num_epochs=100,
    device='cuda',
    model_saving_folder="./Trained_models/LoRA/Period_2"
)
"""

"""
LoRA Integration: Minimal Example
----------------------------------

# Example of a simple LoRA adapter inserted into a model

class LoRAConv1d(nn.Module):
    def __init__(self, conv_layer: nn.Conv1d, rank: int):
        super().__init__()
        self.conv = conv_layer
        self.rank = rank
        self.lora_A = nn.Parameter(torch.randn(conv_layer.out_channels, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, conv_layer.in_channels * conv_layer.kernel_size[0]))

    def forward(self, x):
        # Compute low-rank weight
        lora_weight = torch.matmul(self.lora_A, self.lora_B).view_as(self.conv.weight)
        # Add LoRA perturbation
        adapted_weight = self.conv.weight + lora_weight
        # This replaces conv2 inside the model with a residual low-rank weight
        return F.conv1d(x, adapted_weight, bias=self.conv.bias,
                        stride=self.conv.stride, padding=self.conv.padding,
                        dilation=self.conv.dilation, groups=self.conv.groups)


# Assume your model has blocks where conv2 is the target of LoRA
class MyModelWithLoRA(nn.Module):
    def __init__(self, base_channels=64, output_dim=5, lora_rank=4):
        super().__init__()
        self.conv1 = nn.Conv1d(1, base_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1)
        self.fc = nn.Linear(base_channels, output_dim)
        self.lora_adapter = None
        self.lora_rank = lora_rank

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.lora_adapter:
            x = self.lora_adapter(x)
        else:
            x = self.conv2(x)

        x = x.mean(dim=-1)
        return self.fc(x)

    def init_lora(self):
        # Initialize and attach LoRA adapter to conv2
        if self.lora_adapter is None:
            self.lora_adapter = LoRAConv1d(self.conv2, rank=self.lora_rank)
            print("LoRA adapter initialized")

    def get_trainable_parameters(self):
        # Return LoRA and FC parameters only
        params = []
        if self.lora_adapter:
            params += list(self.lora_adapter.parameters())
        params += list(self.fc.parameters())
        return params
"""

"""
LoRA Training Notes
--------------------

1. init_lora():
   - Must be called after model instantiation and before loading weights.
   - Creates LoRA modules inside base layers (e.g., conv or linear).

2. Weight Loading:
   - Only load base weights; skip keys containing `fc.` or `lora_adapter`.

3. get_trainable_parameters():
   - Should return a list of parameters from LoRA + final classifier only.

4. Freezing:
   - Ensure base weights remain frozen throughout training.

5. Model Structure:
   - Recommended to define a common base LoRA model for all periods (e.g., ResNet18_1D_LoRA).
"""

