# OS & file management
import os
import shutil
import subprocess
import time
import gc
import re
import copy
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, Tuple

# Scientific computation
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Set working directory
Working_directory = os.path.normpath("path/to/your/project")
os.chdir(Working_directory)

# Define base folder paths for HAR dataset
BASE_DIR = "path/to/your/har_dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

# Human Activity label mapping (UCI HAR original labels)
activity_label_map = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}

# Load feature and label data
X_train = np.loadtxt(os.path.join(TRAIN_DIR, "X_train.txt"))
y_train = np.loadtxt(os.path.join(TRAIN_DIR, "y_train.txt")).astype(int)
X_test = np.loadtxt(os.path.join(TEST_DIR, "X_test.txt"))
y_test = np.loadtxt(os.path.join(TEST_DIR, "y_test.txt")).astype(int)

# Period-to-original-label mapping
period_label_map = {
    1: [4, 5],           # SITTING, STANDING
    2: [4, 5, 1, 2, 3],  # + WALKING variants (merged)
    3: [4, 5, 1, 2, 3, 6],  # + LAYING
    4: [4, 5, 1, 2, 3, 6]   # same classes, but with walking variants separated
}

# Consistent label remapping across all periods (final label index)
final_class_map = {
    "SITTING": 0,
    "STANDING": 1,
    "WALKING": 2,
    "LAYING": 3,
    "WALKING_UPSTAIRS": 4,
    "WALKING_DOWNSTAIRS": 5
}

# Reverse mapping for readability
label_name_from_id = {v: k for k, v in final_class_map.items()}


# Convert raw label to remapped label for a specific period
def map_label(label, period):
    name = activity_label_map[label]
    if period < 4:
        if name.startswith("WALKING"):
            return final_class_map["WALKING"]
        else:
            return final_class_map[name]
    else:
        return final_class_map[name]


# Filter dataset by period and remap labels accordingly
def get_period_dataset(X, y, period):
    allowed_labels = period_label_map[period]
    mask = np.isin(y, allowed_labels)
    Xp = X[mask]
    yp = np.array([map_label(label, period) for label in y[mask]])
    return Xp, yp


# Print class distribution for diagnostics
def print_class_distribution(y, var_name: str, label_map: dict) -> None:
    y = np.array(y).flatten()
    unique, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    print(f"\n📦 Class Distribution for {var_name}")
    for i, c in zip(unique, counts):
        print(f"  ├─ Label {i:<2} ({label_map[i]:<20}) → {c:>5} samples ({(c/total)*100:>5.2f}%)")


# Generate remapped dataset for each period
period_datasets = {}

for period in range(1, 5):
    Xp_train, yp_train = get_period_dataset(X_train, y_train, period)
    Xp_test, yp_test = get_period_dataset(X_test, y_test, period)

    period_datasets[period] = {
        "train": (Xp_train, yp_train),
        "test": (Xp_test, yp_test)
    }

    print_class_distribution(yp_train, f"Period {period} (Train)", label_name_from_id)
    print_class_distribution(yp_test, f"Period {period} (Test)", label_name_from_id)

def ensure_folder(folder_path: str) -> None:
    """Create a folder if it does not exist."""
    os.makedirs(folder_path, exist_ok=True)

# CUDA device auto-selection
def auto_select_cuda_device(verbose=True):
    """Automatically select the least-used CUDA device, or fallback to CPU."""
    if not torch.cuda.is_available():
        if verbose:
            print("⚠️ No CUDA device available. Using CPU.")
        return torch.device("cpu")

    try:
        smi_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        memory_used = [int(x) for x in smi_output.strip().split('\n')]
        best_gpu = int(np.argmin(memory_used))
        if verbose:
            print(f"🎯 Auto-selected GPU: {best_gpu} ({memory_used[best_gpu]} MiB used)")
        return torch.device(f"cuda:{best_gpu}")
    except Exception as e:
        if verbose:
            print(f"⚠️ GPU detection failed. Falling back to cuda:0 ({e})")
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Utility Function
def compute_classwise_accuracy(preds, targets, class_correct, class_total):
    """
    Computes per-class accuracy statistics from raw logits and ground truth labels.

    Args:
        preds (Tensor): Raw model outputs, shape (B, num_classes).
        targets (Tensor): Ground truth labels, shape (B,).
        class_correct (dict): Dictionary to store correct counts per class.
        class_total (dict): Dictionary to store total counts per class.
    """
    preds = torch.argmax(preds, dim=-1)
    correct_mask = (preds == targets)
    for label in torch.unique(targets):
        label = label.item()
        label_mask = (targets == label)
        class_total[label] = class_total.get(label, 0) + label_mask.sum().item()
        class_correct[label] = class_correct.get(label, 0) + (correct_mask & label_mask).sum().item()

def compute_fwt_har(previous_model, init_model, X_val, y_val, known_classes, batch_size=64):
    """
    Computes Forward Transfer (FWT) for tabular HAR models (e.g., MLPs).

    Args:
        previous_model (nn.Module): Trained model from previous period.
        init_model (nn.Module): Model initialized at beginning of this period.
        X_val (np.ndarray): Validation features, shape (N, F).
        y_val (np.ndarray): Validation labels, shape (N,).
        known_classes (list[int]): List of class indices known up to this period.
        batch_size (int): Evaluation batch size.

    Returns:
        Tuple[float, float, float]: (FWT, acc_prev_model, acc_init_model)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    previous_model.to(device).eval()
    init_model.to(device).eval()

    # Filter validation set to only include known classes
    mask = np.isin(y_val, known_classes)
    X_known = X_val[mask]
    y_known = y_val[mask]

    if len(y_known) == 0:
        print(f"⚠️ No validation samples for known classes {known_classes}.")
        return None, None, None

    print(f"📋 Total samples for known classes {known_classes}: {len(y_known)}")

    dataset = TensorDataset(
        torch.tensor(X_known, dtype=torch.float32),
        torch.tensor(y_known, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=batch_size)

    correct_prev, correct_init, total = 0, 0, 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            out_prev = previous_model(xb)
            out_init = init_model(xb)

            preds_prev = torch.argmax(out_prev, dim=-1)
            preds_init = torch.argmax(out_init, dim=-1)

            correct_prev += (preds_prev == yb).sum().item()
            correct_init += (preds_init == yb).sum().item()
            total += yb.size(0)

    acc_prev = correct_prev / total
    acc_init = correct_init / total
    fwt_value = acc_prev - acc_init

    return fwt_value, acc_prev, acc_init

def print_model_info(model):
    """
    Prints total number of parameters and estimated model size in MB.

    Args:
        model (nn.Module): The PyTorch model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    param_size_bytes = total_params * 4  # Assume float32 = 4 bytes
    param_size_MB = param_size_bytes / (1024**2)

    print(f"- Total Parameters: {total_params}")
    print(f"- Model Size (float32): {param_size_MB:.2f} MB")

# Model
class HAR_MLP_v2(nn.Module):
    """
    A deeper MLP model for Human Activity Recognition (HAR) classification.

    This model consists of two hidden layers with batch normalization, dropout,
    and ReLU activation. Designed for flattened tabular input features.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of units in each hidden layer.
        output_size (int): Number of output classes.
        dropout (float): Dropout rate (default: 0.2).
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.2):
        super(HAR_MLP_v2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_size, output_size)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        return self.fc3(x)
    

class HAR_PNN_Column_fc2(nn.Module):
    """
    A new column for Progressive Neural Network (PNN) used in HAR tasks.
    This column includes lateral connections injected before the output layer (after fc2).

    Args:
        input_size (int): Input feature size (e.g., 561).
        hidden_size (int): Size of hidden layers.
        new_output_size (int): Number of output classes introduced in the current task.
        dropout (float): Dropout rate applied after each hidden layer.
    """
    def __init__(self, input_size: int, hidden_size: int, new_output_size: int, dropout: float = 0.2):
        super(HAR_PNN_Column_fc2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_size, new_output_size)
        self.lateral_adapter = nn.Linear(hidden_size, hidden_size)  # Lateral fusion after fc2

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, lateral_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Current input features (B, input_size)
            lateral_features (Tensor): Features from previous column (B, hidden_size)

        Returns:
            Tensor: Logits from this column (B, new_output_size)
        """
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        x = x + self.lateral_adapter(lateral_features)
        return self.fc3(x)


class ProgressiveNeuralNetwork_fc2(nn.Module):
    """
    Progressive Neural Network wrapper that adds a new column with lateral connections to a frozen base model.

    Args:
        base_model (nn.Module): The base model (can be another PNN or an MLP).
        new_column (HAR_PNN_Column_fc2): A new column to be added on top of the frozen base.
    """
    def __init__(self, base_model: nn.Module, new_column: HAR_PNN_Column_fc2):
        super(ProgressiveNeuralNetwork_fc2, self).__init__()
        self.base_model = base_model
        self.new_column = new_column

        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Recursively extracts the last hidden representation (after fc2) from base_model.

        Returns:
            Tensor: Base features to be used for lateral connection (B, hidden_size)
        """
        with torch.no_grad():
            if hasattr(self.base_model, "forward_features"):
                return self.base_model.forward_features(x)
            else:
                x = self.base_model.dropout1(self.base_model.relu1(self.base_model.bn1(self.base_model.fc1(x))))
                x = self.base_model.dropout2(self.base_model.relu2(self.base_model.bn2(self.base_model.fc2(x))))
                return x

    def get_logits(self, x: torch.Tensor, base_features: torch.Tensor) -> torch.Tensor:
        """
        Recursively collects logits from all previous columns.

        Returns:
            Tensor: Concatenated logits from all previous columns (B, ∑ old_output_sizes)
        """
        with torch.no_grad():
            if isinstance(self.base_model, ProgressiveNeuralNetwork_fc2):
                return self.base_model(x)
            else:
                return self.base_model.fc3(base_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass of the PNN wrapper.

        Returns:
            Tensor: Concatenated logits from base + new column (B, total_classes)
        """
        base_features = self.forward_features(x)
        base_logits = self.get_logits(x, base_features)
        new_logits = self.new_column(x, lateral_features=base_features)
        return torch.cat([base_logits, new_logits], dim=-1)

# Training Function
def train_with_pnn(
    model,
    output_size,
    criterion,
    optimizer,
    X_train,
    y_train,
    X_val,
    y_val,
    scheduler=None,
    num_epochs=10,
    batch_size=64,
    model_saving_folder=None,
    model_name=None,
    stop_signal_file=None,
    device=None
):
    """
    HAR training loop for Progressive Neural Networks (PNN).

    Arguments:
        model (nn.Module): ProgressiveNN instance with base frozen, new column trainable.
        output_size (int): Number of output classes.
        criterion (Loss): Loss function.
        optimizer (Optimizer): Optimizer instance.
        X_train, y_train: Training data (numpy arrays).
        X_val, y_val: Validation data (numpy arrays).
        scheduler (optional): Learning rate scheduler.
        num_epochs (int): Number of training epochs.
        batch_size (int): Mini-batch size.
        model_saving_folder (str, optional): Output folder to save best model.
        model_name (str, optional): Prefix for model file name.
        stop_signal_file (str, optional): Path to stop signal file.
        device (torch.device, optional): Device to train on.
    """
    device = device or auto_select_cuda_device()
    model_name = model_name or "model"
    model_saving_folder = model_saving_folder or "./saved_models"
    os.makedirs(model_saving_folder, exist_ok=True)
    best_model_path = os.path.join(model_saving_folder, f"{model_name}_best.pth")

    model.to(device)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    best_val_acc = 0.0
    best_record = None

    for epoch in range(num_epochs):
        if stop_signal_file and os.path.exists(stop_signal_file):
            break

        model.train()
        epoch_loss = 0.0
        class_correct, class_total = {}, {}

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).view(-1, output_size)
            y_batch = y_batch.view(-1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
            compute_classwise_accuracy(outputs, y_batch, class_correct, class_total)

        train_loss = epoch_loss / len(train_loader.dataset)
        train_acc = {
            int(c): f"{(class_correct[c] / class_total[c]) * 100:.2f}%"
            if class_total[c] > 0 else "0.00%"
            for c in sorted(class_total.keys())
        }

        # === Validation ===
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_class_correct, val_class_total = {}, {}

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch).view(-1, output_size)
                y_batch = y_batch.view(-1)
                val_loss += criterion(outputs, y_batch).item() * X_batch.size(0)
                predictions = torch.argmax(outputs, dim=-1)
                val_correct += (predictions == y_batch).sum().item()
                val_total += y_batch.size(0)
                compute_classwise_accuracy(outputs, y_batch, val_class_correct, val_class_total)

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_acc_cls = {
            int(c): f"{(val_class_correct[c] / val_class_total[c]) * 100:.2f}%"
            if val_class_total[c] > 0 else "0.00%"
            for c in sorted(val_class_total.keys())
        }

        print(f"[Epoch {epoch+1:03d}/{num_epochs}] Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2%}")

        current = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "train_classwise_accuracy": train_acc,
            "val_classwise_accuracy": val_acc_cls,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "model_path": best_model_path,
        }

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_record = current
            torch.save(current, best_model_path)

        if scheduler:
            scheduler.step(val_loss)

    torch.cuda.empty_cache()
    gc.collect()


# Common Configuration
batch_size = 64
num_epochs = 1000
learning_rate = 0.0001
weight_decay = 1e-5
hidden_size = 128
dropout = 0.2
stop_signal_file = "path/to/your/stop_signal_file.txt"

# Period 1
period = 1

X_train, y_train = period_datasets[period]["train"]
X_val, y_val     = period_datasets[period]["test"]

device = auto_select_cuda_device()
input_size = X_train.shape[1]
output_size = len(set(y_train))

model = HAR_MLP_v2(input_size, hidden_size, output_size, dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

model_saving_folder = "path/to/your/period1_model_folder"
ensure_folder(model_saving_folder)

train_with_pnn(
    model=model,
    output_size=output_size,
    criterion=criterion,
    optimizer=optimizer,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    scheduler=None,
    num_epochs=num_epochs,
    batch_size=batch_size,
    model_saving_folder=model_saving_folder,
    model_name="pnn_period1",
    stop_signal_file=stop_signal_file,
    device=device
)


# Period 2
period = 2

X_train, y_train = period_datasets[period]["train"]
X_val, y_val     = period_datasets[period]["test"]

device = auto_select_cuda_device()
input_size = X_train.shape[1]
output_size = len(set(y_train))

# === Freeze Period 1 ===
prev_model_path = "path/to/your/period1_best.pth"
frozen_model = HAR_MLP_v2(input_size, hidden_size, output_size - 1, dropout)
checkpoint = torch.load(prev_model_path, map_location=device)
frozen_model.load_state_dict(checkpoint["model_state_dict"])
frozen_model.to(device)
frozen_model.eval()
for p in frozen_model.parameters():
    p.requires_grad = False

# === New Column ===
new_column = HAR_PNN_Column_fc2(input_size, hidden_size, new_output_size=1)
model = ProgressiveNeuralNetwork_fc2(base_model=frozen_model, new_column=new_column).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.new_column.parameters(), lr=learning_rate, weight_decay=weight_decay)

model_saving_folder = "path/to/your/period2_model_folder"
ensure_folder(model_saving_folder)

train_with_pnn(
    model=model,
    output_size=output_size,
    criterion=criterion,
    optimizer=optimizer,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    scheduler=None,
    num_epochs=num_epochs,
    batch_size=batch_size,
    model_saving_folder=model_saving_folder,
    model_name="pnn_period2",
    stop_signal_file=stop_signal_file,
    device=device
)


# Period 3
period = 3

X_train, y_train = period_datasets[period]["train"]
X_val, y_val     = period_datasets[period]["test"]

device = auto_select_cuda_device()
input_size = X_train.shape[1]
output_size = len(set(y_train))

# === Restore Period 2 ===
base_model = HAR_MLP_v2(input_size, hidden_size, output_size - 2, dropout)
prev_column = HAR_PNN_Column_fc2(input_size, hidden_size, 1)
frozen_model = ProgressiveNeuralNetwork_fc2(base_model, prev_column)

checkpoint = torch.load("path/to/your/period2_best.pth", map_location=device)
frozen_model.load_state_dict(checkpoint["model_state_dict"])
frozen_model.to(device)
frozen_model.eval()
for p in frozen_model.parameters():
    p.requires_grad = False

# === Add new column ===
new_column = HAR_PNN_Column_fc2(input_size, hidden_size, 1)
model = ProgressiveNeuralNetwork_fc2(base_model=frozen_model, new_column=new_column).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.new_column.parameters(), lr=learning_rate, weight_decay=weight_decay)

model_saving_folder = "path/to/your/period3_model_folder"
ensure_folder(model_saving_folder)

train_with_pnn(
    model=model,
    output_size=output_size,
    criterion=criterion,
    optimizer=optimizer,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    scheduler=None,
    num_epochs=num_epochs,
    batch_size=batch_size,
    model_saving_folder=model_saving_folder,
    model_name="pnn_period3",
    stop_signal_file=stop_signal_file,
    device=device
)


# Period 4
period = 4

X_train, y_train = period_datasets[period]["train"]
X_val, y_val     = period_datasets[period]["test"]

device = auto_select_cuda_device()
input_size = X_train.shape[1]
output_size = len(set(y_train))

# === Rebuild frozen Period 3 model ===
base_model = HAR_MLP_v2(input_size, hidden_size, 2, dropout)
column_2 = HAR_PNN_Column_fc2(input_size, hidden_size, 1)
step1 = ProgressiveNeuralNetwork_fc2(base_model, column_2)

column_3 = HAR_PNN_Column_fc2(input_size, hidden_size, 1)
step2 = ProgressiveNeuralNetwork_fc2(step1, column_3)

checkpoint = torch.load("path/to/your/period3_best.pth", map_location=device)
step2.load_state_dict(checkpoint["model_state_dict"])
step2.to(device)
step2.eval()
for p in step2.parameters():
    p.requires_grad = False

# === Add new column for Period 4 ===
column_4 = HAR_PNN_Column_fc2(input_size, hidden_size, new_output_size=2)
model = ProgressiveNeuralNetwork_fc2(base_model=step2, new_column=column_4).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.new_column.parameters(), lr=learning_rate, weight_decay=weight_decay)

model_saving_folder = "path/to/your/period4_model_folder"
ensure_folder(model_saving_folder)

train_with_pnn(
    model=model,
    output_size=output_size,
    criterion=criterion,
    optimizer=optimizer,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    scheduler=None,
    num_epochs=num_epochs,
    batch_size=batch_size,
    model_saving_folder=model_saving_folder,
    model_name="pnn_period4",
    stop_signal_file=stop_signal_file,
    device=device
)
