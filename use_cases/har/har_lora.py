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

# === Set working directory manually if needed ===
# (This should point to your own project directory)
Working_directory = os.path.normpath("path/to/your/project")
os.chdir(Working_directory)

# === Define base folder paths for HAR dataset ===
# Replace with your actual dataset folder structure
BASE_DIR = "path/to/your/har_dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

# === Human Activity label mapping (UCI HAR original labels) ===
activity_label_map = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}

# === Load feature and label data ===
# You must place preprocessed txt files under your specified directory
X_train = np.loadtxt(os.path.join(TRAIN_DIR, "X_train.txt"))
y_train = np.loadtxt(os.path.join(TRAIN_DIR, "y_train.txt")).astype(int)
X_test = np.loadtxt(os.path.join(TEST_DIR, "X_test.txt"))
y_test = np.loadtxt(os.path.join(TEST_DIR, "y_test.txt")).astype(int)

# === Define label composition for each continual learning period ===
period_label_map = {
    1: [4, 5],           # SITTING, STANDING
    2: [4, 5, 1, 2, 3],  # Add merged WALKING
    3: [4, 5, 1, 2, 3, 6],  # Add LAYING
    4: [4, 5, 1, 2, 3, 6]   # Final: split walking variants
}

# === Define final consistent label IDs for training across all periods ===
final_class_map = {
    "SITTING": 0,
    "STANDING": 1,
    "WALKING": 2,
    "LAYING": 3,
    "WALKING_UPSTAIRS": 4,
    "WALKING_DOWNSTAIRS": 5
}

# === Reverse mapping: ID to label string ===
label_name_from_id = {v: k for k, v in final_class_map.items()}


# === Convert raw label to remapped label for a specific period ===
def map_label(label, period):
    name = activity_label_map[label]
    if period < 4:
        if name.startswith("WALKING"):
            return final_class_map["WALKING"]
        else:
            return final_class_map[name]
    else:
        return final_class_map[name]


# === Filter dataset by period and remap labels accordingly ===
def get_period_dataset(X, y, period):
    allowed_labels = period_label_map[period]
    mask = np.isin(y, allowed_labels)
    Xp = X[mask]
    yp = np.array([map_label(label, period) for label in y[mask]])
    return Xp, yp


# === Utility: Print class distribution for diagnostics ===
def print_class_distribution(y, var_name: str, label_map: dict) -> None:
    y = np.array(y).flatten()
    unique, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    print(f"\n📦 Class Distribution for {var_name}")
    for i, c in zip(unique, counts):
        print(f"  ├─ Label {i:<2} ({label_map[i]:<20}) → {c:>5} samples ({(c/total)*100:>5.2f}%)")


# === Generate remapped dataset for each period ===
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


# === GPU device selection (auto detect least memory usage) ===
def auto_select_cuda_device(verbose=True):
    """
    Automatically selects the CUDA GPU with the least memory usage.
    Falls back to CPU if no GPU is available.
    """
    if not torch.cuda.is_available():
        print("🚫 No CUDA GPU available. Using CPU.")
        return torch.device("cpu")

    try:
        smi_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        memory_used = [int(x) for x in smi_output.strip().split('\n')]
        best_gpu = int(np.argmin(memory_used))

        if verbose:
            print("🎯 Automatically selected GPU:")
            print(f"    - CUDA Device ID : {best_gpu}")
            print(f"    - Memory Used    : {memory_used[best_gpu]} MiB")
            print(f"    - Device Name    : {torch.cuda.get_device_name(best_gpu)}")
        return torch.device(f"cuda:{best_gpu}")
    except Exception as e:
        print(f"⚠️ Failed to auto-detect GPU. Falling back to cuda:0. ({e})")
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# === Device assignment ===
device = auto_select_cuda_device()


# === Per-class accuracy computation ===
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


# === Forward Transfer (FWT) computation for HAR ===
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

    print(f"\n### 🔍 FWT Debug Info (HAR):")
    print(f"- Total evaluated samples: {total}")
    print(f"- Correct (PrevModel): {correct_prev} / {total} → Acc = {acc_prev:.4f}")
    print(f"- Correct (InitModel): {correct_init} / {total} → Acc = {acc_init:.4f}")
    print(f"- FWT = Acc_prev - Acc_init = {fwt_value:.4f}")

    return fwt_value, acc_prev, acc_init


# === Utility: print total parameters and estimated model size ===
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


import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRA(nn.Module):
    """
    LoRA adapter for linear layers.

    This module adds a low-rank trainable delta to the weight matrix of a linear layer,
    enabling parameter-efficient adaptation.

    Args:
        linear_layer (nn.Linear): The original linear layer to adapt.
        rank (int): The rank of the LoRA decomposition (A @ B).
    """
    def __init__(self, linear_layer: nn.Linear, rank: int):
        super(LoRA, self).__init__()
        self.linear = linear_layer
        self.rank = rank

        in_features, out_features = linear_layer.weight.shape
        self.A = nn.Parameter(torch.zeros(in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_features))

        nn.init.normal_(self.A, mean=0, std=1)
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora_delta = self.A @ self.B
        adapted_weight = self.linear.weight + lora_delta
        return F.linear(x, adapted_weight, self.linear.bias)

    def parameters(self, recurse=True):
        return [self.A, self.B]


class HAR_MLP_LoRA_v2(nn.Module):
    """
    MLP model for HAR with LoRA integrated into the second hidden layer.

    This model supports LoRA adapter initialization from Period 2 onward and allows
    access to LoRA and FC3 parameters for selective fine-tuning.

    Args:
        input_size (int): Input feature dimension.
        hidden_size (int): Hidden layer size.
        output_size (int): Number of target classes.
        dropout (float): Dropout rate (default: 0.2).
        lora_rank (int): LoRA rank for the adapter on fc2 (default: 8).
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 dropout: float = 0.2, lora_rank: int = 8):
        super(HAR_MLP_LoRA_v2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_size, output_size)

        self.lora_rank = lora_rank
        self.lora_adapter = None  # Delayed initialization
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def init_lora(self):
        """
        Initialize the LoRA adapter on fc2 if not already initialized.
        Called once at Period 2.
        """
        if self.lora_adapter is None:
            self.lora_adapter = LoRA(self.fc2, self.lora_rank).to(next(self.parameters()).device)
            print("✅ Initialized LoRA adapter for fc2")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        if self.lora_adapter:
            lora_delta = self.lora_adapter.A @ self.lora_adapter.B
            adapted_weight = self.fc2.weight + lora_delta
            x = F.linear(x, adapted_weight, self.fc2.bias)
        else:
            x = self.fc2(x)

        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    def get_trainable_parameters(self):
        """
        Return trainable parameters and print LoRA/FC3 stats.

        Returns:
            List[nn.Parameter]: Only LoRA and FC3 parameters are returned for training.
        """
        lora_params = []
        lora_names = []
        fc_params = []
        fc_names = []

        total_params = sum(p.numel() for p in self.parameters())

        if self.lora_adapter:
            lora_params += [self.lora_adapter.A, self.lora_adapter.B]
            lora_names += ['lora_adapter.A', 'lora_adapter.B']

        for name, param in self.fc3.named_parameters():
            fc_params.append(param)
            fc_names.append(f"fc3.{name}")

        trainable_params = lora_params + fc_params
        lora_param_count = sum(p.numel() for p in lora_params)
        fc_param_count = sum(p.numel() for p in fc_params)
        trainable_param_count = lora_param_count + fc_param_count
        frozen_params = total_params - trainable_param_count

        print(f"📊 Parameter Statistics:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_param_count:,} ({trainable_param_count / total_params * 100:.2f}%)")
        print(f"    - LoRA parameters: {lora_param_count:,} ({lora_param_count / total_params * 100:.2f}%)")
        print(f"    - FC3 parameters: {fc_param_count:,} ({fc_param_count / total_params * 100:.2f}%)")
        print(f"  - Frozen parameters: {frozen_params:,} ({frozen_params / total_params * 100:.2f}%)")

        print(f"🧠 Trainable parameter names:")
        for name in lora_names:
            print(f"  ✅ {name} (LoRA)")
        for name in fc_names:
            print(f"  ✅ {name} (FC3)")

        return trainable_params


def train_with_standard_lora(model, output_size, criterion, optimizer,
                             X_train, y_train, X_val, y_val,
                             num_epochs=10, batch_size=64,
                             model_saving_folder=None, model_name=None,
                             stop_signal_file=None, scheduler=None,
                             period=None):
    """
    General training loop for Standard LoRA on tabular input.

    Args:
        model (nn.Module): LoRA-based MLP model.
        output_size (int): Number of output classes.
        criterion (Loss): Loss function (e.g., CrossEntropyLoss).
        optimizer (Optimizer): Optimizer instance.
        X_train, y_train, X_val, y_val (np.ndarray): Training/Validation data.
        num_epochs (int): Number of training epochs.
        batch_size (int): Mini-batch size.
        model_saving_folder (str): Directory to save model checkpoints.
        model_name (str): Prefix for saved model files.
        stop_signal_file (str): Optional path to stop training early.
        scheduler (optional): Learning rate scheduler.
        period (int): Current period (for reference or logging).
    """
    model_name = model_name or 'standard_lora_model'
    model_saving_folder = model_saving_folder or './saved_models'
    if os.path.exists(model_saving_folder):
        shutil.rmtree(model_saving_folder)
    os.makedirs(model_saving_folder, exist_ok=True)

    device = auto_select_cuda_device()
    model.to(device)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    best_results = []
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
            int(c): f"{(class_correct[c] / class_total[c]) * 100:.2f}%" if class_total[c] > 0 else "0.00%"
            for c in sorted(class_total.keys())
        }

        val_loss, val_correct, val_total = 0.0, 0, 0
        val_class_correct, val_class_total = {}, {}
        model.eval()
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
            int(c): f"{(val_class_correct[c] / val_class_total[c]) * 100:.2f}%" if val_class_total[c] > 0 else "0.00%"
            for c in sorted(val_class_total.keys())
        }

        model_path = os.path.join(model_saving_folder, f"{model_name}_epoch_{epoch+1}.pth")
        current = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'train_classwise_accuracy': train_acc,
            'val_classwise_accuracy': val_acc_cls,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'model_path': model_path
        }

        if len(best_results) < 5 or val_acc > best_results[-1]['val_accuracy']:
            if len(best_results) == 5:
                to_remove = best_results.pop()
                if os.path.exists(to_remove['model_path']):
                    os.remove(to_remove['model_path'])
            best_results.append(current)
            best_results.sort(key=lambda x: (x['val_accuracy'], x['epoch']), reverse=True)
            torch.save(current, model_path)

        if scheduler:
            scheduler.step(val_loss)

    final_model_path = os.path.join(model_saving_folder, f"{model_name}_final.pth")
    torch.save(current, final_model_path)

    if best_results:
        best_model = best_results[0]
        best_model_path = os.path.join(model_saving_folder, f"{model_name}_best.pth")
        torch.save(best_model, best_model_path)

    del X_train, y_train, X_val, y_val, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()


# === Common Configuration ===
batch_size = 64
stop_signal_file = "path/to/your/stop_signal_file.txt"
model_name = "HAR_MLP_LoRA_v2"
num_epochs = 1000
learning_rate = 0.0001
weight_decay = 1e-5
hidden_size = 128
dropout = 0.2
lora_r = 4

# === Period 1 Training (No LoRA) ===
period = 1
device = auto_select_cuda_device()
X_train, y_train = period_datasets[period]["train"]
X_val, y_val     = period_datasets[period]["test"]
input_size       = X_train.shape[1]
output_size      = len(set(y_train))

model_saving_folder = "path/to/your/period1_folder"
os.makedirs(model_saving_folder, exist_ok=True)

model = HAR_MLP_LoRA_v2(input_size, hidden_size, output_size, dropout, lora_rank=lora_r).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

train_with_standard_lora(
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
    model_name=model_name,
    stop_signal_file=stop_signal_file,
    period=period
)

del X_train, y_train, X_val, y_val, model
gc.collect()
torch.cuda.empty_cache()


# === Period 2 Training (Init LoRA AFTER loading previous model) ===
period = 2
device = auto_select_cuda_device()
X_train, y_train = period_datasets[period]["train"]
X_val, y_val     = period_datasets[period]["test"]
input_size       = X_train.shape[1]
output_size      = len(set(y_train))

model_saving_folder = "path/to/your/period2_folder"
os.makedirs(model_saving_folder, exist_ok=True)

# Load previous model (no LoRA)
prev_model_path = "path/to/your/period1_folder/HAR_MLP_LoRA_v2_best.pth"
checkpoint = torch.load(prev_model_path, map_location=device)
prev_state_dict = checkpoint["model_state_dict"]

# Initialize model and load base weights
model = HAR_MLP_LoRA_v2(input_size, hidden_size, output_size, dropout, lora_rank=lora_r).to(device)
model.load_state_dict({
    k: v for k, v in prev_state_dict.items()
    if not k.startswith("fc3") and not k.startswith("lora_adapter")
}, strict=False)
model.init_lora()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=learning_rate, weight_decay=weight_decay)

train_with_standard_lora(
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
    model_name=model_name,
    stop_signal_file=stop_signal_file,
    period=period
)

del X_train, y_train, X_val, y_val, model, checkpoint, prev_state_dict
gc.collect()
torch.cuda.empty_cache()


# === Period 3 & 4 Training (Init LoRA BEFORE loading) ===
for period in [3, 4]:
    device = auto_select_cuda_device()
    X_train, y_train = period_datasets[period]["train"]
    X_val, y_val     = period_datasets[period]["test"]
    input_size       = X_train.shape[1]
    output_size      = len(set(y_train))

    model_saving_folder = f"path/to/your/period{period}_folder"
    os.makedirs(model_saving_folder, exist_ok=True)

    prev_model_path = f"path/to/your/period{period-1}_folder/{model_name}_best.pth"
    checkpoint = torch.load(prev_model_path, map_location=device)
    prev_state_dict = checkpoint["model_state_dict"]

    model = HAR_MLP_LoRA_v2(input_size, hidden_size, output_size, dropout, lora_rank=lora_r).to(device)
    model.init_lora()
    model.load_state_dict({
        k: v for k, v in prev_state_dict.items()
        if not k.startswith("fc3")
    }, strict=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_with_standard_lora(
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
        model_name=model_name,
        stop_signal_file=stop_signal_file,
        period=period
    )

    del X_train, y_train, X_val, y_val, model, checkpoint, prev_state_dict
    gc.collect()
    torch.cuda.empty_cache()