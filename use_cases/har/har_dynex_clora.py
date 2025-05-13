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


class LoRA(nn.Module):
    """
    LoRA module for injecting low-rank adaptation into linear layers.

    This module adds a learnable delta W = A @ B to an existing linear layer,
    where A ∈ R^(in_features x r), B ∈ R^(r x out_features), and r << min(in, out).

    Args:
        linear_layer (nn.Linear): Linear layer to adapt.
        rank (int): LoRA rank (low-dimensional subspace rank).
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
    HAR MLP model with support for multiple progressive LoRA adapters (DynEx-CLoRA).

    This architecture enables dynamic expansion by sequentially adding LoRA adapters
    to the second hidden layer (fc2), each representing a specific task or class group.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of units per hidden layer.
        output_size (int): Number of output classes.
        dropout (float): Dropout rate (default: 0.2).
        lora_rank (int): LoRA rank (default: 8).
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 dropout: float = 0.2, lora_rank: int = 8):
        super(HAR_MLP_LoRA_v2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lora_rank = lora_rank
        self.lora_adapters = nn.ModuleList()  # Store all LoRA modules (one per group)

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

    def add_lora_adapter(self):
        """
        Adds a new LoRA adapter to fc2. Each call appends a group representing
        a newly introduced class cluster (DynEx logic).
        """
        new_lora = LoRA(self.fc2, self.lora_rank).to(self.fc2.weight.device)
        self.lora_adapters.append(new_lora)
        print(f"✅ Added LoRA adapter to HAR_MLP_LoRA_v2 (fc2), total adapters: {len(self.lora_adapters)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x_base = self.fc2(x)

        if self.lora_adapters:
            lora_delta = sum(lora.A @ lora.B for lora in self.lora_adapters)
            adapted_weight = self.fc2.weight + lora_delta
            x = F.linear(x, adapted_weight, self.fc2.bias)
        else:
            x = x_base

        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def train_with_dynex_clora(model, teacher_model, output_size, criterion, optimizer,
                           X_train, y_train, X_val, y_val,
                           num_epochs, batch_size, alpha,
                           model_saving_folder, model_name,
                           stop_signal_file=None, scheduler=None,
                           period=None, stable_classes=None,
                           similarity_threshold=0.0,
                           class_features_dict=None, related_labels=None):
    model_name = model_name or 'dynex_clora_model'
    model_saving_folder = model_saving_folder or './saved_models'

    if model_saving_folder:
        if os.path.exists(model_saving_folder):
            shutil.rmtree(model_saving_folder)
        os.makedirs(model_saving_folder, exist_ok=True)

    device = auto_select_cuda_device()
    model.to(device)
    if teacher_model:
        teacher_model.to(device)
        teacher_model.eval()

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    start_time = time.time()
    best_results = []

    # === Class Feature Extraction ===
    model.eval()
    new_class_features = {}
    with torch.no_grad():
        for xb, yb in train_loader:
            x = model.relu1(model.bn1(model.fc1(xb)))
            for cls in torch.unique(yb):
                cls_mask = (yb == cls)
                cls_feat = x[cls_mask]
                if cls.item() not in new_class_features:
                    new_class_features[cls.item()] = []
                new_class_features[cls.item()].append(cls_feat)
    for cls in new_class_features:
        new_class_features[cls] = torch.cat(new_class_features[cls], dim=0).mean(dim=0)

    to_unfreeze = set()

    if period == 1:
        for p in model.fc1.parameters(): p.requires_grad = True
        for p in model.fc2.parameters(): p.requires_grad = True
        for p in model.fc3.parameters(): p.requires_grad = True
        for adapter in model.lora_adapters:
            for p in adapter.parameters():
                p.requires_grad = True

    elif period > 1 and class_features_dict:
        cosine_sim = torch.nn.CosineSimilarity(dim=0)
        new_lora_indices = []
        existing_classes = set(class_features_dict.keys())
        current_classes = set(new_class_features.keys())
        new_classes = current_classes - existing_classes

        for new_cls in new_classes:
            new_feat = new_class_features[new_cls]
            sims = [cosine_sim(new_feat.to(device), class_features_dict[old_cls].to(device)).item()
                    for old_cls in class_features_dict]

            matched = False
            for i, old_cls in enumerate(class_features_dict):
                if sims[i] >= similarity_threshold:
                    matched = True
                    for k, v in related_labels.items():
                        if old_cls in v:
                            to_unfreeze.add(k)
                            if new_cls not in related_labels[k]:
                                related_labels[k].append(new_cls)
            if not matched:
                model.add_lora_adapter()
                new_idx = len(model.lora_adapters) - 1
                related_labels[new_idx] = [new_cls]
                new_lora_indices.append(new_idx)

        for old_cls in existing_classes:
            if old_cls in new_class_features:
                sim_self = cosine_sim(new_class_features[old_cls].to(device), class_features_dict[old_cls].to(device)).item()
                if sim_self < similarity_threshold:
                    for k, v in related_labels.items():
                        if old_cls in v:
                            to_unfreeze.add(k)

        for adapter in model.lora_adapters:
            for p in adapter.parameters():
                p.requires_grad = False
        for p in model.fc2.parameters():
            p.requires_grad = False

        for idx in to_unfreeze:
            if isinstance(idx, int):
                for p in model.lora_adapters[idx].parameters():
                    p.requires_grad = True
            elif idx == "fc2":
                for p in model.fc2.parameters():
                    p.requires_grad = True
        for idx in new_lora_indices:
            for p in model.lora_adapters[idx].parameters():
                p.requires_grad = True
        for p in model.fc3.parameters():
            p.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=optimizer.param_groups[0]['lr'])

    for epoch in range(num_epochs):
        if stop_signal_file and os.path.exists(stop_signal_file):
            break

        model.train()
        epoch_loss = 0.0
        class_correct, class_total = {}, {}

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb).view(-1, output_size)
            yb_flat = yb.view(-1)
            ce_loss = criterion(logits, yb_flat)

            if teacher_model and stable_classes:
                with torch.no_grad():
                    teacher_logits = teacher_model(xb)
                student_stable = logits[:, stable_classes]
                teacher_stable = teacher_logits[:, stable_classes]
                distill_loss = F.mse_loss(student_stable, teacher_stable)
                total_loss = alpha * distill_loss + (1 - alpha) * ce_loss
            else:
                total_loss = ce_loss

            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item() * xb.size(0)
            compute_classwise_accuracy(logits, yb_flat, class_correct, class_total)

        train_loss = epoch_loss / len(train_loader.dataset)
        train_acc = {int(k): f"{(class_correct[k] / class_total[k]) * 100:.2f}%" if class_total[k] > 0 else "0.00%" for k in class_total}

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_class_correct, val_class_total = {}, {}
        with torch.no_grad():
            for xb, yb in val_loader:
                outputs = model(xb).view(-1, output_size)
                yb_flat = yb.view(-1)
                val_loss += criterion(outputs, yb_flat).item() * xb.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == yb_flat).sum().item()
                val_total += yb_flat.size(0)
                compute_classwise_accuracy(outputs, yb_flat, val_class_correct, val_class_total)

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_acc_cls = {int(k): f"{(val_class_correct[k]/val_class_total[k])*100:.2f}%" if val_class_total[k]>0 else "0.00%" for k in val_class_total}

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
            'model_path': model_path,
            'num_lora_adapters': len(model.lora_adapters),
            'related_labels': related_labels
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

    if class_features_dict is None:
        class_features_dict = {}
    class_features_dict.update(new_class_features)
    with open(os.path.join(model_saving_folder, "class_features.pkl"), 'wb') as f:
        pickle.dump(class_features_dict, f)

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
alpha = 0.1
similarity_threshold = 0.5



# === Period 1 Training ===
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

train_with_dynex_clora(
    model=model,
    teacher_model=None,
    output_size=output_size,
    criterion=criterion,
    optimizer=optimizer,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    num_epochs=num_epochs,
    batch_size=batch_size,
    alpha=alpha,
    model_saving_folder=model_saving_folder,
    model_name=model_name,
    stop_signal_file=stop_signal_file,
    scheduler=None,
    period=period,
    stable_classes=None,
    class_features_dict=None,
    related_labels={"fc2": [0, 1]},
    similarity_threshold=similarity_threshold
)

del X_train, y_train, X_val, y_val, model
gc.collect()
torch.cuda.empty_cache()


# === Period 2 Training ===
period = 2
device = auto_select_cuda_device()
X_train, y_train = period_datasets[period]["train"]
X_val, y_val     = period_datasets[period]["test"]
input_size       = X_train.shape[1]
output_size      = len(set(y_train))
stable_classes   = [0, 1]

prev_folder = "path/to/your/period1_folder"
model_saving_folder = "path/to/your/period2_folder"
os.makedirs(model_saving_folder, exist_ok=True)

with open(os.path.join(prev_folder, "class_features.pkl"), 'rb') as f:
    class_features_dict = pickle.load(f)

teacher_model = HAR_MLP_LoRA_v2(input_size, hidden_size, output_size - 1, dropout, lora_rank=lora_r).to(device)
checkpoint = torch.load(os.path.join(prev_folder, f"{model_name}_best.pth"), map_location=device)
num_lora_adapters = checkpoint.get("num_lora_adapters", 0)
related_labels = checkpoint.get("related_labels", {"fc2": [0, 1]})
for _ in range(num_lora_adapters):
    teacher_model.add_lora_adapter()
teacher_model.load_state_dict(checkpoint["model_state_dict"])

student_model = HAR_MLP_LoRA_v2(input_size, hidden_size, output_size, dropout, lora_rank=lora_r).to(device)
for _ in range(num_lora_adapters):
    student_model.add_lora_adapter()
student_dict = student_model.state_dict()
teacher_dict = teacher_model.state_dict()
for name in student_dict:
    if name in teacher_dict and student_dict[name].shape == teacher_dict[name].shape and not name.startswith("fc3"):
        student_dict[name].copy_(teacher_dict[name])
student_model.load_state_dict(student_dict)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

train_with_dynex_clora(
    model=student_model,
    teacher_model=teacher_model,
    output_size=output_size,
    criterion=criterion,
    optimizer=optimizer,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    num_epochs=num_epochs,
    batch_size=batch_size,
    alpha=alpha,
    model_saving_folder=model_saving_folder,
    model_name=model_name,
    stop_signal_file=stop_signal_file,
    scheduler=None,
    period=period,
    stable_classes=stable_classes,
    class_features_dict=class_features_dict,
    related_labels=related_labels,
    similarity_threshold=similarity_threshold
)

del X_train, y_train, X_val, y_val, teacher_model, student_model
gc.collect()
torch.cuda.empty_cache()


# === Period 3 Training ===
period = 3
device = auto_select_cuda_device()
X_train, y_train = period_datasets[period]["train"]
X_val, y_val     = period_datasets[period]["test"]
input_size       = X_train.shape[1]
output_size      = len(set(y_train))
stable_classes   = [0, 1, 2]

prev_folder = "path/to/your/period2_folder"
model_saving_folder = "path/to/your/period3_folder"
os.makedirs(model_saving_folder, exist_ok=True)

with open(os.path.join(prev_folder, "class_features.pkl"), 'rb') as f:
    class_features_dict = pickle.load(f)

teacher_model = HAR_MLP_LoRA_v2(input_size, hidden_size, output_size - 1, dropout, lora_rank=lora_r).to(device)
checkpoint = torch.load(os.path.join(prev_folder, f"{model_name}_best.pth"), map_location=device)
num_lora_adapters = checkpoint.get("num_lora_adapters", 0)
related_labels = checkpoint.get("related_labels", {})
for _ in range(num_lora_adapters):
    teacher_model.add_lora_adapter()
teacher_model.load_state_dict(checkpoint["model_state_dict"])

student_model = HAR_MLP_LoRA_v2(input_size, hidden_size, output_size, dropout, lora_rank=lora_r).to(device)
for _ in range(num_lora_adapters):
    student_model.add_lora_adapter()
student_dict = student_model.state_dict()
teacher_dict = teacher_model.state_dict()
for name in student_dict:
    if name in teacher_dict and student_dict[name].shape == teacher_dict[name].shape and not name.startswith("fc3"):
        student_dict[name].copy_(teacher_dict[name])
student_model.load_state_dict(student_dict)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

train_with_dynex_clora(
    model=student_model,
    teacher_model=teacher_model,
    output_size=output_size,
    criterion=criterion,
    optimizer=optimizer,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    num_epochs=num_epochs,
    batch_size=batch_size,
    alpha=alpha,
    model_saving_folder=model_saving_folder,
    model_name=model_name,
    stop_signal_file=stop_signal_file,
    scheduler=None,
    period=period,
    stable_classes=stable_classes,
    class_features_dict=class_features_dict,
    related_labels=related_labels,
    similarity_threshold=similarity_threshold
)

del X_train, y_train, X_val, y_val, teacher_model, student_model
gc.collect()
torch.cuda.empty_cache()


# === Period 4 Training ===
period = 4
device = auto_select_cuda_device()
X_train, y_train = period_datasets[period]["train"]
X_val, y_val     = period_datasets[period]["test"]
input_size       = X_train.shape[1]
output_size      = len(set(y_train))
stable_classes   = [0, 1, 3]

prev_folder = "path/to/your/period3_folder"
model_saving_folder = "path/to/your/period4_folder"
os.makedirs(model_saving_folder, exist_ok=True)

# === Load class features ===
class_features_path = os.path.join(prev_folder, "class_features.pkl")
with open(class_features_path, 'rb') as f:
    class_features_dict = pickle.load(f)

# === Load teacher model ===
teacher_model = HAR_MLP_LoRA_v2(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size - 2,  # Period 4 add WALKING_UPSTAIRS and WALKING_DOWNSTAIRS
    dropout=dropout,
    lora_rank=lora_r
).to(device)

checkpoint_path = os.path.join(prev_folder, f"{model_name}_best.pth")
checkpoint = torch.load(checkpoint_path, map_location=device)
num_lora_adapters = checkpoint.get("num_lora_adapters", 0)
related_labels = checkpoint.get("related_labels", {})

for _ in range(num_lora_adapters):
    teacher_model.add_lora_adapter()
teacher_model.load_state_dict(checkpoint["model_state_dict"])

# === Initialize student model ===
student_model = HAR_MLP_LoRA_v2(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    dropout=dropout,
    lora_rank=lora_r
).to(device)

for _ in range(num_lora_adapters):
    student_model.add_lora_adapter()

teacher_dict = teacher_model.state_dict()
student_dict = student_model.state_dict()
for name in student_dict:
    if name in teacher_dict and student_dict[name].shape == teacher_dict[name].shape and not name.startswith("fc3"):
        student_dict[name].copy_(teacher_dict[name])
student_model.load_state_dict(student_dict)

# === Optimizer, Loss, Scheduler ===
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = None

# === Training ===
train_with_dynex_clora(
    model=student_model,
    teacher_model=teacher_model,
    output_size=output_size,
    criterion=criterion,
    optimizer=optimizer,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    num_epochs=num_epochs,
    batch_size=batch_size,
    alpha=alpha,
    model_saving_folder=model_saving_folder,
    model_name=model_name,
    stop_signal_file=stop_signal_file,
    scheduler=scheduler,
    period=period,
    stable_classes=stable_classes,
    class_features_dict=class_features_dict,
    related_labels=related_labels,
    similarity_threshold=similarity_threshold
)

# === Cleanup ===
del X_train, y_train, X_val, y_val, teacher_model, student_model
gc.collect()
torch.cuda.empty_cache()
