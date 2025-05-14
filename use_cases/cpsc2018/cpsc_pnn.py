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

# Update this to your project root
BASE_DIR = "./cpsc2018"
save_dir = os.path.join(BASE_DIR, "processed")  # Path to the preprocessed `.npy` files (one for each continual learning period).
ECG_PATH = os.path.join(BASE_DIR, "datas")      # Directory containing original `.mat` and `.hea` files.
MAX_LEN = 5000  # ECG signal sequence length

# NOMED code mapping
snomed_map = {
    "426783006": "NSR",
    "270492004": "I-AVB",
    "164889003": "AF",
    "164909002": "LBBB",
    "59118001":  "RBBB",
    "284470004": "PAC",
    "164884008": "PVC",
    "429622005": "STD",
    "164931005": "STE"
}

# Class label mapping per training period
period_label_map = {
    1: {"NSR": 0, "OTHER": 1},
    2: {"NSR": 0, "I-AVB": 2, "AF": 3, "OTHER": 1},
    3: {"NSR": 0, "I-AVB": 2, "AF": 3, "LBBB": 4, "RBBB": 5, "OTHER": 1},
    4: {"NSR": 0, "I-AVB": 2, "AF": 3, "LBBB": 4, "RBBB": 5, "PAC": 6, "PVC": 7, "STD": 8, "STE": 9}
}

# Utility Function
def print_class_distribution(y, label_map):
    y = np.array(y).flatten()
    total = len(y)
    all_labels = sorted(label_map.values())
    print("\n📊 Class Distribution")
    for lbl in all_labels:
        count = np.sum(y == lbl)
        label = [k for k, v in label_map.items() if v == lbl]
        name = label[0] if label else str(lbl)
        print(f"  - Label {lbl:<2} ({name:<10}) → {count:>5} samples ({(count/total)*100:5.2f}%)")

def ensure_folder(folder_path: str) -> None:
    """Create a folder if it does not exist."""
    os.makedirs(folder_path, exist_ok=True)

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

def compute_classwise_accuracy(student_logits_flat, y_batch, class_correct, class_total):
    """Compute per-class accuracy."""
    if student_logits_flat.device != y_batch.device:
        raise ValueError("Tensors must be on the same device")

    predictions = torch.argmax(student_logits_flat, dim=-1)
    correct_mask = (predictions == y_batch)
    unique_labels = torch.unique(y_batch)

    for label in unique_labels:
        label = label.item()
        if label not in class_total:
            class_total[label] = 0
            class_correct[label] = 0
        label_mask = (y_batch == label)
        class_total[label] += label_mask.sum().item()
        class_correct[label] += (label_mask & correct_mask).sum().item()

def get_model_parameter_info(model):
    """Return total parameter count and size in MB for the given model."""
    total_params = sum(p.numel() for p in model.parameters())
    param_size_MB = total_params * 4 / (1024**2)
    return total_params, param_size_MB

def compute_class_weights(y: np.ndarray, num_classes: int, exclude_classes: list = None) -> torch.Tensor:
    """Compute inverse-frequency class weights with optional exclusions."""
    exclude_classes = set(exclude_classes or [])
    class_sample_counts = np.bincount(y, minlength=num_classes)
    total_samples = len(y)

    weights = np.zeros(num_classes, dtype=np.float32)
    for cls in range(num_classes):
        if cls in exclude_classes:
            weights[cls] = 0.0
        else:
            weights[cls] = total_samples / (class_sample_counts[cls] + 1e-6)

    valid_mask = np.array([cls not in exclude_classes for cls in range(num_classes)])
    norm_sum = weights[valid_mask].sum()
    if norm_sum > 0:
        weights[valid_mask] /= norm_sum

    print("\n📊 Class Weights (normalized):")
    for i, w in enumerate(weights):
        status = " (excluded)" if i in exclude_classes else ""
        print(f"  - Class {i}: {w:.4f}{status}")

    return torch.tensor(weights, dtype=torch.float32)

def compute_fwt_ecg(previous_model, init_model, X_val, y_val, known_classes, batch_size=64):
    """
    Compute Forward Transfer (FWT) score on ECG classification task.
    FWT = Acc(previous model on known classes) - Acc(init model on known classes)
    
    Args:
        previous_model (nn.Module): Trained model from previous period.
        init_model (nn.Module): Untrained model with same architecture.
        X_val (np.ndarray): Validation inputs of shape (B, T, C).
        y_val (np.ndarray): Ground truth labels.
        known_classes (List[int]): Class indices used to compute FWT.
        batch_size (int): Evaluation batch size.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    previous_model.to(device).eval()
    init_model.to(device).eval()

    mask = np.isin(y_val, known_classes)
    X_known, y_known = X_val[mask], y_val[mask]

    if len(y_known) == 0:
        print(f"⚠️ No samples found for known classes: {known_classes}")
        return None, None, None

    print(f"\n📋 FWT Evaluation | Known classes: {known_classes}, Samples: {len(y_known)}")

    dataset = TensorDataset(
        torch.tensor(X_known, dtype=torch.float32),
        torch.tensor(y_known, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=batch_size)

    correct_prev, correct_init, total = 0, 0, 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred_prev = torch.argmax(previous_model(xb), dim=-1)
            pred_init = torch.argmax(init_model(xb), dim=-1)

            correct_prev += (pred_prev == yb).sum().item()
            correct_init += (pred_init == yb).sum().item()
            total += yb.size(0)

    acc_prev = 100 * correct_prev / total
    acc_init = 100 * correct_init / total
    fwt = acc_prev - acc_init

    print(f"🔍 Accuracy (prev): {acc_prev:.2f}% | Accuracy (init): {acc_init:.2f}% | FWT: {fwt:.2f}%")
    return fwt, acc_prev, acc_init

# Model
class BasicBlock1d(nn.Module):
    """Residual block for 1D convolution (ResNet)."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)

class ResNet18_1D(nn.Module):
    """
    ResNet18 1D variant for ECG signal classification with support for PNN forward_features.
    """
    def __init__(self, input_channels=12, output_size=9, inplanes=64):
        super().__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv1d(input_channels, inplanes, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock1d, 64, 2)
        self.layer2 = self._make_layer(BasicBlock1d, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock1d, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock1d, 512, 2, stride=2)

        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.2)

        self.output_size = output_size
        if output_size > 0:
            self.fc = nn.Linear(512 * 2, output_size)
        else:
            self.fc = None

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        avg_out = self.adaptiveavgpool(x)
        max_out = self.adaptivemaxpool(x)
        x = torch.cat((avg_out, max_out), dim=1).view(x.size(0), -1)
        x = self.dropout(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.fc is not None:
            return self.fc(x)
        return x

# PNN Column Module
class ECG_PNN_Column(nn.Module):
    """
    One column in a Progressive Neural Network for ECG classification.
    Accepts lateral features from previous columns and adds them via lateral adapter.
    """
    def __init__(self, input_channels: int, output_dim: int):
        super().__init__()
        self.feature_extractor = ResNet18_1D(input_channels=input_channels, output_size=0)
        self.lateral_adapter = nn.Linear(1024, 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(1024, output_dim)

    def forward(self, x: torch.Tensor, lateral_features: torch.Tensor) -> torch.Tensor:
        new_features = self.feature_extractor.forward_features(x)  # shape: (B, 1024)
        fused = new_features + self.lateral_adapter(lateral_features)
        fused = self.relu(fused)
        fused = self.dropout(fused)
        return self.classifier(fused)

# Progressive Neural Network Wrapper
class ECG_ProgressiveNN(nn.Module):
    """
    Wrapper for Progressive Neural Network.
    base_model can be a normal ResNet or a previously stacked ECG_ProgressiveNN.
    """
    def __init__(self, base_model: nn.Module, new_column: ECG_PNN_Column):
        super().__init__()
        self.base_model = base_model
        self.new_column = new_column

        # Freeze all base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

    def extract_base_features(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.base_model, ECG_ProgressiveNN):
            return self.base_model.forward_features(x)
        else:
            return self.base_model.forward_features(x)

    def get_base_logits(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if isinstance(self.base_model, ECG_ProgressiveNN):
                return self.base_model(x)
            else:
                return self.base_model(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_base_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_features = self.extract_base_features(x)
        base_logits = self.get_base_logits(x)
        new_logits = self.new_column(x, lateral_features=base_features)
        return torch.cat([base_logits, new_logits], dim=-1)

# Training Function
def train_with_pnn_ecg(
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
    period=None,
    device=None
):
    """
    ECG training loop for Progressive Neural Networks (PNN).
    
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
        period (int, optional): Current continual learning period index.
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
            outputs = model(X_batch)
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
                outputs = model(X_batch)
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



# ==========================================
# 🚨 Note:
# Period 1 model is shared across all methods.
# Ensure it is trained separately and referenced here.
# ==========================================

# ================================
# 📌 Period 2: PNN Training
# ================================
period = 2

stop_signal_file = os.path.join(BASE_DIR, "stop_training.txt")
model_saving_folder = os.path.join(BASE_DIR, "PNN_CIL_v2", f"Period_{period}")
ensure_folder(model_saving_folder)

X_train = np.load(os.path.join(save_dir, f"X_train_p{period}.npy"))
y_train = np.load(os.path.join(save_dir, f"y_train_p{period}.npy"))
X_val   = np.load(os.path.join(save_dir, f"X_test_p{period}.npy"))
y_val   = np.load(os.path.join(save_dir, f"y_test_p{period}.npy"))

device = auto_select_cuda_device()
input_channels = X_train.shape[2]
output_size = len(np.unique(y_train))
new_output_size = 2
prev_output_size = output_size - new_output_size

prev_model_path = os.path.join(BASE_DIR, "ResNet18_Selection", "ResNet18_big_inplane_v1", "ResNet18_big_inplane_1D_best.pth")
prev_checkpoint = torch.load(prev_model_path, map_location=device)

frozen_model = ResNet18_1D(input_channels=input_channels, output_size=prev_output_size)
frozen_model.load_state_dict(prev_checkpoint["model_state_dict"], strict=True)
frozen_model.to(device)
frozen_model.eval()
for p in frozen_model.parameters():
    p.requires_grad = False

new_column = ECG_PNN_Column(input_dim=1024, output_dim=new_output_size, hidden_dim=512, dropout=0.2)
model = ECG_ProgressiveNN(base_model=frozen_model, new_column=new_column).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.new_column.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=10)

train_with_pnn_ecg(
    model=model,
    output_size=output_size,
    criterion=criterion,
    optimizer=optimizer,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    scheduler=scheduler,
    num_epochs=200,
    batch_size=64,
    model_saving_folder=model_saving_folder,
    model_name="ResNet18_PNN",
    stop_signal_file=stop_signal_file,
    period=period,
    device=device
)

del X_train, y_train, X_val, y_val, frozen_model, new_column, model
gc.collect()
torch.cuda.empty_cache()

# ================================
# 📌 Period 3: PNN Training
# ================================
period = 3

stop_signal_file = os.path.join(BASE_DIR, "stop_training.txt")
model_saving_folder = os.path.join(BASE_DIR, "PNN_CIL_v2", f"Period_{period}")
ensure_folder(model_saving_folder)

X_train = np.load(os.path.join(save_dir, f"X_train_p{period}.npy"))
y_train = np.load(os.path.join(save_dir, f"y_train_p{period}.npy"))
X_val   = np.load(os.path.join(save_dir, f"X_test_p{period}.npy"))
y_val   = np.load(os.path.join(save_dir, f"y_test_p{period}.npy"))

device = auto_select_cuda_device()
input_channels = X_train.shape[2]
output_size = len(np.unique(y_train))
new_output_size = 2

base_model = ResNet18_1D(input_channels=input_channels, output_size=2)
prev_column = ECG_PNN_Column(input_dim=1024, output_dim=2, hidden_dim=512, dropout=0.2)
frozen_model = ECG_ProgressiveNN(base_model=base_model, new_column=prev_column)

prev_model_path = os.path.join(BASE_DIR, "PNN_CIL_v2", "Period_2", "ResNet18_PNN_best.pth")
prev_checkpoint = torch.load(prev_model_path, map_location=device)
frozen_model.load_state_dict(prev_checkpoint["model_state_dict"], strict=True)
frozen_model.to(device)
frozen_model.eval()
for p in frozen_model.parameters():
    p.requires_grad = False

new_column = ECG_PNN_Column(input_dim=1024, output_dim=new_output_size, hidden_dim=512, dropout=0.2)
model = ECG_ProgressiveNN(base_model=frozen_model, new_column=new_column).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.new_column.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=10)

train_with_pnn_ecg(
    model=model,
    output_size=output_size,
    criterion=criterion,
    optimizer=optimizer,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    scheduler=scheduler,
    num_epochs=200,
    batch_size=64,
    model_saving_folder=model_saving_folder,
    model_name="ResNet18_PNN",
    stop_signal_file=stop_signal_file,
    period=period,
    device=device
)

del X_train, y_train, X_val, y_val, base_model, prev_column, frozen_model, new_column, model
gc.collect()
torch.cuda.empty_cache()

# ================================
# 📌 Period 4: PNN Training
# ================================
period = 4

stop_signal_file = os.path.join(BASE_DIR, "stop_training.txt")
model_saving_folder = os.path.join(BASE_DIR, "PNN_CIL_v2", f"Period_{period}")
ensure_folder(model_saving_folder)

X_train = np.load(os.path.join(save_dir, f"X_train_p{period}.npy"))
y_train = np.load(os.path.join(save_dir, f"y_train_p{period}.npy"))
X_val   = np.load(os.path.join(save_dir, f"X_test_p{period}.npy"))
y_val   = np.load(os.path.join(save_dir, f"y_test_p{period}.npy"))

device = auto_select_cuda_device()
input_channels = X_train.shape[2]
output_size = int(np.max(y_train)) + 1
new_output_size = 4

base_model = ResNet18_1D(input_channels=input_channels, output_size=2)
prev_column_2 = ECG_PNN_Column(input_dim=1024, output_dim=2, hidden_dim=512, dropout=0.2)
frozen_step1 = ECG_ProgressiveNN(base_model=base_model, new_column=prev_column_2)

prev_column_3 = ECG_PNN_Column(input_dim=1024, output_dim=2, hidden_dim=512, dropout=0.2)
frozen_step2 = ECG_ProgressiveNN(base_model=frozen_step1, new_column=prev_column_3)

prev_model_path = os.path.join(BASE_DIR, "PNN_CIL_v2", "Period_3", "ResNet18_PNN_best.pth")
prev_checkpoint = torch.load(prev_model_path, map_location=device)
frozen_step2.load_state_dict(prev_checkpoint["model_state_dict"], strict=True)
frozen_step2.to(device)
frozen_step2.eval()
for p in frozen_step2.parameters():
    p.requires_grad = False

new_column_4 = ECG_PNN_Column(input_dim=1024, output_dim=new_output_size, hidden_dim=512, dropout=0.2)
model = ECG_ProgressiveNN(base_model=frozen_step2, new_column=new_column_4).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.new_column.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=10)

train_with_pnn_ecg(
    model=model,
    output_size=output_size,
    criterion=criterion,
    optimizer=optimizer,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    scheduler=scheduler,
    num_epochs=200,
    batch_size=64,
    model_saving_folder=model_saving_folder,
    model_name="ResNet18_PNN",
    stop_signal_file=stop_signal_file,
    period=period,
    device=device
)

del X_train, y_train, X_val, y_val
del base_model, prev_column_2, prev_column_3, new_column_4
del frozen_step1, frozen_step2, model
gc.collect()
torch.cuda.empty_cache()
