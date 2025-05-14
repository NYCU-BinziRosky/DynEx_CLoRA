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
class LoRAConv1d(nn.Module):
    """LoRA module for adapting 1D convolutional layers."""
    def __init__(self, conv_layer: nn.Conv1d, rank: int):
        super().__init__()
        self.conv = conv_layer
        self.rank = rank
        self.lora_A = nn.Parameter(torch.zeros(conv_layer.out_channels, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, conv_layer.in_channels * conv_layer.kernel_size[0]))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.01)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        lora_weight = torch.matmul(self.lora_A, self.lora_B).view(
            self.conv.out_channels, self.conv.in_channels, self.conv.kernel_size[0]
        )
        adapted_weight = self.conv.weight + lora_weight
        return F.conv1d(
            x, adapted_weight, bias=self.conv.bias,
            stride=self.conv.stride, padding=self.conv.padding,
            dilation=self.conv.dilation, groups=self.conv.groups
        )

class BasicBlock1d_LoRA(nn.Module):
    """Residual block with optional LoRA adapter on conv2."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, lora_rank=None):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.lora_rank = lora_rank
        self.lora_adapter = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.lora_adapter(out) if self.lora_adapter else self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)

class ResNet18_1D_LoRA(nn.Module):
    """
    ResNet18 1D variant with LoRA support for ECG classification.
    Uses both average and max pooling for richer representation.
    """
    def __init__(self, input_channels=12, output_size=9, inplanes=64, lora_rank=4):
        super().__init__()
        self.inplanes = inplanes
        self.lora_rank = lora_rank

        self.conv1 = nn.Conv1d(input_channels, inplanes, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock1d_LoRA, 64, 2)
        self.layer2 = self._make_layer(BasicBlock1d_LoRA, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock1d_LoRA, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock1d_LoRA, 512, 2, stride=2)

        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512 * 2, output_size)

        self.init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )
        layers = [block(self.inplanes, planes, stride, downsample, self.lora_rank)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, lora_rank=self.lora_rank))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, C, T)
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
        return self.fc(x)

    def init_weights(self):
        """Kaiming initialization for Conv and Linear layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def init_lora(self):
        """Attach LoRA adapters to conv2 layers in BasicBlock1d_LoRA."""
        for module in self.modules():
            if isinstance(module, BasicBlock1d_LoRA) and module.lora_adapter is None:
                module.lora_adapter = LoRAConv1d(module.conv2, self.lora_rank).to(next(self.parameters()).device)

    def get_trainable_parameters(self):
        """Return LoRA + FC trainable parameters only."""
        lora_params = []
        for module in self.modules():
            if isinstance(module, LoRAConv1d):
                lora_params.extend([module.lora_A, module.lora_B])
        fc_params = list(self.fc.parameters())
        return lora_params + fc_params


# Training Function
def train_with_lora_ecg(
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
    ECG training loop using standard LoRA finetuning.

    Arguments:
        model (nn.Module): LoRA-based model.
        output_size (int): Number of output classes.
        criterion (Loss): Loss function (e.g., CrossEntropy).
        optimizer (Optimizer): Optimizer instance.
        X_train, y_train: Training data (numpy arrays).
        X_val, y_val: Validation data (numpy arrays).
        scheduler (optional): Learning rate scheduler.
        num_epochs (int): Training epoch count.
        batch_size (int): Mini-batch size.
        model_saving_folder (str, optional): Output folder for best model.
        model_name (str, optional): File prefix name for saved model.
        stop_signal_file (str, optional): External kill-switch path.
        device (torch.device, optional): Device to train on.
    """
    device = device or auto_select_cuda_device()
    model_name = model_name or "lora_model"
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
# You may reuse the one trained in `cpsc_ewc.py`,
# or train it using the current method's training function
# (as long as the model architecture is consistent).
# ==========================================


# Period 2
period = 2

# ==== Paths ====
stop_signal_file = "path/to/your/stop_signal_file.txt"
model_saving_folder = "path/to/your/model_saving_folder"
prev_model_path = "path/to/your/period1_best_model.pth"
ensure_folder(model_saving_folder)

# ==== Load Data ====
X_train = np.load(os.path.join(save_dir, f"X_train_p{period}.npy"))
y_train = np.load(os.path.join(save_dir, f"y_train_p{period}.npy"))
X_val   = np.load(os.path.join(save_dir, f"X_test_p{period}.npy"))
y_val   = np.load(os.path.join(save_dir, f"y_test_p{period}.npy"))

# ==== Device & Model ====
device = auto_select_cuda_device()
input_channels = X_train.shape[2]
output_size = len(np.unique(y_train))
model = ResNet18_1D_LoRA(input_channels=input_channels, output_size=output_size, lora_rank=4).to(device)

# ==== Load Previous Weights (excluding FC) ====
checkpoint = torch.load(prev_model_path, map_location=device)
model_dict = model.state_dict()
filtered = {
    k: v for k, v in checkpoint["model_state_dict"].items()
    if k in model_dict and model_dict[k].shape == v.shape and not k.startswith("fc")
}
model.load_state_dict(filtered, strict=False)
model.init_lora()

# ==== Optimizer & Training ====
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.9, patience=10)

train_with_lora_ecg(
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
    model_name="lora_period2",
    stop_signal_file=stop_signal_file,
    device=device
)


# Period 3
period = 3

# ==== Paths ====
stop_signal_file = "path/to/your/stop_signal_file.txt"
model_saving_folder = "path/to/your/model_saving_folder"
prev_model_path = "path/to/your/period2_best_model.pth"
ensure_folder(model_saving_folder)

# ==== Load Data ====
X_train = np.load(os.path.join(save_dir, f"X_train_p{period}.npy"))
y_train = np.load(os.path.join(save_dir, f"y_train_p{period}.npy"))
X_val   = np.load(os.path.join(save_dir, f"X_test_p{period}.npy"))
y_val   = np.load(os.path.join(save_dir, f"y_test_p{period}.npy"))

# ==== Device & Model ====
device = auto_select_cuda_device()
input_channels = X_train.shape[2]
output_size = len(np.unique(y_train))
model = ResNet18_1D_LoRA(input_channels=input_channels, output_size=output_size, lora_rank=4).to(device)
model.init_lora()

# ==== Load Previous Weights (excluding FC & LoRA) ====
checkpoint = torch.load(prev_model_path, map_location=device)
model_dict = model.state_dict()
filtered = {
    k: v for k, v in checkpoint["model_state_dict"].items()
    if k in model_dict and model_dict[k].shape == v.shape and not (
        k.startswith("fc") or "lora_adapter.lora_A" in k or "lora_adapter.lora_B" in k
    )
}
model.load_state_dict(filtered, strict=False)

# ==== Optimizer & Training ====
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.9, patience=10)

train_with_lora_ecg(
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
    model_name="lora_period3",
    stop_signal_file=stop_signal_file,
    device=device
)


# Period 4
period = 4

# ==== Paths ====
stop_signal_file = "path/to/your/stop_signal_file.txt"
model_saving_folder = "path/to/your/model_saving_folder"
prev_model_path = "path/to/your/period3_best_model.pth"
ensure_folder(model_saving_folder)

# ==== Load Data ====
X_train = np.load(os.path.join(save_dir, f"X_train_p{period}.npy"))
y_train = np.load(os.path.join(save_dir, f"y_train_p{period}.npy"))
X_val   = np.load(os.path.join(save_dir, f"X_test_p{period}.npy"))
y_val   = np.load(os.path.join(save_dir, f"y_test_p{period}.npy"))

# ==== Device & Model ====
device = auto_select_cuda_device()
input_channels = X_train.shape[2]
output_size = int(np.max(y_train)) + 1
model = ResNet18_1D_LoRA(input_channels=input_channels, output_size=output_size, lora_rank=4).to(device)
model.init_lora()

# ==== Load Previous Weights (excluding FC & LoRA) ====
checkpoint = torch.load(prev_model_path, map_location=device)
model_dict = model.state_dict()
filtered = {
    k: v for k, v in checkpoint["model_state_dict"].items()
    if k in model_dict and model_dict[k].shape == v.shape and not (
        k.startswith("fc") or "lora_adapter.lora_A" in k or "lora_adapter.lora_B" in k
    )
}
model.load_state_dict(filtered, strict=False)

# ==== Optimizer & Training ====
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.9, patience=10)

train_with_lora_ecg(
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
    model_name="lora_period4",
    stop_signal_file=stop_signal_file,
    device=device
)
