# ================================
# 📦 Imports
# ================================

# OS & file management
import os
import shutil
import subprocess
import time
import gc
import re
import copy
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

# ================================
# 📁 Paths & Constants (user-defined)
# ================================

# Update this to your project root
BASE_DIR = "./CPSC_CIL"
save_dir = os.path.join(BASE_DIR, "processed")  # Path to the preprocessed `.npy` files (one for each continual learning period).
ECG_PATH = os.path.join(BASE_DIR, "datas")      # Directory containing original `.mat` and `.hea` files.
MAX_LEN = 5000  # ECG signal sequence length

# ================================
# 🔤 SNOMED code mapping
# ================================
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

# ================================
# 📊 Class Distribution Utility
# ================================
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

# ================================
# 📂 Folder Utility
# ================================
def ensure_folder(folder_path: str) -> None:
    """Create a folder if it does not exist."""
    os.makedirs(folder_path, exist_ok=True)

# ================================
# ⚡ GPU Device Selector
# ================================
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

# ================================
# 📈 Class-wise Accuracy Utility
# ================================
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

# ================================
# 📊 Model Parameter Info
# ================================
def get_model_parameter_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    param_size_MB = total_params * 4 / (1024**2)
    return total_params, param_size_MB

# ================================
# ⚖️ Class Weights Calculator
# ================================
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

# ================================
# 🔁 Forward Transfer Evaluation
# ================================
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

# ================================
# 🧠 ResNet18_1D for ECG Input
# ================================
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
    ResNet18 1D variant for ECG signal classification.
    Uses both average and max pooling for richer representation.
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
        self.fc = nn.Linear(512 * 2, output_size)

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

# ================================
# 📌 Elastic Weight Consolidation (EWC)
# ================================
class EWC:
    """
    Elastic Weight Consolidation (EWC) for continual learning.
    Penalizes changes to important parameters (estimated by Fisher Information Matrix).
    """
    def __init__(self, fisher: dict, params: dict):
        self.fisher = {k: v.cpu() for k, v in fisher.items()}
        self.params = {k: v.cpu() for k, v in params.items()}

    @staticmethod
    def compute_fisher_and_params(model, dataloader, criterion, device, sample_size=None):
        model.train()
        fisher = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}
        params = {n: p.clone().detach().cpu() for n, p in model.named_parameters() if p.requires_grad}

        total_samples = 0
        for x, y in dataloader:
            if sample_size and total_samples >= sample_size:
                break
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += (p.grad ** 2) * x.size(0)
            total_samples += x.size(0)

        fisher = {n: f / total_samples for n, f in fisher.items()}
        return {n: f.cpu() for n, f in fisher.items()}, params

    def penalty(self, model):
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher:
                penalty = self.fisher[n].to(p.device) * (p - self.params[n].to(p.device)) ** 2
                loss += penalty.sum()
        return loss

# ================================
# 🔧 Training Function for EWC on ECG Dataset
# ================================
def train_with_ewc_ecg(
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
    ewc=None,
    lambda_ewc=0.4,
    device=None
):
    print("\n🚀 Starting ECG training with EWC...")
    start_time = time.time()

    device = device or auto_select_cuda_device()
    model_name = model_name or "model"
    model_saving_folder = model_saving_folder or "./saved_models"

    if os.path.exists(model_saving_folder):
        shutil.rmtree(model_saving_folder)
        print(f"✅ Removed existing folder: {model_saving_folder}")
    os.makedirs(model_saving_folder, exist_ok=True)

    model.to(device)

    # Prepare data
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    print("\n✅ Dataset Summary:")
    print(f"  - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  - X_val  : {X_val.shape}, y_val  : {y_val.shape}")

    best_results = []

    for epoch in range(num_epochs):
        if stop_signal_file and os.path.exists(stop_signal_file):
            print("\n🛑 Stop signal detected. Exiting training loop.")
            break

        model.train()
        epoch_loss = 0.0
        class_correct, class_total = {}, {}

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            if ewc:
                loss += (lambda_ewc / 2) * ewc.penalty(model)
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

        print(f"\n🌀 Epoch {epoch + 1}/{num_epochs}")
        print(f"  - Train Loss     : {train_loss:.6f}")
        print(f"  - Train Acc (per class): {train_acc}")
        print(f"  - Val Loss       : {val_loss:.6f}")
        print(f"  - Val Accuracy   : {val_acc * 100:.2f}%")
        print(f"  - Val Acc (per class): {val_acc_cls}")
        print(f"  - LR             : {optimizer.param_groups[0]['lr']:.6f}")

        model_path = os.path.join(model_saving_folder, f"{model_name}_epoch_{epoch+1}.pth")
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
            "model_path": model_path,
        }

        if len(best_results) < 5 or val_acc > best_results[-1]["val_accuracy"]:
            if len(best_results) == 5:
                to_remove = best_results.pop()
                if os.path.exists(to_remove["model_path"]):
                    os.remove(to_remove["model_path"])
                    print(f"🗑 Removed: {to_remove['model_path']}")
            best_results.append(current)
            best_results.sort(key=lambda x: (x["val_accuracy"], x["epoch"]), reverse=True)
            torch.save(current, model_path)
            print(f"✅ Saved model: {model_path}")

        if scheduler:
            scheduler.step(val_loss)

    # === Final Output ===
    training_time = time.time() - start_time
    total_params, param_size_MB = get_model_parameter_info(model)

    if best_results:
        best = best_results[0]
        best_model_path = os.path.join(model_saving_folder, f"{model_name}_best.pth")
        torch.save(best, best_model_path)
        print(f"\n🏆 Best model saved as: {best_model_path} (Val Acc: {best['val_accuracy'] * 100:.2f}%)")

    final_model_path = os.path.join(model_saving_folder, f"{model_name}_final.pth")
    torch.save(current, final_model_path)
    print(f"\n📌 Final model saved as: {final_model_path}")

    print("\n🎯 Top 5 Models:")
    for res in best_results:
        print(f"Epoch {res['epoch']} | Val Acc: {res['val_accuracy']*100:.2f}% | Model Path: {res['model_path']}")

    print(f"\n🧠 Model Summary:")
    print(f"  - Total Parameters: {total_params:,}")
    print(f"  - Model Size      : {param_size_MB:.2f} MB")
    print(f"  - Training Time   : {training_time:.2f} seconds")

    # Markdown-like formatted summary (for logging)
    match = re.search(r'Period_(\d+)', model_saving_folder)
    period_label = match.group(1) if match else "?"
    model_name_str = model.__class__.__name__

    print("\n---")
    print(f"### Period {period_label}")
    print(f"+ Training time : {training_time:.2f} seconds")
    print(f"+ Model         : {model_name_str}")
    print(f"+ Best Epoch    : {best['epoch']}")
    print(f"+ Val Accuracy  : {best['val_accuracy'] * 100:.2f}%")
    print(f"+ Classwise Acc : {best['val_classwise_accuracy']}")
    print(f"+ Parameters    : {total_params:,}")
    print(f"+ Size (float32): {param_size_MB:.2f} MB")

    # Memory cleanup
    del X_train, y_train, X_val, y_val, train_loader, val_loader, current
    torch.cuda.empty_cache()
    gc.collect()


# ==========================================
# 🚨 Note:
# Period 1 model is trained independently and shared across all methods.
# Please ensure it is saved beforehand and correctly referenced here.
# ==========================================


# ================================
# 📌 Period 2: EWC Training (Protect Period 1)
# ================================
period = 2

# ==== Paths ====
stop_signal_file = os.path.join(BASE_DIR, "stop_training.txt")
model_saving_folder = os.path.join(BASE_DIR, "EWC_CIL", f"Period_{period}")
ensure_folder(model_saving_folder)

# ==== Load Data ====
X_train = np.load(os.path.join(save_dir, f"X_train_p{period}.npy"))
y_train = np.load(os.path.join(save_dir, f"y_train_p{period}.npy"))
X_val   = np.load(os.path.join(save_dir, f"X_test_p{period}.npy"))
y_val   = np.load(os.path.join(save_dir, f"y_test_p{period}.npy"))

# ==== Device ====
device = auto_select_cuda_device()

# ==== Model Setup ====
input_channels = X_train.shape[2]
output_size = len(np.unique(y_train))
model = ResNet18_1D(input_channels=input_channels, output_size=output_size).to(device)

# ==== Load Pretrained Period 1 Model ====
prev_model_path = "path/to/your/period1_best_model.pth"
prev_checkpoint = torch.load(prev_model_path, map_location=device)
state_dict = prev_checkpoint["model_state_dict"]
model_dict = model.state_dict()
filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
model.load_state_dict(filtered_dict, strict=False)

# ==== Prepare EWC State (from Period 1) ====
X_prev = np.load(os.path.join(save_dir, "X_train_p1.npy"))
y_prev = np.load(os.path.join(save_dir, "y_train_p1.npy"))
train_loader_prev = DataLoader(
    TensorDataset(torch.tensor(X_prev, dtype=torch.float32), torch.tensor(y_prev, dtype=torch.long)),
    batch_size=64, shuffle=True
)

criterion = nn.CrossEntropyLoss()
fisher_dict, params_dict = EWC.compute_fisher_and_params(model, train_loader_prev, criterion, device=device)
ewc_state = EWC(fisher=fisher_dict, params=params_dict)

# ==== Training ====
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10)

train_with_ewc_ecg(
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
    model_name="ResNet18_1D",
    stop_signal_file=stop_signal_file,
    ewc=ewc_state,
    lambda_ewc=1.0,
    device=device
)

# ==== Cleanup ====
del model, train_loader_prev, fisher_dict, params_dict
gc.collect()
torch.cuda.empty_cache()


# ================================
# 📌 Period 3: EWC Training (Protect Period 2)
# ================================
period = 3

stop_signal_file = os.path.join(BASE_DIR, "stop_training.txt")
model_saving_folder = os.path.join(BASE_DIR, "EWC_CIL", f"Period_{period}")
ensure_folder(model_saving_folder)

X_train = np.load(os.path.join(save_dir, f"X_train_p{period}.npy"))
y_train = np.load(os.path.join(save_dir, f"y_train_p{period}.npy"))
X_val   = np.load(os.path.join(save_dir, f"X_test_p{period}.npy"))
y_val   = np.load(os.path.join(save_dir, f"y_test_p{period}.npy"))

device = auto_select_cuda_device()
input_channels = X_train.shape[2]
output_size = len(np.unique(y_train))
model = ResNet18_1D(input_channels=input_channels, output_size=output_size).to(device)

prev_model_path = os.path.join(BASE_DIR, "EWC_CIL", "Period_2", "ResNet18_1D_best.pth")
prev_checkpoint = torch.load(prev_model_path, map_location=device)
state_dict = prev_checkpoint["model_state_dict"]
model_dict = model.state_dict()
filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
model.load_state_dict(filtered_dict, strict=False)

X_prev = np.load(os.path.join(save_dir, "X_train_p2.npy"))
y_prev = np.load(os.path.join(save_dir, "y_train_p2.npy"))
train_loader_prev = DataLoader(
    TensorDataset(torch.tensor(X_prev, dtype=torch.float32), torch.tensor(y_prev, dtype=torch.long)),
    batch_size=64, shuffle=True
)

criterion = nn.CrossEntropyLoss()
fisher_dict, params_dict = EWC.compute_fisher_and_params(model, train_loader_prev, criterion, device=device)
ewc_state = EWC(fisher=fisher_dict, params=params_dict)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10)

train_with_ewc_ecg(
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
    model_name="ResNet18_1D",
    stop_signal_file=stop_signal_file,
    ewc=ewc_state,
    lambda_ewc=1.0,
    device=device
)

del model, train_loader_prev, fisher_dict, params_dict
gc.collect()
torch.cuda.empty_cache()


# ================================
# 📌 Period 4: EWC Training (Protect Period 3)
# ================================
period = 4

stop_signal_file = os.path.join(BASE_DIR, "stop_training.txt")
model_saving_folder = os.path.join(BASE_DIR, "EWC_CIL", f"Period_{period}")
ensure_folder(model_saving_folder)

X_train = np.load(os.path.join(save_dir, f"X_train_p{period}.npy"))
y_train = np.load(os.path.join(save_dir, f"y_train_p{period}.npy"))
X_val   = np.load(os.path.join(save_dir, f"X_test_p{period}.npy"))
y_val   = np.load(os.path.join(save_dir, f"y_test_p{period}.npy"))

device = auto_select_cuda_device()
input_channels = X_train.shape[2]
output_size = int(np.max(y_train)) + 1  # Avoid class indexing issues
model = ResNet18_1D(input_channels=input_channels, output_size=output_size).to(device)

prev_model_path = os.path.join(BASE_DIR, "EWC_CIL", "Period_3", "ResNet18_1D_best.pth")
prev_checkpoint = torch.load(prev_model_path, map_location=device)
state_dict = prev_checkpoint["model_state_dict"]
model_dict = model.state_dict()
filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
model.load_state_dict(filtered_dict, strict=False)

X_prev = np.load(os.path.join(save_dir, "X_train_p3.npy"))
y_prev = np.load(os.path.join(save_dir, "y_train_p3.npy"))
train_loader_prev = DataLoader(
    TensorDataset(torch.tensor(X_prev, dtype=torch.float32), torch.tensor(y_prev, dtype=torch.long)),
    batch_size=64, shuffle=True
)

criterion = nn.CrossEntropyLoss()
fisher_dict, params_dict = EWC.compute_fisher_and_params(model, train_loader_prev, criterion, device=device)
ewc_state = EWC(fisher=fisher_dict, params=params_dict)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10)

train_with_ewc_ecg(
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
    model_name="ResNet18_1D",
    stop_signal_file=stop_signal_file,
    ewc=ewc_state,
    lambda_ewc=1.0,
    device=device
)

del model, train_loader_prev, fisher_dict, params_dict
gc.collect()
torch.cuda.empty_cache()
