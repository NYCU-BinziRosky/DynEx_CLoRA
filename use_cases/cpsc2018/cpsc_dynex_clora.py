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
    """LoRA adapter for Conv1d layer (adds low-rank delta weights)."""
    def __init__(self, conv_layer: nn.Conv1d, rank: int):
        super().__init__()
        self.conv = conv_layer
        self.rank = rank
        self.lora_A = nn.Parameter(torch.zeros(conv_layer.out_channels, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, conv_layer.in_channels * conv_layer.kernel_size[0]))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.01)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.conv(x)

    def get_delta(self):
        lora_weight = torch.matmul(self.lora_A, self.lora_B).view(
            self.conv.out_channels, self.conv.in_channels, self.conv.kernel_size[0]
        )
        return lora_weight

    def parameters(self, recurse=True):
        return [self.lora_A, self.lora_B]


class BasicBlock1d_LoRA(nn.Module):
    """1D Residual block with optional multiple LoRA adapters on conv2."""
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
        self.lora_adapters = nn.ModuleList()

    def add_lora_adapter(self):
        """Attach a new LoRA adapter to conv2."""
        new_lora = LoRAConv1d(self.conv2, self.lora_rank).to(next(self.parameters()).device)
        self.lora_adapters.append(new_lora)
        return new_lora

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))

        if len(self.lora_adapters) > 0:
            base_out = self.conv2(out)
            lora_weight_delta = sum(adapter.get_delta() for adapter in self.lora_adapters)
            adapted_weight = self.conv2.weight + lora_weight_delta
            out = F.conv1d(out, adapted_weight, bias=self.conv2.bias,
                           stride=self.conv2.stride, padding=self.conv2.padding,
                           dilation=self.conv2.dilation, groups=self.conv2.groups)
        else:
            out = self.conv2(out)

        out = self.bn2(out)
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet18_1D_LoRA(nn.Module):
    """
    ResNet18 1D variant for ECG classification with multiple LoRA adapters per BasicBlock.
    Supports dynamic addition of adapters for continual learning.
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

        x1 = self.adaptiveavgpool(x)
        x2 = self.adaptivemaxpool(x)
        x = torch.cat((x1, x2), dim=1).view(x.size(0), -1)

        x = self.dropout(x)
        return self.fc(x)

    def init_weights(self):
        """Kaiming initialization for Conv1d, Linear, and BatchNorm layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def add_lora_adapter(self):
        """Attach a LoRA adapter to each BasicBlock in the network."""
        adapters = []
        for module in self.modules():
            if isinstance(module, BasicBlock1d_LoRA):
                adapters.append(module.add_lora_adapter())
        return adapters

    def count_lora_adapters(self):
        """Return total number of LoRA adapters and blocks that contain them."""
        total, blocks_with_adapters = 0, 0
        for module in self.modules():
            if isinstance(module, BasicBlock1d_LoRA) and module.lora_adapters:
                blocks_with_adapters += 1
                total += len(module.lora_adapters)
        return total

    def count_lora_groups(self):
        """Return number of LoRA adapter groups (per-block adapter count)."""
        blocks = [m for m in self.modules() if isinstance(m, BasicBlock1d_LoRA)]
        return len(blocks[0].lora_adapters) if blocks else 0



def extract_features(model, x):
    """Helper function to extract features from the model for similarity calculation"""
    # This is a placeholder - you'll need to adapt this based on your actual model architecture
    # The goal is to extract meaningful features before the classification layer
    # For ResNet18_1D_LoRA model, this would typically be the features right before the fc layer
    
    # Example (pseudo-code - adapt to your actual model):
    x = x.permute(0, 2, 1)  # Convert to (batch_size, channels, time_steps)
    
    # Feed through the network up to the point before classification
    with torch.no_grad():
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        
        # Apply pooling
        x1 = model.adaptiveavgpool(x)
        x2 = model.adaptivemaxpool(x)
        
        # Concatenate pooling results
        x = torch.cat((x1, x2), dim=1)
        
        # Flatten
        features = x.view(x.size(0), -1)
    
    return features

# Training Function
def train_with_dynex_clora_ecg(model, teacher_model, output_size, criterion, optimizer,
                               X_train, y_train, X_val, y_val,
                               num_epochs, batch_size, alpha,
                               model_saving_folder, model_name,
                               stop_signal_file=None, scheduler=None,
                               period=None, stable_classes=None,
                               similarity_threshold=0.0,
                               class_features_dict=None, related_labels=None, device=None):
    """
    DynEx-CLoRA training loop for ECG task with adapter expansion and similarity comparison.
    """
    device = device or auto_select_cuda_device()
    model.to(device)
    if teacher_model:
        teacher_model.to(device)
        teacher_model.eval()

    model_name = model_name or "dynex_model"
    os.makedirs(model_saving_folder, exist_ok=True)
    best_model_path = os.path.join(model_saving_folder, f"{model_name}_best.pth")

    # === Prepare Data ===
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    # === Feature Extraction (train set) ===
    model.eval()
    new_class_features = {}
    with torch.no_grad():
        for xb, yb in train_loader:
            features = extract_features(model, xb)
            for cls in torch.unique(yb):
                cls_feat = features[yb == cls]
                new_class_features.setdefault(cls.item(), []).append(cls_feat)
    for cls in new_class_features:
        new_class_features[cls] = torch.cat(new_class_features[cls], dim=0).mean(dim=0)

    # === LoRA Adapter Assignment ===
    related_labels = related_labels or {}
    to_unfreeze = set()
    new_lora_indices = []
    cosine_sim = torch.nn.CosineSimilarity(dim=0)

    if period == 1:
        related_labels["base"] = list(new_class_features.keys())
    elif class_features_dict:
        existing_classes = set(class_features_dict.keys())
        current_classes = set(new_class_features.keys())
        new_classes = current_classes - existing_classes

        for new_cls in new_classes:
            new_feat = new_class_features[new_cls]
            sims = sorted(
                [(old_cls, cosine_sim(new_feat.to(device), class_features_dict[old_cls].to(device)).item())
                 for old_cls in class_features_dict],
                key=lambda x: x[1], reverse=True
            )
            matched = False
            for old_cls, sim in sims:
                if sim >= similarity_threshold:
                    for group_idx, class_list in related_labels.items():
                        if old_cls in class_list:
                            related_labels[group_idx].append(new_cls)
                            to_unfreeze.add(group_idx)
                            matched = True
                            break
                if matched:
                    break
            if not matched:
                model.add_lora_adapter()
                group_idx = max([g for g in related_labels.keys() if isinstance(g, int)], default=-1) + 1
                related_labels[group_idx] = [new_cls]
                new_lora_indices.append(group_idx)

        for old_cls in existing_classes & current_classes:
            sim_self = cosine_sim(new_class_features[old_cls].to(device), class_features_dict[old_cls].to(device)).item()
            if sim_self < similarity_threshold:
                for group_idx, class_list in related_labels.items():
                    if old_cls in class_list:
                        to_unfreeze.add(group_idx)

        # === Parameter Freezing/Unfreezing ===
        for name, param in model.named_parameters():
            param.requires_grad = False
        for group_idx in to_unfreeze | set(new_lora_indices):
            for module in model.modules():
                if isinstance(module, BasicBlock1d_LoRA):
                    if group_idx == "base":
                        for p in module.conv2.parameters():
                            p.requires_grad = True
                    elif group_idx < len(module.lora_adapters):
                        for p in module.lora_adapters[group_idx].parameters():
                            p.requires_grad = True
        for p in model.fc.parameters():
            p.requires_grad = True

    # === Optimizer Reset ===
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=optimizer.param_groups[0]["lr"],
        weight_decay=optimizer.param_groups[0].get("weight_decay", 0)
    )

    # === Training Loop ===
    best_val_acc = 0.0
    best_record = None

    for epoch in range(num_epochs):
        if stop_signal_file and os.path.exists(stop_signal_file):
            break

        model.train()
        total_loss = 0.0
        class_correct, class_total = {}, {}

        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = model(xb)
            ce_loss = criterion(outputs, yb)

            if teacher_model and stable_classes:
                with torch.no_grad():
                    teacher_outputs = teacher_model(xb)
                distill_loss = F.mse_loss(outputs[:, stable_classes], teacher_outputs[:, stable_classes])
                loss = alpha * distill_loss + (1 - alpha) * ce_loss
            else:
                loss = ce_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            compute_classwise_accuracy(outputs, yb, class_correct, class_total)

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = {
            int(c): f"{(class_correct[c] / class_total[c]) * 100:.2f}%" if class_total[c] > 0 else "0.00%"
            for c in class_total
        }

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_class_correct, val_class_total = {}, {}
        with torch.no_grad():
            for xb, yb in val_loader:
                outputs = model(xb)
                val_loss += criterion(outputs, yb).item() * xb.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)
                compute_classwise_accuracy(outputs, yb, val_class_correct, val_class_total)

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_acc_cls = {
            int(c): f"{(val_class_correct[c] / val_class_total[c]) * 100:.2f}%" if val_class_total[c] > 0 else "0.00%"
            for c in val_class_total
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
            "num_lora_groups": model.count_lora_groups(),
            "related_labels": related_labels
        }

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_record = current
            torch.save(current, best_model_path)

        if scheduler:
            scheduler.step(val_loss)

    # === Save class features ===
    if class_features_dict is not None:
        class_features_dict.update(new_class_features)
        with open(os.path.join(model_saving_folder, "class_features.pkl"), "wb") as f:
            pickle.dump(class_features_dict, f)

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
class_features_path = "path/to/your/period1_class_features.pkl"
ensure_folder(model_saving_folder)

# ==== Load Data & Features ====
X_train = np.load(os.path.join(save_dir, f"X_train_p{period}.npy"))
y_train = np.load(os.path.join(save_dir, f"y_train_p{period}.npy"))
X_val = np.load(os.path.join(save_dir, f"X_test_p{period}.npy"))
y_val = np.load(os.path.join(save_dir, f"y_test_p{period}.npy"))
with open(class_features_path, "rb") as f:
    class_features_dict = pickle.load(f)

# ==== Model Setup ====
device = auto_select_cuda_device()
input_channels = X_train.shape[2]
output_size = len(np.unique(y_train))

teacher_model = None
model = ResNet18_1D_LoRA(input_channels=input_channels, output_size=output_size).to(device)

checkpoint = torch.load(prev_model_path, map_location=device)
num_lora_groups = checkpoint.get("num_lora_groups", 0)
related_labels = checkpoint.get("related_labels", {"base": [0, 1]})
for _ in range(num_lora_groups):
    model.add_lora_adapter()

model_dict = model.state_dict()
prev_state_dict = checkpoint["model_state_dict"]
filtered_dict = {
    k: v for k, v in prev_state_dict.items()
    if k in model_dict and model_dict[k].shape == v.shape and not k.startswith("fc")
}
model.load_state_dict(filtered_dict, strict=False)

# ==== Training ====
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.9, patience=10)

train_with_dynex_clora_ecg(
    model=model,
    teacher_model=teacher_model,
    output_size=output_size,
    criterion=criterion,
    optimizer=optimizer,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    num_epochs=200,
    batch_size=64,
    alpha=0.0,
    model_saving_folder=model_saving_folder,
    model_name="dynex_period2",
    stop_signal_file=stop_signal_file,
    scheduler=scheduler,
    period=period,
    stable_classes=[0],
    similarity_threshold=0.99,
    class_features_dict=class_features_dict,
    related_labels=related_labels,
    device=device
)


# Period 3
period = 3

# ==== Paths ====
stop_signal_file = "path/to/your/stop_signal_file.txt"
model_saving_folder = "path/to/your/model_saving_folder"
prev_model_path = "path/to/your/period2_best_model.pth"
class_features_path = "path/to/your/period2_class_features.pkl"
ensure_folder(model_saving_folder)

# ==== Load Data & Features ====
X_train = np.load(os.path.join(save_dir, f"X_train_p{period}.npy"))
y_train = np.load(os.path.join(save_dir, f"y_train_p{period}.npy"))
X_val = np.load(os.path.join(save_dir, f"X_test_p{period}.npy"))
y_val = np.load(os.path.join(save_dir, f"y_test_p{period}.npy"))
with open(class_features_path, "rb") as f:
    class_features_dict = pickle.load(f)

# ==== Model Setup ====
device = auto_select_cuda_device()
input_channels = X_train.shape[2]
output_size = len(np.unique(y_train))

teacher_model = None
model = ResNet18_1D_LoRA(input_channels=input_channels, output_size=output_size).to(device)

checkpoint = torch.load(prev_model_path, map_location=device)
num_lora_groups = checkpoint.get("num_lora_groups", 0)
related_labels = checkpoint.get("related_labels", {"base": [0, 1]})
for _ in range(num_lora_groups):
    model.add_lora_adapter()

model_dict = model.state_dict()
prev_state_dict = checkpoint["model_state_dict"]
filtered_dict = {
    k: v for k, v in prev_state_dict.items()
    if k in model_dict and model_dict[k].shape == v.shape and not k.startswith("fc")
}
model.load_state_dict(filtered_dict, strict=False)

# ==== Training ====
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.9, patience=10)

train_with_dynex_clora_ecg(
    model=model,
    teacher_model=teacher_model,
    output_size=output_size,
    criterion=criterion,
    optimizer=optimizer,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    num_epochs=200,
    batch_size=64,
    alpha=0.0,
    model_saving_folder=model_saving_folder,
    model_name="dynex_period3",
    stop_signal_file=stop_signal_file,
    scheduler=scheduler,
    period=period,
    stable_classes=[0, 2, 3],
    similarity_threshold=0.85,
    class_features_dict=class_features_dict,
    related_labels=related_labels,
    device=device
)


# Period 4
period = 4

# ==== Paths ====
stop_signal_file = "path/to/your/stop_signal_file.txt"
model_saving_folder = "path/to/your/model_saving_folder"
prev_model_path = "path/to/your/period3_best_model.pth"
class_features_path = "path/to/your/period3_class_features.pkl"
ensure_folder(model_saving_folder)

# ==== Load Data & Features ====
X_train = np.load(os.path.join(save_dir, f"X_train_p{period}.npy"))
y_train = np.load(os.path.join(save_dir, f"y_train_p{period}.npy"))
X_val = np.load(os.path.join(save_dir, f"X_test_p{period}.npy"))
y_val = np.load(os.path.join(save_dir, f"y_test_p{period}.npy"))
with open(class_features_path, "rb") as f:
    class_features_dict = pickle.load(f)

# ==== Model Setup ====
device = auto_select_cuda_device()
input_channels = X_train.shape[2]
output_size = int(np.max(y_train)) + 1

teacher_model = None
model = ResNet18_1D_LoRA(input_channels=input_channels, output_size=output_size).to(device)

checkpoint = torch.load(prev_model_path, map_location=device)
num_lora_groups = checkpoint.get("num_lora_groups", 0)
related_labels = checkpoint.get("related_labels", {"base": [0, 1]})
for _ in range(num_lora_groups):
    model.add_lora_adapter()

model_dict = model.state_dict()
prev_state_dict = checkpoint["model_state_dict"]
filtered_dict = {
    k: v for k, v in prev_state_dict.items()
    if k in model_dict and model_dict[k].shape == v.shape and not k.startswith("fc")
}
model.load_state_dict(filtered_dict, strict=False)

# ==== Training ====
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.9, patience=10)

train_with_dynex_clora_ecg(
    model=model,
    teacher_model=teacher_model,
    output_size=output_size,
    criterion=criterion,
    optimizer=optimizer,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    num_epochs=200,
    batch_size=64,
    alpha=0.0,
    model_saving_folder=model_saving_folder,
    model_name="dynex_period4",
    stop_signal_file=stop_signal_file,
    scheduler=scheduler,
    period=period,
    stable_classes=[0, 2, 3, 4, 5],
    similarity_threshold=0.65,
    class_features_dict=class_features_dict,
    related_labels=related_labels,
    device=device
)
