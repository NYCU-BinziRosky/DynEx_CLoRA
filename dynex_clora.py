"""
dynex_clora.py — Dynamic Expandable LoRA for Continual Learning

This module implements DynEx-CLoRA: a continual learning method that:
- Dynamically grows LoRA adapter groups based on class similarity
- Selectively reuses or unfreezes adapters for drifting classes
- Supports optional knowledge distillation on stable classes

The user must supply:
- A LoRA-compatible model with `add_lora_adapter()` and `count_lora_groups()`
- A feature extractor function for computing class prototypes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
import os
import gc
import pickle


def train_with_dynex_clora(model, dataloader, val_loader, optimizer, criterion,
                           extract_features_fn,
                           teacher_model=None, stable_classes=None,
                           alpha=0.1, period=None, similarity_threshold=0.85,
                           related_labels=None, class_features_dict=None,
                           scheduler=None, num_epochs=100, device='cuda',
                           model_saving_folder=None):
    """
    DynEx-CLoRA training loop with similarity-driven LoRA expansion.

    Arguments:
        model (nn.Module): Target model with LoRA support.
        dataloader (DataLoader): Training set.
        val_loader (DataLoader): Validation set.
        optimizer (Optimizer): Optimizer instance.
        criterion (Loss): CrossEntropyLoss or other classification loss.
        extract_features_fn (function): Custom function(model, x) → feature tensor.
        teacher_model (optional): Frozen teacher model for distillation.
        stable_classes (list[int], optional): Indices of old stable classes.
        alpha (float, optional): Distillation strength (0.0 ~ 1.0).
        period (int): Current period (used for first-time logic).
        similarity_threshold (float): Threshold τ for adapter reuse.
        related_labels (dict): Mapping from adapter group to class indices.
        class_features_dict (dict): Historical mean feature vectors per class.
        scheduler (optional): Learning rate scheduler.
        num_epochs (int): Total number of epochs.
        device (str): Computation device.
        model_saving_folder (str): Where to save best checkpoint.
    """
    model.to(device)
    best_val_acc = 0.0

    if model_saving_folder:
        os.makedirs(model_saving_folder, exist_ok=True)
        best_model_path = os.path.join(model_saving_folder, "best_model.pth")

    # === Feature Extraction ===
    model.eval()
    new_class_features = defaultdict(list)
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            features = extract_features_fn(model, x_batch)
            for c in torch.unique(y_batch):
                new_class_features[int(c.item())].append(features[y_batch == c])
    new_class_features = {k: torch.cat(v).mean(dim=0) for k, v in new_class_features.items()}

    related_labels = related_labels or {}
    to_unfreeze = set()
    cosine_sim = nn.CosineSimilarity(dim=0)

    # === Adapter Assignment ===
    if period == 1:
        related_labels["base"] = list(new_class_features.keys())
    else:
        existing_classes = set(class_features_dict.keys())
        current_classes = set(new_class_features.keys())
        new_classes = current_classes - existing_classes

        for c_new in new_classes:
            sims = [(c_old, cosine_sim(new_class_features[c_new], class_features_dict[c_old]).item())
                    for c_old in existing_classes]
            sims.sort(key=lambda x: x[1], reverse=True)
            matched = False
            for c_old, sim in sims:
                if sim >= similarity_threshold:
                    for idx, group in related_labels.items():
                        if c_old in group:
                            related_labels[idx].append(c_new)
                            to_unfreeze.add(idx)
                            matched = True
                            break
                if matched:
                    break
            if not matched:
                model.add_lora_adapter()
                new_idx = max([k for k in related_labels.keys() if isinstance(k, int)], default=-1) + 1
                related_labels[new_idx] = [c_new]

        # Existing class drift detection
        for c in existing_classes & current_classes:
            sim = cosine_sim(new_class_features[c], class_features_dict[c]).item()
            if sim < similarity_threshold:
                for idx, group in related_labels.items():
                    if c in group:
                        to_unfreeze.add(idx)

        # Freeze all then selectively unfreeze
        for p in model.parameters():
            p.requires_grad = False
        for idx in to_unfreeze:
            for module in model.modules():
                if hasattr(module, "lora_adapters") and idx < len(module.lora_adapters):
                    for p in module.lora_adapters[idx].parameters():
                        p.requires_grad = True
        for p in model.fc.parameters():
            p.requires_grad = True

    # === Training Loop ===
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            if teacher_model and stable_classes:
                with torch.no_grad():
                    teacher_logits = teacher_model(x_batch)
                student_stable = outputs[:, stable_classes]
                teacher_stable = teacher_logits[:, stable_classes]
                distill_loss = F.mse_loss(student_stable, teacher_stable)
                loss = alpha * distill_loss + (1 - alpha) * loss

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
                preds = torch.argmax(model(x_val), dim=1)
                val_correct += (preds == y_val).sum().item()
                val_total += y_val.size(0)

        val_acc = val_correct / val_total
        print(f"[Epoch {epoch+1:03d}/{num_epochs}] Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2%}")

        if model_saving_folder and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "related_labels": related_labels,
                "num_lora_groups": model.count_lora_groups()
            }, best_model_path)

        if scheduler:
            scheduler.step(val_acc)

    # === Update Feature Dictionary ===
    if class_features_dict is not None:
        class_features_dict.update(new_class_features)
        with open(os.path.join(model_saving_folder, "class_features.pkl"), "wb") as f:
            pickle.dump(class_features_dict, f)

    gc.collect()
    torch.cuda.empty_cache()



"""
Example: DynEx-CLoRA Training with Adapter Expansion
----------------------------------------------------

# === Period 1: Train from scratch (no adapters) ===
model = YourLoRAModel(...)
model.to(device)

# Train with standard optimizer and CrossEntropy loss
...

# === Period 1: Save class feature centroids ===
class_features_dict = compute_class_prototypes(model, dataloader)
with open(".../Period_1/class_features.pkl", "wb") as f:
    pickle.dump(class_features_dict, f)

# Save model state, related_labels = {"base": [...]}, num_lora_groups = 0
torch.save({
    "model_state_dict": model.state_dict(),
    "related_labels": {"base": [...]},
    "num_lora_groups": 0
}, ".../Period_1/best_model.pth")


# === Period 2: Load previous state ===
with open(".../Period_1/class_features.pkl", "rb") as f:
    class_features_dict = pickle.load(f)

checkpoint = torch.load(".../Period_1/best_model.pth", map_location=device)
related_labels = checkpoint["related_labels"]
num_lora_groups = checkpoint["num_lora_groups"]

# === Rebuild model and add prior adapters ===
model = YourLoRAModel(...)
model.to(device)
for _ in range(num_lora_groups):
    model.add_lora_adapter()

# Load base weights only (skip final classifier layer)
state_dict = checkpoint["model_state_dict"]
filtered = {
    k: v for k, v in state_dict.items()
    if k in model.state_dict() and model.state_dict()[k].shape == v.shape and
       not k.startswith("fc.")
}
model.load_state_dict(filtered, strict=False)

# === Define feature extractor and optimizer ===
def extract_features_fn(model, x):
    return model.extract_pre_fc_features(x)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

train_with_dynex_clora(
    model=model,
    dataloader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=nn.CrossEntropyLoss(),
    extract_features_fn=extract_features_fn,
    teacher_model=None,
    stable_classes=None,
    alpha=0.0,
    period=2,
    similarity_threshold=0.85,
    related_labels=related_labels,
    class_features_dict=class_features_dict,
    scheduler=scheduler,
    num_epochs=100,
    device="cuda",
    model_saving_folder=".../Period_2"
)
"""

"""
DynEx-CLoRA Usage Notes
------------------------

1. Adapter-Aware Model:
   - Your model must implement:
     • `add_lora_adapter()` — adds a new group of LoRA adapters
     • `count_lora_groups()` — returns the total number of LoRA groups
     • A method like `extract_pre_fc_features()` for extracting features before classification

2. Period 1:
   - Train from scratch (no adapters)
   - Save `class_features.pkl`: maps class ID → feature vector
   - Set `related_labels = {"base": [...]}`
   - Do not call `add_lora_adapter()` yet

3. Period 2 and Beyond:
   - Load model weights from previous period
   - Call `model.add_lora_adapter()` for each group used so far
   - Load weights: exclude final layer (`fc.`)
   - Provide `related_labels` and `class_features_dict` for adapter selection

4. Similarity-Based Expansion:
   - Each new class is compared with previous class prototypes
   - If similarity ≥ τ:
     • Link new class to that adapter group (reuse)
   - Else:
     • Add a new LoRA group and associate it with the new class

5. Existing Class Drift:
   - If a class already seen in the past has low similarity to its previous feature vector,
     its adapter group will be unfrozen for refinement

6. Distillation (Optional):
   - Set `teacher_model` and `stable_classes` if using MSE loss for retained knowledge
   - Adjust `alpha` to balance distillation and classification loss (e.g., 0.1 ~ 0.5)

7. Output & State Saving:
   - The following will be saved:
     • `model_state_dict`
     • `related_labels`
     • `num_lora_groups`
     • `class_features.pkl`

8. Feature Format:
   - Output of `extract_features_fn(model, x)` must be a 2D tensor of shape (batch_size, feature_dim)
"""
