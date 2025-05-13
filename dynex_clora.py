"""
dynex_clora.py — Dynamic Expandable LoRA for Continual Learning

This module implements DynEx-CLoRA: a dynamic low-rank adapter framework 
for class-incremental continual learning. It supports:

- Feature extraction and similarity computation
- Adapter group allocation based on class drift
- Selective freezing/unfreezing of adapter groups
- Knowledge distillation (optional)

This implementation is task-agnostic. Users must define:
- Feature extractor logic for their model
- When and how to load teacher/previous weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os, time, shutil, gc, pickle
from collections import defaultdict

# === Feature Extraction (to be customized for your model) ===
def extract_features(model, x):
    x = x.permute(0, 2, 1)
    with torch.no_grad():
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x1 = model.adaptiveavgpool(x)
        x2 = model.adaptivemaxpool(x)
        x = torch.cat((x1, x2), dim=1)
        return x.view(x.size(0), -1)

# === Training Function ===
def train_with_dynex_clora(model, teacher_model, output_size, criterion, optimizer,
                           X_train, y_train, X_val, y_val,
                           num_epochs, batch_size, alpha,
                           model_saving_folder, model_name,
                           stop_signal_file=None, scheduler=None,
                           period=None, stable_classes=None,
                           similarity_threshold=0.0,
                           class_features_dict=None, related_labels=None,
                           device=None):
    
    # === Setup ===
    start_time = time.time()
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = model_name or "dynex_clora_model"
    os.makedirs(model_saving_folder, exist_ok=True)

    model.to(device)
    if teacher_model:
        teacher_model.to(device)
        teacher_model.eval()

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    # === Extract new class features ===
    model.eval()
    new_class_features = defaultdict(list)
    with torch.no_grad():
        for xb, yb in train_loader:
            feats = extract_features(model, xb)
            for cls in torch.unique(yb):
                new_class_features[int(cls.item())].append(feats[yb == cls])

    for cls in new_class_features:
        new_class_features[cls] = torch.cat(new_class_features[cls]).mean(dim=0)

    # === Similarity-based LoRA management ===
    to_unfreeze = set()
    new_lora_indices = []
    related_labels = related_labels or {}
    cosine_sim = nn.CosineSimilarity(dim=0)

    if period == 1:
        related_labels['base'] = list(new_class_features.keys())
    else:
        existing_classes = set(class_features_dict.keys())
        current_classes = set(new_class_features.keys())
        new_classes = current_classes - existing_classes

        for new_cls in new_classes:
            sims = [(old_cls, cosine_sim(new_class_features[new_cls], class_features_dict[old_cls]).item())
                    for old_cls in existing_classes]
            sims.sort(key=lambda x: x[1], reverse=True)

            matched = False
            for old_cls, sim in sims:
                if sim >= similarity_threshold:
                    for idx, rel_cls in related_labels.items():
                        if old_cls in rel_cls:
                            related_labels[idx].append(new_cls)
                            to_unfreeze.add(idx)
                            matched = True
                            break
                if matched: break

            if not matched:
                model.add_lora_adapter()
                group_idx = max([k for k in related_labels.keys() if isinstance(k, int)], default=-1) + 1
                related_labels[group_idx] = [new_cls]
                new_lora_indices.append(group_idx)

        for old_cls in existing_classes & current_classes:
            sim_self = cosine_sim(new_class_features[old_cls], class_features_dict[old_cls]).item()
            if sim_self < similarity_threshold:
                for idx, rel_cls in related_labels.items():
                    if old_cls in rel_cls:
                        to_unfreeze.add(idx)

        # Freeze all, then selectively unfreeze
        for p in model.parameters():
            p.requires_grad = False
        for idx in to_unfreeze:
            for m in model.modules():
                if hasattr(m, "lora_adapters") and idx < len(m.lora_adapters):
                    for p in m.lora_adapters[idx].parameters():
                        p.requires_grad = True
        for p in model.fc.parameters():
            p.requires_grad = True

    # === Training Loop ===
    best_model = None
    best_acc = 0.0
    for epoch in range(num_epochs):
        if stop_signal_file and os.path.exists(stop_signal_file):
            break
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)

            if teacher_model and stable_classes:
                with torch.no_grad():
                    t_logits = teacher_model(xb)
                student_logits = out[:, stable_classes]
                teacher_logits = t_logits[:, stable_classes]
                distill_loss = F.mse_loss(student_logits, teacher_logits)
                loss = alpha * distill_loss + (1 - alpha) * loss

            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = {
                'epoch': epoch + 1,
                'val_accuracy': best_acc,
                'model_state_dict': model.state_dict(),
                'related_labels': related_labels,
                'num_lora_groups': model.count_lora_groups()
            }
            torch.save(best_model, os.path.join(model_saving_folder, f"{model_name}_best.pth"))

        if scheduler:
            scheduler.step(val_acc)

    # Save final model and features
    torch.save(best_model, os.path.join(model_saving_folder, f"{model_name}_final.pth"))
    if class_features_dict is not None:
        class_features_dict.update(new_class_features)
        with open(os.path.join(model_saving_folder, "class_features.pkl"), 'wb') as f:
            pickle.dump(class_features_dict, f)

    print(f"\n✅ Period {period} completed. Best Val Acc: {best_acc:.2%}")


"""
Example: Applying DynEx-CLoRA in a Continual Learning Pipeline
--------------------------------------------------------------

This example demonstrates how to dynamically expand and manage LoRA adapters 
between Period 2 and Period 3.

# ==== Period 2: After training ==== 
# Save class_features.pkl and model checkpoint including:
# - related_labels (mapping adapter groups to classes)
# - num_lora_groups (number of LoRA adapter groups added)

# ==== Period 3: Load previous state and train with DynEx-CLoRA ====

from dynex_clora import train_with_dynex_clora, ResNet18_1D_LoRA

# === Load previous features and model ===
with open("./Trained_models/DynEx_CLoRA_CIL_v3/Period_2/class_features.pkl", "rb") as f:
    class_features_dict = pickle.load(f)

checkpoint = torch.load("./Trained_models/DynEx_CLoRA_CIL_v3/Period_2/ResNet18_1D_LoRA_best.pth", map_location=device)
related_labels = checkpoint.get("related_labels", {"base": [0, 1]})
num_lora_groups = checkpoint.get("num_lora_groups", 0)

# === Construct current model ===
model = ResNet18_1D_LoRA(input_channels=12, output_size=8).to(device)
for _ in range(num_lora_groups):
    model.add_lora_adapter()

# === Load weights from Period 2 (exclude FC and LoRA) ===
model_dict = model.state_dict()
prev_state = checkpoint["model_state_dict"]
filtered_state = {
    k: v for k, v in prev_state.items()
    if k in model_dict and model_dict[k].shape == v.shape and not (k.startswith("fc") or "lora_adapter" in k)
}
model.load_state_dict(filtered_state, strict=False)

# === Define optimizer, scheduler, etc ===
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

# === Train ===
train_with_dynex_clora(
    model=model,
    teacher_model=None,
    output_size=8,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optimizer,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    num_epochs=200,
    batch_size=64,
    alpha=0.0,  # No distillation
    similarity_threshold=0.85,
    class_features_dict=class_features_dict,
    related_labels=related_labels,
    model_saving_folder="./Trained_models/DynEx_CLoRA_CIL_v3/Period_3",
    model_name="ResNet18_1D_LoRA",
    period=3,
    device=device
)
"""

"""
Guidelines for Using DynEx-CLoRA
-------------------------------

1. Period 1:
   - Train model from scratch.
   - Save related_labels = {"base": [class0, class1, ...]} to associate all classes with base layers.

2. Period 2 and beyond:
   - Load previous model checkpoint and class_features.pkl.
   - Call `model.add_lora_adapter()` for each group from the previous period.
   - Load weights **excluding** fc layer and lora_adapter weights.
   - Maintain consistency using:
     - `related_labels`: dict[int|str → list[int]] — maps adapter groups to class labels.
     - `num_lora_groups`: total number of adapter sets added so far (equals to how many times `add_lora_adapter()` was called).

3. LoRA Expansion:
   - New classes are compared to past class features.
   - If similarity < threshold for all existing classes, a new LoRA adapter group is created.

4. Selective Update:
   - Existing LoRA adapters are selectively unfrozen if their associated classes drift semantically.
   - Final classifier layer (`fc`) is always unfrozen.

5. Feature Representation:
   - Feature vectors extracted using `extract_features()` (typically before FC).
   - Cosine similarity guides adapter sharing or expansion.

6. Saving State:
   - Each period must save:
     - `class_features.pkl` for the next round.
     - `model_state_dict` including updated LoRA groups.
     - `related_labels` and `num_lora_groups` in checkpoint metadata.

7. When using Knowledge Distillation (optional):
   - Set `teacher_model` and `stable_classes` in `train_with_dynex_clora()`.
   - Use `alpha` to control balance between CE loss and distillation (e.g., 0.1 ~ 0.5).

"""