# DynEx-CLoRA

A modular framework for **Class-Incremental Continual Learning (CIL)**, supporting five mainstream strategies:

- **EWC** (Elastic Weight Consolidation)
- **MSE Distillation** (Knowledge Distillation via MSE Loss)
- **PNN** (Progressive Neural Networks)
- **LoRA** (Standard Low-Rank Adapter)
- **DynEx-CLoRA** (Ours: Dynamically Expandable LoRA with class-guided adaptation)

This repository provides reusable training **frameworks** and **task-specific implementations** across multiple domains, enabling easy experimentation and benchmarking for evolving continual learning problems.

---

## 🔧 Project Structure
```
DynEx_CLoRA/
│
├── ewc.py
├── mse.py
├── pnn.py
├── lora.py
├── dynex_clora.py
│
└── use_cases/
    ├── btc/ # Bitcoin Trend Prediction
    │ ├── btc_ewc.py
    │ ├── btc_mse.py
    │ ├── btc_pnn.py
    │ ├── btc_lora.py
    │ ├── btc_dynex_clora.py
    │ ├── model.py # Tested model(s)
    │ └── data_source.txt # Data preparation notes
    ├── cpsc2018/ # ECG Classification (CPSC2018)
    └── har/ # Human Activity Recognition
```

---

## 🚀 Features

- ✅ Implements five major continual learning methods: **EWC**, **MSE Distillation**, **PNN**, **Standard LoRA**, and **DynEx-CLoRA**
- ✅ Includes three real-world use cases (BTC trends, ECG classification, HAR) for evaluating method behavior
- ✅ Modular code structure — can be extended to new datasets or architectures

---

## 📁 `use_cases/` Structure

Each subfolder in `use_cases/` contains:

- ✅ 5 Python scripts: one per method (e.g., `btc_ewc.py`, `btc_dynex_clora.py`)
- ✅ `model.py`: candidate backbone models (tested and tuned for the task)
- ✅ Optional evaluation utilities (e.g., accuracy breakdown, model stats) are included in each script and can be integrated or modified as needed.

> ⚠️ **Note:** Datasets are not included in this repo. Please refer to each folder's `data_source.txt` for how to prepare input files.

---

## 🧠 Base Model Selection

For each use case, we tested several models (e.g., MLP, ResNet18_1D, BiGRUWithAttention) and selected the **most stable and performant one** to serve as the base model for all five continual learning methods.

| Use Case     | Selected Base Model     |
|--------------|--------------------------|
| `btc/`       | `BiGRUWithAttention`     |
| `har/`       | `MLP`                    |
| `cpsc2018/`  | `ResNet18_1D`            |

Each `model.py` includes candidate architectures for the task. You can select one to test, or swap in your own model implementation if preferred.

---

## 🧩 Method Frameworks (Top-Level `*.py`)

These files provide minimal yet complete implementations of each method’s core mechanism.
You are expected to adapt them with your own model, training data, and period-wise configuration logic.

| File             | Description                                                                             |
| ---------------- | --------------------------------------------------------------------------------------- |
| `ewc.py`         | Core logic for Elastic Weight Consolidation (EWC) with Fisher matrix support            |
| `mse.py`         | MSE-based knowledge distillation on stable classes using a frozen teacher               |
| `pnn.py`         | Progressive Neural Network with column-wise growth and lateral fusion                   |
| `lora.py`        | Standard LoRA fine-tuning with adapter initialization and trainable parameter selection |
| `dynex_clora.py` | Dynamic LoRA expansion based on class similarity and concept drift detection            |

---

## 📊 Experimental Results

To evaluate all five continual learning methods under consistent settings, we begin by reporting their **initial validation accuracy on Period 1**, using the selected backbone models for each task. These values establish a shared baseline for all subsequent comparisons.

### 🔹 Initial Validation Accuracy (Period 1)

| Task                       | Backbone Model        | Val Acc (%) |
|----------------------------|------------------------|-------------|
| ECG Classification         | ResNet18\_1D           | 88.86       |
| Human Activity Recognition | MLP                    | 95.31       |
| Financial Trend Prediction | BiGRU + Attention      | 98.35       |

> All methods start from these trained models, using only the initial subset of classes.

We now present the detailed results on one use case: **CPSC2018 ECG classification**, across four periods of class-incremental learning.

---

### 🔹 Continual Learning Performance (CPSC2018 ECG)

| Method            | Period | AA_old | AA_new | BWT    | FWT    |
|-------------------|--------|--------|--------|--------|--------|
| **DynEx-CLoRA**   | 2      | 85.69  | 92.68  | -3.17  | 46.26  |
|                   | 3      | 87.61  | 87.81  | -1.57  | 66.26  |
|                   | 4      | 93.67  | 73.21  | +5.18  | 74.55  |
| PNN               | 2      | 81.15  | 83.11  | -7.71  | 46.26  |
|                   | 3      | 82.00  | 84.46  | -0.13  | 61.63  |
|                   | 4      | 91.36  | 63.33  | +2.30  | 70.64  |
| EWC               | 2      | 86.98  | 90.11  | -1.88  | 44.16  |
|                   | 3      | 87.46  | 86.41  | -1.08  | 66.92  |
|                   | 4      | 91.01  | 70.60  | +2.60  | 73.18  |
| MSE Distillation  | 2      | 86.42  | 92.90  | -2.44  | 46.26  |
|                   | 3      | 84.46  | 87.66  | -5.20  | 63.84  |
|                   | 4      | 89.94  | 68.75  | +4.08  | 76.45  |
| Standard LoRA     | 2      | 82.83  | 91.07  | -6.03  | 39.72  |
|                   | 3      | 81.75  | 87.06  | -5.20  | 55.02  |
|                   | 4      | 84.38  | 59.22  | -0.27  | 69.17  |

> **AA_old**: Avg. accuracy on old classes  
> **AA_new**: Accuracy on new classes  
> **BWT**: Backward Transfer  
> **FWT**: Forward Transfer

---

### 🔹 Model Size and Growth Rate (CPSC2018 ECG)

| Method            | MS₁ (MB) | MS₂ | MS₃ | MS₄ | MGR (%) |
|-------------------|----------|-----|-----|-----|---------|
| DynEx-CLoRA       | 14.71    |14.84|14.85|14.86| +0.34   |
| Standard LoRA     | 14.71    |14.84|14.85|14.86| +0.34   |
| PNN               | 14.71    |33.43|52.15|70.87| +127.10 |
| EWC               | 14.71    |14.72|14.73|14.74| +0.07   |
| MSE Distillation  | 14.71    |14.72|14.73|14.74| +0.07   |

> **MSₜ**: Model size at Period *t*  
> **MGR**: Model Growth Rate from Period 1 → 4

---

### 🔹 Adapter Expansion & Efficiency (CPSC2018 ECG)

| Method            | Expansion Type  | TPR_max (%) |
|-------------------|------------------|-------------|
| DynEx-CLoRA       | Selective LoRA   | 54.67       |
| PNN               | Full Column      | 55.97       |
| EWC               | None             | 100.00      |
| MSE Distillation  | None             | 100.00      |
| Standard LoRA     | Fixed LoRA       | 1.05        |

> **TPR_max**: Maximum trainable parameter ratio across all periods

---

## 📐 DynEx-CLoRA Framework (Architecture)

**DynEx-CLoRA** is a dynamic continual learning framework that combines:

- 🧩 **Low-Rank Adapters (LoRA)** for efficient and modular updates
- 🔁 **Similarity-driven expansion** for handling evolving class semantics

Unlike traditional continual learning methods that treat all tasks equally or add fixed-size modules, DynEx-CLoRA evaluates **semantic similarity between new and existing classes** to decide how the model should evolve.

### 🔍 Key Concepts

- Each LoRA adapter group is linked to one or more classes.
- When new classes arrive, their feature representations are compared with past class prototypes using **cosine similarity**.
- Based on the similarity score:
  - ✅ Reuse: use the same adapter if a new class is semantically similar.
  - 🔓 Unfreeze: update an old adapter if a known class has drifted.
  - ➕ Expand: add a new LoRA adapter group for novel or dissimilar classes.

This results in a flexible architecture that **grows only when needed**, while keeping prior knowledge **frozen and intact**.

---

### 📐 Visual Overview

<p align="center">
  <img src="img/DynEx-CLoRA_Framework.png" alt="DynEx-CLoRA Architecture" width="95%">
</p>

### 📦 Internal Mechanism

At each new period:

1. The model computes **mean feature embeddings** for each class (i.e., class prototypes).
2. It compares new classes with previous ones using **cosine similarity**.
3. If a new class is:
   - Similar to a prior class: it's linked to that adapter group.
   - Very different: a new adapter group is added (`add_lora_adapter()`).
4. For existing classes with notable semantic change, their adapters are **unfrozen** for selective refinement.

The mapping between classes and adapters is stored in a dictionary called `related_labels`.  
This supports **modular updates** and avoids retraining the full model.

This strategy helps DynEx-CLoRA **balance knowledge retention and adaptation**, making it especially suitable for domains with evolving or ambiguous class definitions.

---

## 📝 Tips for Using This Repository

### ⚙️ General Setup

- Make sure to properly set:
  - Base directory (`BASE_DIR`)
  - Data files (`X_train`, `y_train`, `X_val`, etc.)
  - Save folder and stop-signal logic (`stop_signal_file`)

- Each task folder includes a `model.py` with pre-evaluated backbones.  
  You can use one of these, or **replace it with your own model**.

### ⚠️ Implementation Reminders per Method

| Method         | Key Considerations                                                                 |
|----------------|--------------------------------------------------------------------------------------|
| `ewc.py`       | After training each period, remember to store Fisher matrix and parameter snapshot |
| `mse.py`       | Freeze the teacher model, and adjust `alpha` to balance CE and distillation loss   |
| `pnn.py`       | Each new period must **recursively rebuild** the full prior PNN structure          |
| `lora.py`      | At Period 2, load weights *before* `init_lora()`; only train LoRA + final classifier layer in later periods |
| `dynex_clora.py` | Uses class similarity to trigger `add_lora_adapter()` and unfreeze logic. Adjust `similarity_threshold` (τ) based on dataset. Maintain `related_labels` to track adapter-to-class mapping |

---

Happy continual learning! ✨

