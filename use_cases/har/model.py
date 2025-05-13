"""
File: model.py
Description:
    Core model definitions for Human Activity Recognition (HAR) in continual learning experiments.
    Includes two MLP variants designed for flattened time-series feature input.

Input Format:
    - Shape: (Batch, 561)
    - Description: Flattened 561-dimensional statistical feature vector per HAR sample.

Output:
    - Model outputs are logits of shape (Batch, num_classes) for classification tasks.

Model Variants:
    ▸ HAR_MLP_v1
        - Lightweight baseline MLP with single hidden layer.
        - Suitable for simple classification tasks with limited capacity.

    ▸ HAR_MLP_v2
        - Deeper architecture with two hidden layers and batch normalization.
        - Provides better representation capacity and stability.
"""

import torch
import torch.nn as nn

class HAR_MLP_v1(nn.Module):
    """
    A simple MLP baseline for HAR classification.
    This model consists of a single hidden layer with dropout and ReLU activation.

    Args:
        input_size (int): Size of input features (default 561).
        hidden_size (int): Number of units in hidden layer.
        output_size (int): Number of output classes.
        dropout (float): Dropout probability applied after hidden layer.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.2):
        super(HAR_MLP_v1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class HAR_MLP_v2(nn.Module):
    """
    An enhanced MLP for HAR classification with two hidden layers,
    batch normalization, ReLU activations, and dropout.

    Args:
        input_size (int): Size of input features (default 561).
        hidden_size (int): Number of units in each hidden layer.
        output_size (int): Number of output classes.
        dropout (float): Dropout probability applied after each hidden layer.
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
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x
