"""
File: model.py
Description:
    Core model definitions for Financial Asset Market Trend Prediction experiments.
    Includes MLP, Bi-GRU, Bi-GRU with Attention, and ResNet18_1D variants.

Input Format:
    - Shape: (Batch, 1000, 7)
    - Description: Minute-level time-series technical indicators (1000 timesteps × 7 features).

Output:
    - Shape: (Batch, 1000, num_classes)
    - Description: Per-timestep class logits for trend classification.

Model Variants:
    ▸ MLP
        - Flattened time-series vector input; loses temporal structure.
        - Lightweight, non-sequential baseline.

    ▸ BiGRU
        - Bidirectional GRU with temporal modeling.
        - Better suited for sequence-based data.

    ▸ BiGRUWithAttention
        - BiGRU with element-wise attention mechanism.
        - Enhances focus on informative time steps.

    ▸ ResNet18_1D
        - 1D convolutional ResNet adapted from torchvision's ResNet18.
        - Preserves sequential features but performs poorly in this domain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class MLP(nn.Module):
    """
    A simple MLP model for time-series classification.
    This model processes each timestep as an independent feature vector (flattened MLP),
    and is suitable for shallow modeling when temporal structure is not critical.

    Args:
        input_size (int): Number of input features per timestep (e.g., 7 for BTC task).
        hidden_size (int): Number of units in each hidden layer.
        output_size (int): Number of output classes.
        dropout (float): Dropout rate applied after each activation (default 0.2).
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.2):
        super(MLP, self).__init__()
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
        if x.ndim == 3:
            B, T, D = x.shape
            x = x.view(B * T, D)
            is_sequence = True
        else:
            is_sequence = False

        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        if is_sequence:
            x = x.view(B, T, -1)
        return x


class BiGRU(nn.Module):
    """
    Bidirectional GRU for modeling sequential dependencies in time-series data.

    This model processes sequences using a multi-layer bidirectional GRU and applies a linear
    classification head on each timestep. Suitable for tasks with preserved temporal structure.

    Args:
        input_size (int): Number of input features at each timestep.
        hidden_size (int): Hidden units per GRU direction.
        output_size (int): Number of output classes.
        num_layers (int): Number of GRU layers.
        dropout (float): Dropout rate applied to GRU outputs and classifier (only active if num_layers > 1).
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int, dropout: float = 0.0):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0.0)

        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class BiGRUWithAttention(nn.Module):
    """
    Bidirectional GRU with element-wise attention mechanism for time-series classification.

    This model extends BiGRU by introducing a linear attention module that performs element-wise
    reweighting over the GRU outputs, enhancing the model’s focus on informative time steps.

    Args:
        input_size (int): Number of input features at each timestep.
        hidden_size (int): Hidden units per GRU direction.
        output_size (int): Number of output classes.
        num_layers (int): Number of GRU layers.
        dropout (float): Dropout rate applied after attention mechanism.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int, dropout: float = 0.0):
        super(BiGRUWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True, bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0.0)

        self.attention_fc = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)

        attn = torch.tanh(self.attention_fc(out))
        out = attn * out  # Element-wise attention
        out = self.dropout(out)
        out = self.fc(out)
        return out


class ResNet18_1D(nn.Module):
    """
    1D ResNet18 for time-series classification via convolutional feature extraction.

    This model adapts the original 2D ResNet18 from torchvision to 1D convolutions,
    suitable for time-series inputs with strong local temporal patterns. The final output
    is interpolated back to the original sequence length.

    Args:
        input_channels (int): Number of input features (channels) at each timestep.
        output_size (int): Number of output classes.
        seq_len (int): Original sequence length (e.g., 1000) to be restored via interpolation.
    """
    def __init__(self, input_channels: int, output_size: int, seq_len: int):
        super(ResNet18_1D, self).__init__()
        self.seq_len = seq_len

        base_model = resnet18(pretrained=False)

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.classifier = nn.Conv1d(512, output_size, kernel_size=1)

        self._convert_layers_to_1d()

    def _convert_layers_to_1d(self):
        for name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self, name)
            for block in layer:
                block.conv1 = nn.Conv1d(block.conv1.in_channels, block.conv1.out_channels,
                                        kernel_size=3, stride=block.conv1.stride[0],
                                        padding=1, bias=False)
                block.bn1 = nn.BatchNorm1d(block.bn1.num_features)
                block.conv2 = nn.Conv1d(block.conv2.in_channels, block.conv2.out_channels,
                                        kernel_size=3, stride=1, padding=1, bias=False)
                block.bn2 = nn.BatchNorm1d(block.bn2.num_features)
                if block.downsample is not None:
                    conv = nn.Conv1d(block.downsample[0].in_channels,
                                     block.downsample[0].out_channels,
                                     kernel_size=1, stride=block.downsample[0].stride[0], bias=False)
                    bn = nn.BatchNorm1d(block.downsample[1].num_features)
                    block.downsample = nn.Sequential(conv, bn)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, T, D) → (B, D, T)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.classifier(x)
        x = F.interpolate(x, size=self.seq_len, mode='linear', align_corners=False)
        x = x.permute(0, 2, 1)  # (B, output_size, T) → (B, T, output_size)
        return x
