"""
File: model.py
Description:
    Core model definitions for ECG classification in CPSC2018 experiments.
    Includes MLP, GRU-based models, and various ResNet18-1D variants adapted for multichannel time series.

Input Format:
    - Shape: (Batch, 5000, 12)
    - Description: 12-lead ECG signals of 5000 time steps per sample.

Output:
    - Model outputs are logits of shape (Batch, num_classes) for classification tasks.

Model Categories:
    ▸ MLP
        - Flattened ECG vector (5000×12) passed through two FC layers.
        - Lightweight baseline, good for non-sequential classification.

    ▸ BiGRU
        - Bidirectional GRU with mean pooling.
        - Suitable for capturing temporal dynamics in ECG signals.

    ▸ BiGRUWithAttention
        - Adds element-wise attention on top of BiGRU.
        - Emphasizes informative time steps, improves interpretability.

    ▸ ResNet18_1D
        - Vanilla ResNet18 adapted from torchvision with 1D convolutions.
        - Uses conv7-kernel, stride=2, and AdaptiveAvgPool1d.

    ▸ ResNet18_1D_big_ker
        - First convolution uses kernel_size=15 instead of 7.
        - Captures broader temporal context early on.

    ▸ ResNet18_1D_big_inplane
        - Customizable inplanes, combines both AdaptiveAvgPool1d and AdaptiveMaxPool1d.
        - Final FC receives concatenated pooled features.

    ▸ ResNet18_1D_SE
        - Adds SEBlock1D after each residual layer from torchvision's ResNet18.
        - Preserves original structure with channel attention.

    ▸ ResNet18_1D_SE_Standard
        - Fully customized SEBasicBlock1D residual layers with Squeeze-and-Excitation.
        - More flexible, modular SE-enhanced architecture.

    ▸ ResNet18_1D_HighRes
        - Disables downsampling: stride=1 and no maxpool.
        - Keeps full temporal resolution, suitable for subtle temporal features.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class MLP(nn.Module):
    """
    A simple MLP baseline for ECG classification.

    This model flattens the 12-lead ECG input into a vector and processes it using two
    fully connected layers with batch normalization and dropout.

    Args:
        input_dim (int): Total number of input features (e.g., 5000 × 12 = 60000).
        hidden_dim (int): Number of units in each hidden layer.
        output_dim (int): Number of output classes.
        dropout (float): Dropout rate applied after each activation (default: 0.2).
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.out = nn.Linear(hidden_dim, output_dim)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = x.view(B, -1)  # Flatten input to (B, 5000*12)
        x = self.drop1(self.relu1(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu2(self.bn2(self.fc2(x))))
        return self.out(x)  # Output shape: (B, output_dim)


class BiGRU(nn.Module):
    """
    A simple MLP baseline for ECG classification.

    This model flattens the 12-lead ECG input into a vector and processes it using two
    fully connected layers with batch normalization and dropout.

    Args:
        input_dim (int): Total number of input features (e.g., 5000 × 12 = 60000).
        hidden_dim (int): Number of units in each hidden layer.
        output_dim (int): Number of output classes.
        dropout (float): Dropout rate applied after each activation (default: 0.2).
    """
    def __init__(self, input_size=12, hidden_size=64, num_classes=2, num_layers=2, dropout=0.3):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, B, self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)        # Output: (B, 5000, 2*hidden_size)
        out = self.drop(out)
        out = out.mean(dim=1)          # Mean pooling across time
        return self.fc(out)            # Output shape: (B, num_classes)


class BiGRUWithAttention(nn.Module):
    """
    BiGRU with element-wise attention for ECG classification.

    This model adds an attention mechanism on top of the BiGRU outputs to emphasize
    important time steps. The result is aggregated by mean pooling before classification.

    Args:
        input_size (int): Number of input features at each time step (default: 12).
        hidden_size (int): Number of hidden units per GRU direction.
        output_size (int): Number of output classes.
        num_layers (int): Number of GRU layers.
        dropout (float): Dropout rate applied after attention (default: 0.0).
    """
    def __init__(self, input_size: int = 12, hidden_size: int = 64, output_size: int = 2,
                 num_layers: int = 2, dropout: float = 0.0):
        super(BiGRUWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.attention_fc = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)  # Shape: (B, 5000, 2*hidden_size)

        attn = torch.tanh(self.attention_fc(out))  # Attention: (B, 5000, 2*hidden_size)
        out = attn * out                           # Element-wise modulation
        out = self.dropout(out)

        out = out.mean(dim=1)                     # Mean pooling across time
        return self.fc(out)                       # Output shape: (B, output_size)


class ResNet18_1D(nn.Module):
    """
    Standard ResNet18 adapted for 1D ECG signals.

    This model uses the original ResNet18 backbone (converted to 1D convolutions),
    with the default kernel size 7 in the first conv layer and adaptive average pooling.
    Suitable for general ECG sequence classification.

    Args:
        input_channels (int): Number of input channels (e.g., 12 for 12-lead ECG).
        output_size (int): Number of output classes.
    """
    def __init__(self, input_channels: int, output_size: int):
        super().__init__()
        base_model = resnet18(pretrained=False)

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(512, output_size)

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
        x = x.permute(0, 2, 1)  # (B, D, T)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)
    

class ResNet18_1D_big_ker(nn.Module):
    """
    ResNet18-1D variant using a larger initial kernel (kernel_size=15).

    This model is designed to capture broader temporal patterns in the early stage.
    It uses a wider receptive field in the first convolution layer, and otherwise retains
    the same ResNet18 structure with 1D convolutions.

    Args:
        input_channels (int): Number of input channels.
        output_size (int): Number of output classes.
    """
    def __init__(self, input_channels: int, output_size: int):
        super().__init__()
        base_model = resnet18(pretrained=False)

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(512, output_size)

        self._convert_layers_to_1d()

    def _convert_layers_to_1d(self):
        for name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self, name)
            for block in layer:
                block.conv1 = nn.Conv1d(block.conv1.in_channels, block.conv1.out_channels,
                                        kernel_size=3, stride=block.conv1.stride[0], padding=1, bias=False)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (B, D, T)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)


class BasicBlock1d(nn.Module):
    """
    Basic residual block for 1D ResNet architectures.

    Consists of two 1D convolution layers with a residual connection.
    Can optionally apply downsampling on the shortcut path if dimensions mismatch.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int): Stride of the first convolution.
        downsample (nn.Module or None): Downsampling layer to match residual dimensions.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet18_1D_big_inplane(nn.Module):
    """
    ResNet18-1D variant with adjustable input plane width.

    This model uses custom `BasicBlock1d`, larger first conv kernel,
    and concatenates both average and max pooled features for final classification.
    Designed for better generalization and feature aggregation on ECG sequences.

    Args:
        input_channels (int): Number of input features per time step (e.g., 12).
        output_size (int): Number of output classes.
        inplanes (int): Width of the first convolution layer (default: 64).
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
        self.fc = nn.Linear(512 * BasicBlock1d.expansion * 2, output_size)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        avg_out = self.adaptiveavgpool(x)
        max_out = self.adaptivemaxpool(x)
        x = torch.cat([avg_out, max_out], dim=1).view(x.size(0), -1)

        x = self.dropout(x)
        return self.fc(x)


class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for 1D CNNs.

    Applies channel-wise recalibration using global average pooling followed by
    a bottleneck fully connected module (two-layer MLP with reduction ratio).

    Args:
        channel (int): Number of input/output channels.
        reduction (int): Reduction ratio for bottleneck FC layer (default: 16).
    """
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResNet18_1D_SE(nn.Module):
    """
    ResNet18-1D with inserted SE blocks after each residual layer.

    Retains the original ResNet18 structure (converted to 1D), and applies SE-based
    channel-wise attention after each residual block stage.

    Args:
        input_channels (int): Number of input channels (e.g., 12-lead ECG = 12).
        output_size (int): Number of output classes.
    """
    def __init__(self, input_channels: int, output_size: int):
        super().__init__()
        base_model = resnet18(pretrained=False)

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.se_blocks = nn.ModuleList([
            SEBlock1D(64), SEBlock1D(128), SEBlock1D(256), SEBlock1D(512)
        ])

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(512, output_size)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.se_blocks[0](self.layer1(x))
        x = self.se_blocks[1](self.layer2(x))
        x = self.se_blocks[2](self.layer3(x))
        x = self.se_blocks[3](self.layer4(x))

        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)


class SEBasicBlock1D(nn.Module):
    """
    Basic residual block with Squeeze-and-Excitation (SE) attention for 1D signals.

    This block integrates channel-wise attention after the residual transformation.
    Used as the core building block for custom SE-based ResNet variants.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int): Convolution stride (default: 1).
        downsample (nn.Module or None): Optional downsampling layer for shortcut.
        reduction (int): SE bottleneck reduction ratio (default: 16).
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SEBlock1D(planes, reduction)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class ResNet18_1D_SE_Standard(nn.Module):
    """
    Fully customized ResNet18-1D using SEBasicBlock1D.

    This model replaces all residual blocks with custom SE-enabled blocks, allowing full
    control over SE placement and downsampling behavior.

    Args:
        input_channels (int): Number of input features per timestep.
        output_size (int): Number of output classes.
    """
    def __init__(self, input_channels: int, output_size: int):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(SEBasicBlock1D, 64, 2)
        self.layer2 = self._make_layer(SEBasicBlock1D, 128, 2, stride=2)
        self.layer3 = self._make_layer(SEBasicBlock1D, 256, 2, stride=2)
        self.layer4 = self._make_layer(SEBasicBlock1D, 512, 2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(512, output_size)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)


class ResNet18_1D_HighRes(nn.Module):
    """
    ResNet18-1D variant without downsampling for high-resolution ECG input.

    Replaces the first conv stride with 1 and disables max pooling to retain temporal
    resolution across all stages. Suitable for tasks where fine-grained temporal patterns matter.

    Args:
        input_channels (int): Number of input channels (e.g., 12).
        output_size (int): Number of output classes.
    """
    def __init__(self, input_channels: int, output_size: int):
        super().__init__()
        base_model = resnet18(pretrained=False)

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()  # no downsampling

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(512, output_size)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)  # identity

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)
