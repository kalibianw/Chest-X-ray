import torch
import torch.nn as nn
import torch.functional as F

conv1 = nn.Conv2d(
    in_channels=1,
    out_channels=64,
    kernel_size=(3, 3),
    stride=1,
    padding=(1, 1)
)
print(conv1.weight.shape)
