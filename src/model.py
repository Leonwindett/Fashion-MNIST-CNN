"""
This module contains the model architecture for the Fashion MNIST classification task.
"""


import torch 
import torch.nn as nn

import torch
import torch.nn as nn

class Simple_Model(nn.Module):
    def __init__(self, out_channels, kernel_size, input_size=(1, 28, 28), num_classes=10):
        super().__init__()
        self.H_in, self.W_in = input_size[1], input_size[2]
        self.H_conv, self.W_conv = self.H_in - (kernel_size - 1), self.W_in - (kernel_size - 1)

        self.conv = nn.Conv2d(in_channels=input_size[0],
                              out_channels=out_channels,
                              kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(in_features=out_channels * (self.H_conv // 2) * (self.W_conv // 2),
                            out_features=num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x