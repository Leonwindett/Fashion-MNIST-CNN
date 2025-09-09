"""
This module contains the model architecture for the Fashion MNIST classification task.
"""

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
    

    # For more advanced architectures: https://www.reddit.com/r/learnmachinelearning/comments/1d4txo8/fashionmnist_best_accuracy/ ie GaP
    # Also add more convolutional layers with wider channels. Look into expanding the data set with rotations, translations etc. 


class FashionMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        
        self.layer9 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.batchnorm1d = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.batchnorm1d(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x