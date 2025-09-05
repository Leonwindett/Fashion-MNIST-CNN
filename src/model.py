"""
This module contains the model architecture for the Fashion MNIST classification task.
"""


import torch 
import torch.nn as nn

def simple_model(out_channels, kernel_size, input_size = (1, 28, 28)):  

    H_in, W_in = input_size[1], input_size[2]
    H_conv, W_conv = H_in - (kernel_size - 1), W_in - (kernel_size - 1)

    model = nn.Sequential(nn.Conv2d(in_channels = input_size[0],
                                    out_channels = out_channels,
                                    kernel_size = kernel_size),
                        nn.ReLU(),

                        nn.MaxPool2d(kernel_size = 2,
                                    stride = 2), # Reduce the spatial dimensions by a factor of 2

                        nn.Flatten(start_dim = 1), # Flatten all dimensions but batch

                        nn.Linear(in_features =  out_channels * (H_conv // 2) * (W_conv // 2),
                                    out_features = 10)) # 10 output classes
    
    return model