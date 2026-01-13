import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for binary classification.
    
    Architecture:
    - Input layer: variable size (based on number of features)
    - Hidden layer 1: 64 neurons + ReLU + Dropout
    - Hidden layer 2: 32 neurons + ReLU + Dropout
    - Output layer: 1 neuron (binary classification logit)
    
    Args:
        input_dim (int): Number of input features
        dropout_rate (float): Dropout probability for regularization (default: 0.3)
    """
    
    def __init__(self, input_dim, dropout_rate=0.3):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.output = nn.Linear(32, 1)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 1)
        """
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.output(x)
        
        return x