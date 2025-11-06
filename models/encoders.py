"""
encoders whose input is group of control points of bazier pieces
"""

import torch
import math
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor

from typing import Callable, Union, Optional
from dgl.nn.pytorch.glob import MaxPooling
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.conv import NNConv


class BezierEncoderMLP_(nn.Module):
    def __init__(self, out_dim=64, input_dim=28 * 4, hidden_dim=256):
        super().__init__()
        self.mlp = _MLP(num_layers=3, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=out_dim)
        self.mlp2 = _MLP(num_layers=3, input_dim=out_dim, hidden_dim=hidden_dim, output_dim=out_dim)

    def forward(self, x: torch.Tensor):
        # x: control pts in shape [batch,dim,h,w]
        # x: output: [batch, out_dim]
        x = self.mlp(x)
        x = x + self.mlp2(x)
        return x

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding.

    Args:
        d_model: The dimension of the embedding vector.
        max_len: The maximum length of input sequences.
        dropout: The dropout probability.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 8000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape (max_len, d_model) with positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute the div_term as specified in the paper
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sinusoidal functions to even and odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices

        pe = pe.unsqueeze(0)  # Shape becomes (1, max_len, d_model)
        self.register_buffer("pe", pe)  # Register as buffer to avoid updating during training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input embeddings.

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model) or
               (seq_len, batch_size, d_model)

        Returns:
            Tensor with positional encodings added, same shape as input.
        """
        # Determine the shape of input x
        if x.dim() == 3:
            # Assumes shape is (batch_size, seq_len, d_model)
            x = x + self.pe[:, : x.size(1), :]
        elif x.dim() == 2:
            # Assumes shape is (seq_len, d_model)
            x = x + self.pe[:, : x.size(0), :]
        else:
            raise ValueError("Input tensor must have 2 or 3 dimensions")

        return self.dropout(x)


class _NonLinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        """
        A 3-layer MLP with linear outputs

        Args:
            input_dim (int): Dimension of the input tensor
            num_classes (int): Dimension of the output logits
            dropout (float, optional): Dropout used after each linear layer. Defaults to 0.3.
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, num_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, inp):
        """
        Forward pass

        Args:
            inp (torch.tensor): Inputs features to be mapped to logits
                                (batch_size x input_dim)

        Returns:
            torch.tensor: Logits (batch_size x num_classes)
        """
        x = F.relu(self.bn1(self.linear1(inp)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class _MLP(nn.Module):
    """"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
        MLP with linear output
        Args:
            num_layers (int): The number of linear layers in the MLP
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden feature dimensions for all hidden layers
            output_dim (int): Output feature dimension

        Raises:
            ValueError: If the given number of layers is <1
        """
        super(_MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("Number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            # TODO: this could move inside the above loop
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class TransformerEncoderBLock(nn.Module):
    def __init__(self, input_dim, c_hidden, n_layers, n_heads, dropout=0.01, batch_first=True):
        """
        A transformer layer with a single attention layer
        """
        super().__init__()
        encoder_layers = nn.TransformerEncoderLayer(input_dim, n_heads, c_hidden, dropout, batch_first=batch_first)
        self.encoder = nn.TransformerEncoder(encoder_layers, n_layers)

    def forward(self, x, src_key_padding_mask=None, src_mask=None):
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask, mask=src_mask)
