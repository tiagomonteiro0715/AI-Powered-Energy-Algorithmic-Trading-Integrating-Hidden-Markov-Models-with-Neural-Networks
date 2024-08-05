"""
MIT License

Copyright (c) 2024 Tiago Monteiro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
"""
DISCLAMER: This trading algorithm is provided for research purposes only and
does not constitute financial advice. Trading in financial markets involves
substantial risk and is not suitable for every investor. Past performance is
not indicative of future results. The author assumes no responsibility for any
financial losses or damages incurred as a result of using this software. Use
at your own risk.
"""

import torch.nn as nn  # Import PyTorch's neural network module
from AlgorithmImports import *  # Import necessary classes and methods from QuantConnect

class NeuralNetwork(nn.Module):
    """
    A neural network model implemented using PyTorch's nn.Module.

    - Consists of multiple hidden layers with ReLU activation functions.
    - Designed to process inputs and produce outputs through a series of linear transformations.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the neural network layers and activation function.

        :param input_size: Number of input features
        :param hidden_size: Number of neurons in each hidden layer
        :param output_size: Number of output features
        """
        super(NeuralNetwork, self).__init__()  # Call the parent class initializer

        # Define the hidden layers with specified input and output sizes
        self.hidden_layer1 = nn.Linear(input_size, hidden_size)    # First hidden layer
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)   # Second hidden layer
        self.hidden_layer3 = nn.Linear(hidden_size, hidden_size)   # Third hidden layer
        self.hidden_layer4 = nn.Linear(hidden_size, 5)             # Fourth hidden layer
        self.hidden_layer5 = nn.Linear(5, 1)                       # Fifth hidden layer

        # Define the output layer
        self.output_layer = nn.Linear(1, output_size)              # Final output layer

        # Define the activation function
        self.activation = nn.ReLU()  # ReLU activation function for non-linearity

    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        - Applies ReLU activation to each hidden layer.
        - Processes the input through the series of hidden layers to the output layer.

        :param x: Input tensor
        :return: Output tensor after passing through the network
        """
        # Pass the input through each layer with ReLU activation
        x = self.activation(self.hidden_layer1(x))
        x = self.activation(self.hidden_layer2(x))
        x = self.activation(self.hidden_layer3(x))
        x = self.activation(self.hidden_layer4(x))
        x = self.activation(self.hidden_layer5(x))

        # Pass through the output layer to get the final output
        x = self.output_layer(x)
        return x