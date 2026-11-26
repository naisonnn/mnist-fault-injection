# Multi-Layer Perceptron for recognising MNIST handwritten digits

# === Neural Network-Imports ===
from torch import nn

class MNIST_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.hidden = nn.Linear(28 * 28, 12)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(12, 10)

    def forward(self, x):
        x = self.flatten(x)         # [N, 1, 28, 28] -> [N, 784]
        x = self.hidden(x)          # [N, 784] -> [N, 12]
        x = self.relu(x)            # [N, 12] -> [N, 12]
        x = self.classifier(x)      # [N, 12] -> [N, 10]
        return x