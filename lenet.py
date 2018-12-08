import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential( nn.Conv2d(1, 6, kernel_size=3),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(2),
                                  nn.Conv2d(6, 16, kernel_size=3),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(2) )
        self.fc = nn.Sequential( nn.Linear(400,120),
                                 nn.ReLU(True),
                                 nn.Linear(120, 84),
                                 nn.ReLU(True),
                                 nn.Linear(84, 10) )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 400)
        x = self.fc(x)
        return x