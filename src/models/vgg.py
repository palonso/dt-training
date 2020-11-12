import torch
import torch.nn as nn


class VGG(nn.Module):    
    '''
    Short-chunk CNN architecture.
    So-called vgg-ish model with a small receptive field.
    Deeper layers, smaller pooling (2x2).
    '''
    def __init__(self, n_channels=128, n_class=200):
        super(VGG, self).__init__()
        # CNN
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, n_channels, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels * 2, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(n_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(n_channels * 2, n_channels * 2, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(n_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(n_channels * 2, n_channels * 2, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(n_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer6 = nn.Sequential(
            nn.Conv2d(n_channels * 2, n_channels * 2, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(n_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer7 = nn.Sequential(
            nn.Conv2d(n_channels * 2, n_channels * 4, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(n_channels * 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=4))

        # Dense
        self.dense1 = nn.Linear(n_channels * 4, n_channels * 4)
        self.bn = nn.BatchNorm1d(n_channels * 4)
        self.dense2 = nn.Linear(n_channels * 4, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        x = x.squeeze(2)
#         print(x.shape)
        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.dense2(x)
        sigmoid = nn.Sigmoid()(logits)

        return logits, sigmoid

