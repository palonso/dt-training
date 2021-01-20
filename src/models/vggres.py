# Also heavily based on this:
# https://github.com/minzwon/sota-music-tagging-models/blob/master/training/model.py

import torch
import torch.nn as nn


class VGGRes(nn.Module):
    '''
    VGG blocks with residual conections.
    '''
    def __init__(self, n_channels=64, n_class=500):
        super().__init__()
        self.embedding_size = 128
        self.n_channels = n_channels

        self.bn2d = nn.BatchNorm2d(1)

        # CNN
        self.coupling1 = nn.Sequential(
            nn.Conv2d(1, n_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
        )

        self.coupling2 = nn.Sequential(
            nn.Conv2d(n_channels, 2 * n_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * n_channels),
            nn.ReLU(),
        )

        # nn.MaxPool2d(kernel_size=2, stride=2))
        self.block2 = nn.Sequential(
            nn.Conv2d(n_channels * 2, n_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_channels * 2),
            nn.ReLU(),
            nn.Conv2d(n_channels * 2, n_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_channels * 2),
            nn.ReLU(),
        )

        self.coupling3 = nn.Sequential(
            nn.Conv2d(2 * n_channels, 4 * n_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * n_channels),
            nn.ReLU(),
        )
            # nn.MaxPool2d(kernel_size=2, stride=2))
        self.block3 = nn.Sequential(
            nn.Conv2d(n_channels * 4, n_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_channels * 4),
            nn.ReLU(),
            nn.Conv2d(n_channels * 4, n_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_channels * 4),
            nn.ReLU(),
        )
            # nn.MaxPool2d(kernel_size=2, stride=4))

        self.coupling4 = nn.Sequential(
            nn.Conv2d(4 * n_channels, 4 * n_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4 * n_channels),
            nn.ReLU(),
        )

        # Dense
        self.embeddings = nn.Linear(n_channels * 4, self.embedding_size)
        self.bn = nn.BatchNorm1d(self.embedding_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(self.embedding_size, n_class)

        self.skip = nn.Identity()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=4)

    def forward(self, x):

        x = torch.unsqueeze(x, 1)

        # input normalization
        x = self.bn2d(x)

        # CNN
        inb1 = self.coupling1(x)
        outb11 = self.block1(inb1) + self.skip(inb1)
        # outb12 = self.block1(outb11) + self.skip(outb11)
        outb1 = self.pool(outb11)

        inb2 = self.coupling2(outb1)
        outb21 = self.block2(inb2) + self.skip(inb2)
        outb22 = self.block2(outb21) + self.skip(outb21)
        outb23 = self.block2(outb22) + self.skip(outb22)
        outb2 = self.pool(outb23)

        inb3 = self.coupling3(outb2)
        outb31 = self.block3(inb3) + self.skip(inb3)
        outb32 = self.block3(outb31) + self.skip(outb31)
        outb3 = self.pool(outb32)

        outb4 = self.coupling4(outb3)

        x = torch.flatten(outb4, start_dim=1, end_dim=-1)
        # x = torch.reshape(outb4, [-1, self.n_channels * 4])

        # Dense
        x = self.embeddings(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.dense2(x)

        return logits

