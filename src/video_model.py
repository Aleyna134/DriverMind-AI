"""
video_model.py

MLP baseline for video features extracted
with ResNet50.
"""

import torch
import torch.nn as nn


class VideoClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 3),

        )

    def forward(self, x):

        return self.network(x)