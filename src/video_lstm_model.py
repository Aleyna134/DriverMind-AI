"""
video_lstm_model.py

LSTM model for temporal video features.
"""

import torch
import torch.nn as nn


class VideoLSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size=2048,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        num_classes=3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):

        _, (hidden, _) = self.lstm(x)

        last_hidden = hidden[-1]

        logits = self.classifier(last_hidden)

        return logits