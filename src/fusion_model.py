"""
fusion_model.py

Multimodal LSTM + ResNet50 Fusion model
for driver behaviour classification.
"""

import torch
import torch.nn as nn


class MultimodalFusion(nn.Module):

    def __init__(
        self,
        sensor_input_size=13,
        lstm_hidden_size=64,
        lstm_layers=2,
        dropout=0.3,
        video_feature_size=2048,
        video_embedding_size=256,
        num_classes=3,
    ):
        super().__init__()

        # --------------------------------------------------
        # Sensor Branch
        # --------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=sensor_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
        )

        # --------------------------------------------------
        # Video Branch
        # --------------------------------------------------

        self.video_encoder = nn.Sequential(
            nn.Linear(
                video_feature_size,
                video_embedding_size,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --------------------------------------------------
        # Fusion Classifier
        # --------------------------------------------------

        fusion_size = (
            lstm_hidden_size
            + video_embedding_size
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                fusion_size,
                128,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(
                128,
                num_classes,
            ),
        )

    def forward(
        self,
        sensor,
        video,
    ):

        # -------------------------------
        # LSTM
        # -------------------------------

        _, (hidden, _) = self.lstm(sensor)

        sensor_feature = hidden[-1]

        # -------------------------------
        # Video Encoder
        # -------------------------------

        video_feature = self.video_encoder(video)

        # -------------------------------
        # Fusion
        # -------------------------------

        fusion = torch.cat(
            (
                sensor_feature,
                video_feature,
            ),
            dim=1,
        )

        logits = self.classifier(fusion)

        return logits