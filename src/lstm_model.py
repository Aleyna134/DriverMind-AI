import torch.nn as nn


class LSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size=13,
        hidden_size=64,
        num_layers=2,
        num_classes=3,
        dropout=0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        x = hidden[-1]

        x = self.dropout(x)

        x = self.fc(x)

        return x