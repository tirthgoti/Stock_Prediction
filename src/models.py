"""Neural forecasting models with a shared regression and direction head."""

import torch
from torch import nn


class RecurrentModel(nn.Module):
    """LSTM/GRU model that predicts price and direction from the last state."""

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float, kind: str
    ) -> None:
        super().__init__()
        recurrent = nn.LSTM if kind == "lstm" else nn.GRU
        self.encoder = recurrent(
            input_size, hidden_size, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.price_head = nn.Linear(hidden_size, 1)
        self.direction_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return next-close regression and direction logits."""
        output, _ = self.encoder(x)
        state = self.dropout(output[:, -1, :])
        return self.price_head(state).squeeze(-1), self.direction_head(state).squeeze(-1)


class LSTMModel(RecurrentModel):
    """Long Short-Term Memory forecasting model."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__(input_size, hidden_size, num_layers, dropout, "lstm")


class GRUModel(RecurrentModel):
    """Gated Recurrent Unit forecasting model."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__(input_size, hidden_size, num_layers, dropout, "gru")


class TransformerModel(nn.Module):
    """Small Transformer encoder using the last token as the forecast state."""

    def __init__(
        self, input_size: int, hidden_size: int, heads: int, layers: int, dropout: float
    ) -> None:
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by transformer_heads.")
        self.projection = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=heads, dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.dropout = nn.Dropout(dropout)
        self.price_head = nn.Linear(hidden_size, 1)
        self.direction_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the window and return price and direction predictions."""
        state = self.dropout(self.encoder(self.projection(x))[:, -1, :])
        return self.price_head(state).squeeze(-1), self.direction_head(state).squeeze(-1)
