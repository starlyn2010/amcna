import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """LSTM baseline for sequence classification. Reads full sequence, predicts from last hidden state."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 10, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        output, (hn, cn) = self.lstm(x)
        # Use last timestep output
        return self.out_proj(output)  # (batch, seq_len, output_size) - trainer will slice last


class GRUBaseline(nn.Module):
    """GRU baseline for sequence classification."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 10, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        output, hn = self.gru(x)
        return self.out_proj(output)


def create_lstm_baseline(hidden_dim: int, input_size: int = 1, output_size: int = 10):
    return LSTMBaseline(input_size, hidden_dim, output_size)

def create_gru_baseline(hidden_dim: int, input_size: int = 1, output_size: int = 10):
    return GRUBaseline(input_size, hidden_dim, output_size)
