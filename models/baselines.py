import torch
import torch.nn as nn
from ncps.torch import CfCCell

class CfCBaseline(nn.Module):
    """
    Baseline puro usando la implementación original de NCPs (CfCCell).
    Procesa secuencias paso a paso de manera idéntica a como lo haría AMCNA.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.cfc = CfCCell(input_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        hx = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        outputs = []
        ts = torch.ones(batch_size, 1, device=x.device)
        for t in range(seq_len):
            hx, _ = self.cfc(x[:, t, :], hx, ts)
            outputs.append(hx.unsqueeze(1))
            
        outputs = torch.cat(outputs, dim=1)
        return self.out_proj(outputs)


class MLPBaseline(nn.Module):
    """
    Baseline 'tonto' no-recurrente (MLP paso a paso / Time-Distributed).
    Sirve para evaluar si la tarea requiere memoria temporal real.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.hidden_size = hidden_size
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        return self.net(x)


# Funciones helper para el emparejamiento de parámetros
def create_cfc_baseline(hidden_dim: int, input_size: int = 1, output_size: int = 1):
    return CfCBaseline(input_size, hidden_dim, output_size)

def create_mlp_baseline(hidden_dim: int, input_size: int = 1, output_size: int = 1):
    return MLPBaseline(input_size, hidden_dim, output_size)
