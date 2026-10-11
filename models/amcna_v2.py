import torch
import torch.nn as nn
from ncps.torch import CfCCell

class AMCNA_v2(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1,
                 delay_steps: int = 5, deliberation_steps: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.delay_steps = delay_steps
        self.L = deliberation_steps
        self.env_proj = nn.Linear(input_size, hidden_size)
        self.w1 = nn.Linear(hidden_size, hidden_size, bias=False) # C1: Identity
        self.w2 = nn.Linear(hidden_size, hidden_size, bias=False) # C2: Integral
        self.w3 = nn.Linear(hidden_size, hidden_size, bias=False) # C3: Delay
        self.w4 = nn.Linear(hidden_size, hidden_size, bias=False) # C4: Derivative
        self.w5 = nn.Linear(hidden_size, hidden_size, bias=False) # C5: Gating linear
        self.v5 = nn.Linear(hidden_size, hidden_size, bias=True)  # C5: Gating nonlinear

        # Change A: Independent gain heads
        self.gain_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_size, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())
            for _ in range(5)
        ])

        self.cfc = CfCCell(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, output_size)
        self.alpha = 0.1
        self.beta = 0.5
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        device = x.device
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        h_prev = torch.zeros_like(h)
        z = torch.zeros_like(h)
        d = torch.zeros_like(h)
        delay_buffer = [torch.zeros(batch_size, self.hidden_size, device=device) for _ in range(self.delay_steps)]
        outputs = []
        gains_history = []
        ts = torch.ones(batch_size, 1, device=device)
        for t in range(seq_len):
            env_in = self.env_proj(x[:, t, :])
            for l in range(self.L):
                z = (1 - self.alpha) * z + self.alpha * h
                delta_h = h - h_prev
                d = self.beta * d + (1 - self.beta) * delta_h
                h_delay = delay_buffer[0]
                
                # Compute independent gains
                gain_list = [head(h) for head in self.gain_heads]
                gains = torch.cat(gain_list, dim=1) # (batch_size, 5)

                if l == self.L - 1:
                    gains_history.append(gains)
                    
                c1 = self.w1(h)
                c2 = self.w2(z)
                c3 = self.w3(h_delay)
                c4 = self.w4(d)
                c5 = torch.sigmoid(self.v5(h)) * self.w5(h)
                c_total = (gains[:, 0:1] * c1 + gains[:, 1:2] * c2 + gains[:, 2:3] * c3 + gains[:, 3:4] * c4 + gains[:, 4:5] * c5)
                node_input = c_total + env_in
                h_prev = h.clone()
                h, _ = self.cfc(node_input, h, ts)
            delay_buffer.pop(0)
            delay_buffer.append(h)
            outputs.append(h.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        gains_history_tensor = torch.stack(gains_history, dim=1) # (batch, seq, 5)
        return self.out_proj(outputs), gains_history_tensor

    def decorrelation_loss(self, gains):
        # gains: (batch, seq, 5)
        b, s, c = gains.shape
        gains_flat = gains.view(b * s, c)
        
        # Center the gains
        gains_centered = gains_flat - gains_flat.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix
        cov = torch.matmul(gains_centered.t(), gains_centered) / (b * s - 1)
        
        # Compute standard deviations
        std = torch.sqrt(torch.diag(cov))
        std_matrix = torch.outer(std, std)
        std_matrix = torch.clamp(std_matrix, min=1e-8)
        
        # Correlation matrix
        corr = cov / std_matrix
        
        # Identity matrix
        I = torch.eye(c, device=gains.device)
        
        # Frobenius norm of (corr - I)
        loss = torch.norm(corr - I, p='fro')**2
        return loss

def create_amcna_v2(hidden_dim: int, input_size: int = 1, output_size: int = 1, deliberation_steps: int = 1):
    return AMCNA_v2(input_size, hidden_dim, output_size, deliberation_steps=deliberation_steps)
