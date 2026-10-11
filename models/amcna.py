import torch
import torch.nn as nn
from ncps.torch import CfCCell

class AMCNA(nn.Module):
    """
    Adaptive Multi-Channel Neural Architecture (AMCNA).
    Implementa 5 canales de comunicación recurrente ortogonal y un bucle de deliberación.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1,
                 delay_steps: int = 5, deliberation_steps: int = 1, mod_dim: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.delay_steps = delay_steps
        self.L = deliberation_steps
        
        # Proyección de entrada del entorno
        self.env_proj = nn.Linear(input_size, hidden_size)
        
        # Canales (K=5)
        self.w1 = nn.Linear(hidden_size, hidden_size, bias=False) # C1: Identity
        self.w2 = nn.Linear(hidden_size, hidden_size, bias=False) # C2: Integral
        self.w3 = nn.Linear(hidden_size, hidden_size, bias=False) # C3: Delay
        self.w4 = nn.Linear(hidden_size, hidden_size, bias=False) # C4: Derivative
        
        self.w5 = nn.Linear(hidden_size, hidden_size, bias=False) # C5: Gating linear part
        self.v5 = nn.Linear(hidden_size, hidden_size, bias=True)  # C5: Gating non-linear part
        
        # Estado Modulador Global
        self.mod_dim = mod_dim
        self.w_s = nn.Linear(mod_dim, mod_dim, bias=False)
        self.v_s = nn.Linear(hidden_size, mod_dim, bias=True)
        
        # Ganancias por canal a partir del modulador
        self.u_k = nn.Linear(mod_dim, 5, bias=False)
        
        # Dinámica de nodo base (CfC)
        # Recibe la suma de los canales como "input"
        self.cfc = CfCCell(hidden_size, hidden_size)
        
        self.out_proj = nn.Linear(hidden_size, output_size)
        
        # Constantes de filtrado
        self.alpha = 0.1  # Decay para el integrador (C2)
        self.beta = 0.5   # Decay para el derivativo (C4)
        
    def forward(self, x, return_gains: bool = False):
        # x: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Estados internos
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        h_prev = torch.zeros_like(h)
        z = torch.zeros_like(h)
        d = torch.zeros_like(h)
        s = torch.zeros(batch_size, self.mod_dim, device=device)
        
        # Buffer de estados para el delay (C3) implementado como lista (evita operaciones inplace)
        delay_buffer = [torch.zeros(batch_size, self.hidden_size, device=device) for _ in range(self.delay_steps)]
        
        outputs = []
        gains_history = []
        ts = torch.ones(batch_size, 1, device=device)
        
        for t in range(seq_len):
            env_in = self.env_proj(x[:, t, :])
            
            # Bucle deliberativo interno
            for l in range(self.L):
                # 1. Actualización de estados secundarios funcionales
                z = (1 - self.alpha) * z + self.alpha * h
                
                # Derivada filtrada (h - h_prev)
                delta_h = h - h_prev
                d = self.beta * d + (1 - self.beta) * delta_h
                
                # Retardo puro
                h_delay = delay_buffer[0]
                
                # 2. Modulador global
                s = torch.tanh(self.w_s(s) + self.v_s(h))
                # Ganancias (batch_size, 5)
                gains = torch.sigmoid(self.u_k(s)) 
                if return_gains and l == self.L - 1:
                    gains_history.append(gains.detach())
                
                # 3. Procesamiento de Canales
                c1 = self.w1(h)
                c2 = self.w2(z)
                c3 = self.w3(h_delay)
                c4 = self.w4(d)
                c5 = torch.sigmoid(self.v5(h)) * self.w5(h)
                
                # Mezcla modulada
                c_total = (gains[:, 0:1] * c1 + 
                           gains[:, 1:2] * c2 + 
                           gains[:, 2:3] * c3 + 
                           gains[:, 3:4] * c4 + 
                           gains[:, 4:5] * c5)
                
                # 4. Inyección del entorno (solo en el primer paso deliberativo o siempre? 
                # Generalmente se mantiene constante durante la deliberación)
                node_input = c_total + env_in
                
                h_prev = h.clone()
                # 5. Ecuación dinámica de nodo (CfC)
                h, _ = self.cfc(node_input, h, ts)
            
            # Actualizar buffer de delay al final del paso del entorno
            delay_buffer.pop(0)
            delay_buffer.append(h)
            
            outputs.append(h.unsqueeze(1))
            
        outputs = torch.cat(outputs, dim=1)
        if return_gains:
            # gains_history: (batch_size, seq_len, 5)
            return self.out_proj(outputs), torch.stack(gains_history, dim=1)
        return self.out_proj(outputs)

def create_amcna(hidden_dim: int, input_size: int = 1, output_size: int = 1, deliberation_steps: int = 1):
    return AMCNA(input_size, hidden_dim, output_size, deliberation_steps=deliberation_steps)
