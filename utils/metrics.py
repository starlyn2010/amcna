import torch
import torch.nn as nn
import time

def count_parameters(model: nn.Module) -> int:
    """Devuelve el número total de parámetros entrenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def match_parameters_by_hidden_dim(
    create_model_fn, 
    target_params: int, 
    dim_range: range = range(2, 256), 
    tolerance: float = 0.05
):
    """
    Busca heurísticamente la dimensión oculta (d) que hace que el modelo creado
    por `create_model_fn(d)` tenga un número de parámetros dentro de un +-5% 
    respecto a `target_params`.
    """
    best_d = None
    best_diff = float('inf')
    best_model = None
    
    for d in dim_range:
        try:
            model = create_model_fn(d)
        except Exception:
            continue
            
        params = count_parameters(model)
        diff = abs(params - target_params)
        
        if diff < best_diff:
            best_diff = diff
            best_d = d
            best_model = model
            
        # Si la diferencia es de crecimiento monótono y ya pasamos el target
        if params > target_params * 1.5:
            break
            
    if best_diff / target_params > tolerance:
        print(f"[WARNING] No se pudo emparejar parámetros en < {tolerance*100}%. "
              f"Target: {target_params}, Best: {count_parameters(best_model)} (d={best_d})")
    
    return best_model, best_d

class ComputationalCostTracker:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        
    def start(self):
        self.start_time = time.time()
        
    def stop(self):
        self.end_time = time.time()
        return self.end_time - self.start_time
        
    def estimate_macs_rnn(self, model: nn.Module, seq_len: int, batch_size: int, input_dim: int, hidden_dim: int):
        """
        Estimación simple de MACs (Multiply-Accumulates) para la estructura de la Fase 1.
        Dado que PtFlops u otras librerías pueden fallar en bucles for internos de RNNs custom,
        hacemos un recuento basal. A mejorar si se requiere precisión exacta de hardware.
        """
        # Para redes donde matrix mul domina, MACs ~ params * seq_len * batch_size.
        # Es una heurística de primer orden útil para Sanity Checks CPU.
        params = count_parameters(model)
        return params * seq_len * batch_size
