import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from utils.seed import set_seed
from utils.metrics import match_parameters_by_hidden_dim, count_parameters, ComputationalCostTracker
from models.baselines import create_cfc_baseline
from models.amcna import create_amcna
from datasets.synthetic import get_dataloaders

def check_gradients_and_nans(model, loss):
    loss.backward()
    
    max_grad = 0.0
    sum_grad = 0.0
    count_grad = 0
    has_nans = False
    
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                print(f"[ERROR] NaN encontrado en gradiente de {name}")
                has_nans = True
            else:
                max_grad = max(max_grad, p.grad.abs().max().item())
                sum_grad += p.grad.abs().sum().item()
                count_grad += p.grad.numel()
                
    mean_grad = sum_grad / count_grad if count_grad > 0 else 0.0
    return not has_nans, max_grad, mean_grad

def run_sanity_check():
    set_seed(42)
    print("=== INICIANDO SANITY CHECK SUITE ===")
    
    # 1. Emparejamiento de Parámetros
    print("\n[1] Probando Emparejamiento de Parámetros...")
    # Configuramos AMCNA con una dimensión oculta arbitraria para el test (ej. d=16, 2 neuronas simuladas)
    amcna_hidden = 16
    amcna_model = create_amcna(hidden_dim=amcna_hidden, input_size=1, output_size=1, deliberation_steps=2)
    target_params = count_parameters(amcna_model)
    print(f"Parámetros objetivo (AMCNA d={amcna_hidden}): {target_params}")
    
    cfc_model, cfc_d = match_parameters_by_hidden_dim(
        lambda d: create_cfc_baseline(hidden_dim=d, input_size=1, output_size=1),
        target_params=target_params
    )
    print(f"Baseline CfC encontrado (d={cfc_d}): {count_parameters(cfc_model)} parámetros")
    
    amcna_no_delib, _ = match_parameters_by_hidden_dim(
        lambda d: create_amcna(hidden_dim=d, input_size=1, output_size=1, deliberation_steps=1),
        target_params=target_params
    )
    print(f"AMCNA Sin Deliberación encontrado (d={_}): {count_parameters(amcna_no_delib)} parámetros")
    
    # 2. Datasets
    print("\n[2] Probando Datasets (Forward)...")
    batch_size = 16
    seq_len = 20
    tr_dl, _, _ = get_dataloaders("xor", batch_size=batch_size, num_samples=100, seq_len=seq_len)
    x, y = next(iter(tr_dl))
    print(f"Batch X shape: {x.shape}, Y shape: {y.shape}")
    
    # 3. Forward y Backward (Sanity Check Numérico)
    print("\n[3] Probando Propagación y Gradientes...")
    models = {
        "AMCNA (Delib=2)": amcna_model,
        "AMCNA (Delib=1)": amcna_no_delib,
        "CfC Baseline": cfc_model
    }
    
    criterion = nn.MSELoss()
    tracker = ComputationalCostTracker()
    
    for name, m in models.items():
        print(f"\n--- Evaluando: {name} ---")
        
        # Test Forward Time & Mem
        tracker.start()
        out = m(x)
        fwd_time = tracker.stop()
        
        loss = criterion(out[:, -1, :], y)  # Evaluamos solo en el último step para XOR
        
        # Test Backward & NaNs
        tracker.start()
        ok, mx_grad, mn_grad = check_gradients_and_nans(m, loss)
        bwd_time = tracker.stop()
        
        macs = tracker.estimate_macs_rnn(m, seq_len=seq_len, batch_size=batch_size, input_dim=1, hidden_dim=m.hidden_size)
        
        print(f"Salida shape: {out.shape} - Esperado: ({batch_size}, {seq_len}, 1)")
        print(f"Forward Pass : {fwd_time*1000:.2f} ms")
        print(f"Backward Pass: {bwd_time*1000:.2f} ms")
        print(f"MACs Estimados (Batched): ~{macs:,}")
        print(f"Gradientes OK (sin NaNs): {ok}")
        print(f"Max Grad: {mx_grad:.6f} | Mean Grad: {mn_grad:.6f}")
        
    print("\n=== SANITY CHECK COMPLETADO ===")

if __name__ == "__main__":
    run_sanity_check()
