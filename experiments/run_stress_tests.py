import os
import sys
import torch
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.seed import set_seed
from utils.metrics import match_parameters_by_hidden_dim, count_parameters
from datasets.synthetic import get_dataloaders
from models.baselines import create_cfc_baseline, create_mlp_baseline
from models.amcna import create_amcna
from trainers.trainer import ExperimentTrainer, compute_statistics

# Intentar importar Mutual Information
try:
    from sklearn.feature_selection import mutual_info_regression
except ImportError:
    # Fallback si sklearn no está instalado
    def mutual_info_regression(x, y):
        # Estimación básica por histograma
        x_bins = np.histogram_bin_edges(x, bins='auto')
        y_bins = np.histogram_bin_edges(y, bins='auto')
        c_xy, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
        p_xy = c_xy / np.sum(c_xy)
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)
        
        mi = 0.0
        for i in range(len(p_x)):
            for j in range(len(p_y)):
                if p_xy[i, j] > 0:
                    mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
        return [mi]

def run_delayed_memory_sweep(device):
    print("\n================ [STRESS TEST] DELAYED MEMORY SWEEP ================", flush=True)
    delays = [15, 30, 50, 100, 200, 300]
    results = {d: {} for d in delays}
    
    for d in delays:
        print(f"\n--- Evaluando Delay D = {d} ---", flush=True)
        set_seed(42)
        tr_dl, vl_dl, ts_dl = get_dataloaders("memory", batch_size=32, num_samples=500, delay=d)
        
        # Parámetros emparejados
        amcna_base = create_amcna(hidden_dim=16, input_size=1, deliberation_steps=2)
        target_params = count_parameters(amcna_base)
        
        models = {
            "MLP_Baseline": match_parameters_by_hidden_dim(lambda hd: create_mlp_baseline(hd, 1), target_params)[0],
            "CfC_Baseline": match_parameters_by_hidden_dim(lambda hd: create_cfc_baseline(hd, 1), target_params)[0],
            "AMCNA_Full": amcna_base
        }
        
        for name, model in models.items():
            print(f"  Entrenando {name}...", flush=True)
            trainer = ExperimentTrainer(model, tr_dl, vl_dl, ts_dl, lr=5e-3, epochs=40, patience=15, device=device)
            res = trainer.train(verbose_every=0)
            results[d][name] = res["test_acc"]
            print(f"    Test Acc: {res['test_acc']:.4f} | Latencia: {res['inference_latency_ms']:.2f}ms", flush=True)
            
    # Guardar resultados
    os.makedirs("research/amcna_phase1/outputs", exist_ok=True)
    with open("research/amcna_phase1/outputs/delayed_memory_sweep.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Graficar
    plt.figure(figsize=(8, 5))
    for name in ["MLP_Baseline", "CfC_Baseline", "AMCNA_Full"]:
        accs = [results[d][name] for d in delays]
        plt.plot(delays, accs, marker='o', label=name)
    plt.title("Delayed Memory: Accuracy vs Delay (Punto de Ruptura)")
    plt.xlabel("Delay Steps")
    plt.ylabel("Test Accuracy")
    plt.grid(True)
    plt.legend()
    plt.savefig("research/amcna_phase1/outputs/delayed_memory_sweep.png")
    plt.close()

def run_xor_pulse_sweep(device):
    print("\n================ [STRESS TEST] MULTI-PULSE XOR SWEEP ================", flush=True)
    pulses = [2, 3, 5, 7]
    results = {p: {} for p in pulses}
    
    for p in pulses:
        print(f"\n--- Evaluando {p} Pulsos ---", flush=True)
        set_seed(42)
        tr_dl, vl_dl, ts_dl = get_dataloaders("xor", batch_size=32, num_samples=600, seq_len=40, num_pulses=p)
        
        amcna_base = create_amcna(hidden_dim=16, input_size=1, deliberation_steps=2)
        target_params = count_parameters(amcna_base)
        
        models = {
            "MLP_Baseline": match_parameters_by_hidden_dim(lambda hd: create_mlp_baseline(hd, 1), target_params)[0],
            "CfC_Baseline": match_parameters_by_hidden_dim(lambda hd: create_cfc_baseline(hd, 1), target_params)[0],
            "AMCNA_Full": amcna_base
        }
        
        for name, model in models.items():
            print(f"  Entrenando {name}...", flush=True)
            trainer = ExperimentTrainer(model, tr_dl, vl_dl, ts_dl, lr=5e-3, epochs=40, patience=15, device=device)
            res = trainer.train(verbose_every=0)
            results[p][name] = res["test_acc"]
            print(f"    Test Acc: {res['test_acc']:.4f}", flush=True)
            
    with open("research/amcna_phase1/outputs/xor_pulse_sweep.json", "w") as f:
        json.dump(results, f, indent=4)

def run_deliberation_sweep(device):
    print("\n================ [STRESS TEST] BARRIDO DE DELIBERACIÓN (L) ================", flush=True)
    L_steps = [1, 2, 4, 8]
    results = {}
    
    set_seed(42)
    # Usando Sequence Copy
    tr_dl, vl_dl, ts_dl = get_dataloaders("copy", batch_size=32, num_samples=500, S=5, D=10)
    
    for L in L_steps:
        print(f"\n--- Evaluando L = {L} ---", flush=True)
        model = create_amcna(hidden_dim=16, input_size=2, deliberation_steps=L)
        trainer = ExperimentTrainer(model, tr_dl, vl_dl, ts_dl, lr=5e-3, epochs=30, patience=10, device=device)
        res = trainer.train(verbose_every=0)
        
        results[L] = {
            "test_acc": res["test_acc"],
            "latency_ms": res["inference_latency_ms"]
        }
        print(f"    Test Acc: {res['test_acc']:.4f} | Latency: {res['inference_latency_ms']:.2f}ms", flush=True)
        
    with open("research/amcna_phase1/outputs/deliberation_sweep.json", "w") as f:
        json.dump(results, f, indent=4)

def analyze_channel_independence(device):
    print("\n================ [INTERPRETACIÓN] CORRELACIÓN E INFORMACIÓN MUTUA DE CANALES ================", flush=True)
    set_seed(42)
    tr_dl, vl_dl, ts_dl = get_dataloaders("copy", batch_size=32, num_samples=300, S=5, D=10)
    
    # Evaluar AMCNA
    model = create_amcna(hidden_dim=16, input_size=2, deliberation_steps=2).to(device)
    trainer = ExperimentTrainer(model, tr_dl, vl_dl, ts_dl, lr=5e-3, epochs=20, patience=5, device=device)
    trainer.train(verbose_every=0)
    
    model.eval()
    x, _ = next(iter(ts_dl))
    x = x.to(device)
    
    # Extraer activaciones de canales
    # Para capturar la correlación temporal de las activaciones de los canales, añadimos hooks o modificamos localmente
    # Pero podemos usar la historia de ganancias del modulador como proxy directo de la correlación de control
    with torch.no_grad():
        _, gains = model(x, return_gains=True)
        # gains shape: (batch, seq, 5) -> aplanar para correlaciones individuales
        gains_flat = gains.view(-1, 5).cpu().numpy()
        
    # Calcular Pearson
    pearson_matrix = np.corrcoef(gains_flat, rowvar=False)
    
    # Calcular Mutual Information
    mi_matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            if i == j:
                # MI consigo mismo
                mi_matrix[i, j] = 1.0
            else:
                mi_matrix[i, j] = mutual_info_regression(gains_flat[:, i].reshape(-1, 1), gains_flat[:, j])[0]
                
    print("\nMatriz de Correlación de Pearson (Gains):")
    print(np.round(pearson_matrix, 3))
    print("\nMatriz de Información Mutua (Gains):")
    print(np.round(mi_matrix, 3))
    
    # Guardar matrices
    np.save("research/amcna_phase1/outputs/pearson_matrix.npy", pearson_matrix)
    np.save("research/amcna_phase1/outputs/mi_matrix.npy", mi_matrix)

def run_ablation_study(device):
    print("\n================ [ABLACIÓN] ESTUDIO DE CASUALIDAD DE CANALES ================", flush=True)
    set_seed(42)
    # Tarea Copy
    tr_dl, vl_dl, ts_dl = get_dataloaders("copy", batch_size=32, num_samples=600, S=5, D=15)
    
    # Entrenar AMCNA base
    model = create_amcna(hidden_dim=16, input_size=2, deliberation_steps=2).to(device)
    trainer = ExperimentTrainer(model, tr_dl, vl_dl, ts_dl, lr=5e-3, epochs=45, patience=15, device=device)
    res = trainer.train(verbose_every=0)
    print(f"  AMCNA Full Test Acc: {res['test_acc']:.4f}", flush=True)
    
    # Ablaciones por modificación del forward desactivando pesos
    # C2 (Integrador)
    model_no_c2 = create_amcna(hidden_dim=16, input_size=2, deliberation_steps=2).to(device)
    model_no_c2.load_state_dict(model.state_dict())
    model_no_c2.w2.weight.data.fill_(0.0) # Apagar pesos del canal 2
    
    # C4 (Derivativo)
    model_no_c4 = create_amcna(hidden_dim=16, input_size=2, deliberation_steps=2).to(device)
    model_no_c4.load_state_dict(model.state_dict())
    model_no_c4.w4.weight.data.fill_(0.0) # Apagar pesos del canal 4
    
    print("  Evaluando impacto causal directo sin reentrenar:")
    
    # Re-evaluar temporalmente usando el modelo correspondiente
    original_model = trainer.model
    
    trainer.model = model_no_c2
    _, acc_no_c2 = trainer.evaluate(ts_dl)
    print(f"    Ablación C2 (Causal) Acc: {acc_no_c2:.4f}", flush=True)
    
    trainer.model = model_no_c4
    _, acc_no_c4 = trainer.evaluate(ts_dl)
    print(f"    Ablación C4 (Causal) Acc: {acc_no_c4:.4f}", flush=True)
    
    # Restaurar
    trainer.model = original_model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo de ejecución: {device}")
    
    os.makedirs("research/amcna_phase1/outputs", exist_ok=True)
    
    run_delayed_memory_sweep(device)
    run_xor_pulse_sweep(device)
    run_deliberation_sweep(device)
    analyze_channel_independence(device)
    run_ablation_study(device)
    
    print("\n========== FASE 1.5 COMPLETADA ==========")

if __name__ == "__main__":
    main()
