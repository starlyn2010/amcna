import os
import sys
import torch
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.seed import set_seed
from utils.metrics import match_parameters_by_hidden_dim, count_parameters
from datasets.synthetic import get_dataloaders
from models.baselines import create_cfc_baseline
from models.amcna import create_amcna
from trainers.trainer import ExperimentTrainer

def run_experiment(task_name, num_seeds=3, epochs=100, device="cpu"):
    results = {}
    
    tasks = {
        "xor": {"func": lambda: get_dataloaders("xor", batch_size=64, num_samples=300, seq_len=30)},
        "memory": {"func": lambda: get_dataloaders("memory", batch_size=64, num_samples=300, delay=15)},
        "copy": {"func": lambda: get_dataloaders("copy", batch_size=64, num_samples=300, S=5, D=10)}
    }
    
    print(f"\n========== TAREA: {task_name.upper()} ==========", flush=True)
    
    for seed in range(num_seeds):
        print(f"\n--- SEED {seed} ---", flush=True)
        set_seed(seed)
        
        tr_dl, vl_dl, ts_dl = tasks[task_name]["func"]()
        
        # Determine Input Size
        x_sample, y_sample = next(iter(tr_dl))
        input_size = x_sample.shape[2]
        
        # Models configuration (Param matched)
        amcna_hidden = 16
        amcna = create_amcna(hidden_dim=amcna_hidden, input_size=input_size, deliberation_steps=2)
        target_params = count_parameters(amcna)
        
        cfc, _ = match_parameters_by_hidden_dim(
            lambda d: create_cfc_baseline(hidden_dim=d, input_size=input_size), target_params)
            
        amcna_no_delib, _ = match_parameters_by_hidden_dim(
            lambda d: create_amcna(hidden_dim=d, input_size=input_size, deliberation_steps=1), target_params)
            
        models_to_test = {
            "CfC_Baseline": cfc,
            "AMCNA_NoDelib": amcna_no_delib,
            "AMCNA_Full": amcna
        }
        
        if seed not in results:
            results[seed] = {}
            
        for m_name, model in models_to_test.items():
            print(f"Entrenando {m_name}...")
            trainer = ExperimentTrainer(model, tr_dl, vl_dl, ts_dl, lr=5e-3, epochs=epochs, device=device)
            res = trainer.train()
            
            results[seed][m_name] = {
                "test_acc": res["test_acc"],
                "test_loss": res["test_loss"],
                "total_time": res["total_time"],
                "epochs_run": res["epochs_run"]
            }
            
            print(f"-> {m_name} Test Acc: {res['test_acc']:.4f} | Epochs: {res['epochs_run']} | Time: {res['total_time']:.2f}s")
            
            # Interpretability only for AMCNA and for first seed
            if seed == 0 and "AMCNA" in m_name:
                print(f"Registrando contribución de canales para {m_name}...")
                model.eval()
                with torch.no_grad():
                    # Check first batch
                    x, y = next(iter(ts_dl))
                    x = x.to(device)
                    out, gains = model(x, return_gains=True)
                    # gains shape: (batch, seq, 5) -> average over batch and seq
                    avg_gains = gains.mean(dim=(0, 1)).cpu().numpy()
                    print(f"   Ganancias promedio: C1(Id):{avg_gains[0]:.2f}, C2(Int):{avg_gains[1]:.2f}, "
                          f"C3(Del):{avg_gains[2]:.2f}, C4(Der):{avg_gains[3]:.2f}, C5(NL):{avg_gains[4]:.2f}")
                          
    return results

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")
    
    all_results = {}
    for task in ["xor", "memory", "copy"]:
        # Corriendo 1 semilla para validación super rápida
        all_results[task] = run_experiment(task, num_seeds=1, epochs=30, device=device)
        
    # Agregando reporte final
    print("\n\n================ REPORTE FINAL ================")
    for task, res in all_results.items():
        print(f"\n--- Tarea: {task} ---")
        models = res[0].keys()
        for m_name in models:
            accs = [res[s][m_name]["test_acc"] for s in res.keys()]
            times = [res[s][m_name]["total_time"] for s in res.keys()]
            print(f"{m_name:15s} | Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f} | Tiempo prom: {np.mean(times):.2f}s")
            
    with open("first_validation_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    main()
