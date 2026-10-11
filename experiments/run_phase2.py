import sys
import os
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import time
import json
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.discriminative import get_dataloaders
from models.baselines import create_cfc_baseline
from models.amcna import create_amcna
from models.amcna_v2 import create_amcna_v2
from trainers.trainer import ExperimentTrainer
from utils.seed import set_seed
from utils.metrics import count_parameters, match_parameters_by_hidden_dim

def align_shapes(out, y):
    if y.dim() == 2:
        y = y.unsqueeze(-1)
    elif y.dim() == 1:
        y = y.view(-1, 1, 1)
    if out.shape != y.shape:
        out = out[:, -y.shape[1]:, :]
    return out, y

def train_amcna_v2_with_decorr(model, tr, vl, ts, lr, epochs, patience, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=15, min_lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    lambda_decorr = 0.1
    best_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out, gains = model(x)
            out, y = align_shapes(out, y)
            task_loss = criterion(out, y)
            decorr = model.decorrelation_loss(gains)
            loss = task_loss + lambda_decorr * decorr
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device)
                out, gains = model(x)
                out, y = align_shapes(out, y)
                task_loss = criterion(out, y)
                val_loss += task_loss.item()
        
        val_loss /= len(vl)
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    if best_model is not None:
        model.load_state_dict(best_model)
    
    # test
    model.eval()
    correct = 0
    total = 0
    all_gains = []
    
    with torch.no_grad():
        for x, y in ts:
            x, y = x.to(device), y.to(device)
            out, gains = model(x)
            out, y = align_shapes(out, y)
            preds = (torch.sigmoid(out) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.numel()
            all_gains.append(gains)
            
    acc = correct / total if total > 0 else 0
    
    # measure latency
    times = []
    x_b, _ = next(iter(ts))
    x_b = x_b.to(device)
    with torch.no_grad():
        for _ in range(20):
            t0 = time.perf_counter()
            model(x_b)
            times.append((time.perf_counter() - t0) * 1000)
    latency_ms = float(np.mean(times[5:])) if len(times) > 5 else 0.0
    
    all_gains = torch.cat(all_gains, dim=0) # (test_size, seq, 5)
    return acc, latency_ms, all_gains

def get_pearson_matrix(gains):
    # gains: (N, seq, 5)
    b, s, c = gains.shape
    gains_flat = gains.view(b * s, c)
    gains_centered = gains_flat - gains_flat.mean(dim=0, keepdim=True)
    cov = torch.matmul(gains_centered.t(), gains_centered) / (b * s - 1)
    std = torch.sqrt(torch.diag(cov))
    std_matrix = torch.outer(std, std)
    std_matrix = torch.clamp(std_matrix, min=1e-8)
    corr = cov / std_matrix
    return corr

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tasks = ['oscillation', 'multiscale', 'changepoint']
    models = ['CfC', 'AMCNA_v1', 'AMCNA_v2']
    seeds = [42, 43, 44]
    
    epochs = 80
    patience = 25
    lr = 5e-3
    batch_size = 32
    num_samples = 800
    
    results = defaultdict(lambda: defaultdict(list))
    latencies = defaultdict(lambda: defaultdict(list))
    correlations = defaultdict(list)
    
    for task in tasks:
        print(f"\n{'='*50}\nTASK: {task}\n{'='*50}")
        
        for seed in seeds:
            set_seed(seed)
            tr, vl, ts = get_dataloaders(task, batch_size=batch_size, num_samples=num_samples)
            
            # Baseline is CfC
            base_model = create_cfc_baseline(32, input_size=1, output_size=1)
            target_params = count_parameters(base_model)
            
            for model_name in models:
                if model_name == 'CfC':
                    model = create_cfc_baseline(32, input_size=1, output_size=1)
                elif model_name == 'AMCNA_v1':
                    model, _ = match_parameters_by_hidden_dim(
                        lambda d: create_amcna(d, input_size=1, output_size=1),
                        target_params=target_params
                    )
                elif model_name == 'AMCNA_v2':
                    model, _ = match_parameters_by_hidden_dim(
                        lambda d: create_amcna_v2(d, input_size=1, output_size=1),
                        target_params=target_params
                    )
                
                model = model.to(device)
                
                if model_name == 'AMCNA_v2':
                    acc, lat, gains = train_amcna_v2_with_decorr(model, tr, vl, ts, lr, epochs, patience, device)
                    corr_matrix = get_pearson_matrix(gains)
                    correlations[task].append(corr_matrix.cpu())
                    results[task][model_name].append(acc)
                    latencies[task][model_name].append(lat)
                else:
                    trainer = ExperimentTrainer(model, tr, vl, ts, lr, epochs, patience, device)
                    res = trainer.train(verbose_every=0)
                    results[task][model_name].append(res['test_acc'])
                    latencies[task][model_name].append(res['inference_latency_ms'])
                    
                print(f"[{seed}] {model_name} Params: {count_parameters(model)} | Acc: {results[task][model_name][-1]:.4f}")

    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    print(f"{'Task':<15} | {'Model':<12} | {'Acc (Mean ± Std)':<20} | {'Latency (ms)':<15}")
    print("-" * 70)
    for task in tasks:
        for model_name in models:
            accs = results[task][model_name]
            lats = latencies[task][model_name]
            acc_str = f"{np.mean(accs):.4f} ± {np.std(accs):.4f}"
            lat_str = f"{np.mean(lats):.2f}"
            print(f"{task:<15} | {model_name:<12} | {acc_str:<20} | {lat_str:<15}")
            
    print("\nAMCNA_v2 Gains Correlation Matrices (Mean across seeds):")
    for task in tasks:
        mean_corr = torch.mean(torch.stack(correlations[task]), dim=0)
        print(f"\nTask: {task}")
        print(mean_corr.numpy())

if __name__ == "__main__":
    main()
