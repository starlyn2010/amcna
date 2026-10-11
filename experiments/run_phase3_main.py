import os
import sys
import json
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(".")
sys.path.insert(0, PROJECT_ROOT)

from utils.seed import set_seed
from utils.metrics import count_parameters

# =======================
# Model Wrappers
# =======================
class CfCClassifier(nn.Module):
    def __init__(self, hidden_size, input_size=1, output_size=10):
        super().__init__()
        from ncps.torch import CfCCell
        self.hidden_size = hidden_size
        self.cfc = CfCCell(input_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        hx = torch.zeros(batch_size, self.hidden_size, device=x.device)
        ts = torch.ones(batch_size, 1, device=x.device)
        for t in range(seq_len):
            hx, _ = self.cfc(x[:, t, :], hx, ts)
        return self.out_proj(hx).unsqueeze(1)

class AMCNA_v2_Classifier(nn.Module):
    def __init__(self, hidden_size, input_size=1, output_size=10, deliberation_steps=1):
        super().__init__()
        from models.amcna_v2 import AMCNA_v2
        self.net = AMCNA_v2(input_size, hidden_size, output_size, deliberation_steps=deliberation_steps)
    def forward(self, x):
        out, gains = self.net(x)
        return out[:, -1:, :], gains

class LSTMClassifier(nn.Module):
    def __init__(self, hidden_size, input_size=1, output_size=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.out_proj = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        output, _ = self.lstm(x)
        return self.out_proj(output[:, -1:, :])

class GRUClassifier(nn.Module):
    def __init__(self, hidden_size, input_size=1, output_size=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.out_proj = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        output, _ = self.gru(x)
        return self.out_proj(output[:, -1:, :])

# =======================
# Data Loading
# =======================
def get_dataloaders(batch_size=128):
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1, 1))
    ])
    
    train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    
    train_size = 5000
    val_size = 1000
    test_size = 1000
    
    # Deterministic subset for faster training
    generator = torch.Generator().manual_seed(42)
    
    # We only take a subset from the datasets to drastically reduce time
    train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset) - train_size], generator=generator)
    val_dataset, _ = torch.utils.data.random_split(test_dataset, [val_size, len(test_dataset) - val_size], generator=generator)
    test_dataset, _ = torch.utils.data.random_split(test_dataset, [test_size, len(test_dataset) - test_size], generator=torch.Generator().manual_seed(43))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Fixed permutation for psMNIST
    np.random.seed(42)
    perm = torch.from_numpy(np.random.permutation(784)).long()
    
    return train_loader, val_loader, test_loader, perm

# =======================
# Decorrelation Loss
# =======================
def compute_decorrelation_loss(gains):
    if len(gains.shape) == 3:
        b, t, m = gains.shape
        gains = gains.view(-1, m)
    elif len(gains.shape) > 3:
        gains = gains.view(-1, gains.shape[-1])
        
    mean = gains.mean(dim=0, keepdim=True)
    centered = gains - mean
    cov = torch.matmul(centered.t(), centered) / (centered.size(0) - 1 + 1e-8)
    
    var = torch.diag(cov)
    std = torch.sqrt(var + 1e-8)
    corr = cov / (torch.outer(std, std) + 1e-8)
    
    eye = torch.eye(corr.size(0), device=corr.device)
    loss = torch.norm(corr - eye, p='fro')
    return loss

# =======================
# Training Loop
# =======================
def train_model(model, train_loader, val_loader, test_loader, perm, task, model_name, lr, device, seed):
    set_seed(seed)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    patience = 8
    epochs = 25
    
    val_acc_curve = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for x, y in train_loader:
            if task == 'psmnist':
                x = x[:, perm, :]
            
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            out = model(x)
            if isinstance(out, tuple):
                logits, gains = out
                logits = logits.squeeze(1)
                loss_ce = criterion(logits, y)
                loss_dec = compute_decorrelation_loss(gains)
                loss = loss_ce + 0.1 * loss_dec
            else:
                logits = out.squeeze(1)
                loss = criterion(logits, y)
                
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                if task == 'psmnist':
                    x = x[:, perm, :]
                x, y = x.to(device), y.to(device)
                
                out = model(x)
                if isinstance(out, tuple):
                    logits, _ = out
                else:
                    logits = out
                    
                logits = logits.squeeze(1)
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                
        val_acc = correct / total
        val_acc_curve.append(val_acc)
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            
        print(f"Task: {task} | Model: {model_name} | Seed: {seed} | Epoch {epoch+1}/{epochs} - Val Acc: {val_acc:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
            
    # Test phase
    model.load_state_dict(best_model_state)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            if task == 'psmnist':
                x = x[:, perm, :]
            x, y = x.to(device), y.to(device)
            out = model(x)
            if isinstance(out, tuple):
                logits, _ = out
            else:
                logits = out
            logits = logits.squeeze(1)
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
    test_acc = correct / total
    
    # Latency measurement
    latency_measurements = []
    dummy_x = torch.randn(1, 784, 1, device=device)
    with torch.no_grad():
        # Warm up
        for _ in range(5):
            model(dummy_x)
        # Measure
        for _ in range(20):
            start = time.perf_counter()
            model(dummy_x)
            end = time.perf_counter()
            latency_measurements.append((end - start) * 1000) # ms
            
    avg_latency = np.mean(latency_measurements[-15:])
    
    return test_acc, best_val_acc, val_acc_curve, avg_latency

# =======================
# Stats and Formatting
# =======================
def compute_cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pool_var = ((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof
    d = (np.mean(x) - np.mean(y)) / np.sqrt(pool_var)
    return d

def compute_statistics(task_results, ref_model='CfC'):
    stats_dict = {}
    ref_accs = task_results[ref_model]['test_accs']
    
    for m, r in task_results.items():
        accs = r['test_accs']
        mean_acc = np.mean(accs)
        std_acc = np.std(accs, ddof=1)
        ci_95 = stats.t.interval(0.95, len(accs)-1, loc=mean_acc, scale=stats.sem(accs))
        lat = np.mean(r['latencies'])
        
        # Compare with ref
        if m == ref_model:
            d = float('nan')
            p = float('nan')
        else:
            d = compute_cohens_d(accs, ref_accs)
            # Shapiro-Wilk test for normality
            _, p_shapiro_m = stats.shapiro(accs)
            _, p_shapiro_ref = stats.shapiro(ref_accs)
            
            if p_shapiro_m > 0.05 and p_shapiro_ref > 0.05:
                _, p = stats.ttest_ind(accs, ref_accs, equal_var=False)
            else:
                _, p = stats.mannwhitneyu(accs, ref_accs)
                
        stats_dict[m] = {
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'ci': ci_95,
            'd': d,
            'p': p,
            'lat': lat,
            'params': r['params']
        }
    return stats_dict

def print_table(task_name, stats_dict, ref_model='CfC'):
    print(f"\n=== {task_name} Results (N=5) ===")
    print(f"{'Model':<12} | {'Params':<8} | {'Test Acc (Mean ± Std)':<22} | {'95% CI':<17} | {'vs CfC (Cohen d)':<19} | {'p-value':<7} | {'Latency (ms)':<12}")
    print("-" * 115)
    
    for m, s in stats_dict.items():
        ci_str = f"[{s['ci'][0]:.4f}, {s['ci'][1]:.4f}]"
        if m == ref_model:
            d_str = "—"
            p_str = "—"
        else:
            d_str = f"{s['d']:.2f}"
            p_str = f"{s['p']:.3f}"
            
        print(f"{m:<12} | {s['params']:<8} | {s['mean_acc']:.4f} ± {s['std_acc']:.4f} | {ci_str:<17} | {d_str:<19} | {p_str:<7} | {s['lat']:.2f}")

# =======================
# Main
# =======================
def main():
    device = torch.device('cpu')
    
    hpo_file = os.path.join(PROJECT_ROOT, "outputs", "hpo_results.json")
    hpo_results = {}
    if os.path.exists(hpo_file):
        with open(hpo_file, 'r') as f:
            hpo_results = json.load(f)
            
    models_dict = {
        'CfC': CfCClassifier,
        'AMCNA_v2': AMCNA_v2_Classifier,
        'LSTM': LSTMClassifier,
        'GRU': GRUClassifier
    }
    
    tasks = ['sMNIST', 'psMNIST']
    n_seeds = 5
    
    train_loader, val_loader, test_loader, perm = get_dataloaders(batch_size=128)
    
    results = {t: {m: {'test_accs': [], 'val_acc_curves': [], 'latencies': [], 'params': 0} for m in models_dict} for t in tasks}
    
    os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "figures"), exist_ok=True)
    
    for task in tasks:
        task_lower = task.lower()
        for model_name, model_cls in models_dict.items():
            # Get hyperparams
            if model_name in hpo_results:
                lr = hpo_results[model_name].get('lr', 1e-3)
                hidden_dim = hpo_results[model_name].get('hidden_dim', 96)
            else:
                lr = 1e-3
                hidden_dim = 96
                
            for seed in range(n_seeds):
                # Init model
                model = model_cls(hidden_dim, input_size=1, output_size=10).to(device)
                if seed == 0:
                    results[task][model_name]['params'] = count_parameters(model)
                    
                test_acc, best_val, curve, lat = train_model(
                    model, train_loader, val_loader, test_loader, perm, task_lower, model_name, lr, device, seed
                )
                
                results[task][model_name]['test_accs'].append(test_acc)
                results[task][model_name]['val_acc_curves'].append(curve)
                results[task][model_name]['latencies'].append(lat)
                
        # Stats & Table
        stats_dict = compute_statistics(results[task])
        print_table(task, stats_dict)
        
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, task in enumerate(tasks):
        ax = axes[i]
        for model_name in models_dict:
            curves = results[task][model_name]['val_acc_curves']
            max_len = max(len(c) for c in curves)
            # Pad with last value
            padded_curves = []
            for c in curves:
                padded = c + [c[-1]] * (max_len - len(c))
                padded_curves.append(padded)
            
            curves_arr = np.array(padded_curves)
            mean_curve = curves_arr.mean(axis=0)
            std_curve = curves_arr.std(axis=0)
            x_vals = np.arange(1, max_len + 1)
            
            ax.plot(x_vals, mean_curve, label=model_name)
            ax.fill_between(x_vals, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
            
        ax.set_title(f"{task} Validation Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, "figures", "phase3_curves.png"))
    
    # Save results
    with open(os.path.join(PROJECT_ROOT, "outputs", "phase3_results.json"), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    print("Starting Phase 3 Experiment...")
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    total_minutes = (end_time - start_time) / 60
    print(f"\n✅ All tests finished. Total execution time: {total_minutes:.2f} minutes.")
