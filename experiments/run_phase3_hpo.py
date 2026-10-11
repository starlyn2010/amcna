import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')

# Insert project root into path
PROJECT_ROOT = os.path.abspath(".")
sys.path.insert(0, PROJECT_ROOT)

from utils.seed import set_seed
from models.amcna_v2 import AMCNA_v2
from ncps.torch import CfCCell

class CfCClassifier(nn.Module):
    def __init__(self, hidden_size, input_size=1, output_size=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.cfc = CfCCell(input_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        hx = torch.zeros(batch_size, self.hidden_size, device=x.device)
        ts = torch.ones(batch_size, 1, device=x.device)
        for t in range(seq_len):
            hx, _ = self.cfc(x[:, t, :], hx, ts)
        return self.out_proj(hx).unsqueeze(1)  # (batch, 1, 10)

class AMCNA_v2_Classifier(nn.Module):
    def __init__(self, hidden_size, input_size=1, output_size=10, deliberation_steps=1):
        super().__init__()
        self.net = AMCNA_v2(input_size, hidden_size, output_size, deliberation_steps=deliberation_steps)
        
    def forward(self, x):
        out, gains = self.net(x)
        return out[:, -1:, :], gains  # last timestep only

class LSTMClassifier(nn.Module):
    def __init__(self, hidden_size, input_size=1, output_size=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.out_proj = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        output, _ = self.lstm(x)
        return self.out_proj(output[:, -1:, :])  # (batch, 1, 10)

class GRUClassifier(nn.Module):
    def __init__(self, hidden_size, input_size=1, output_size=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.out_proj = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        output, _ = self.gru(x)
        return self.out_proj(output[:, -1:, :])  # (batch, 1, 10)


def get_model(model_name, hidden_size):
    if model_name == "CfC":
        return CfCClassifier(hidden_size)
    elif model_name == "AMCNA_v2":
        return AMCNA_v2_Classifier(hidden_size)
    elif model_name == "LSTM":
        return LSTMClassifier(hidden_size)
    elif model_name == "GRU":
        return GRUClassifier(hidden_size)
    else:
        raise ValueError(f"Unknown model {model_name}")

def train_and_eval(model, train_loader, val_loader, lr, epochs=15, patience=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.view(-1, 784, 1).to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            out = model(x)
            if isinstance(out, tuple):
                out, gains = out
                
            logits = out[:, -1, :]
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.view(-1, 784, 1).to(device)
                y = y.to(device)
                
                out = model(x)
                if isinstance(out, tuple):
                    out, gains = out
                    
                logits = out[:, -1, :]
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    return best_val_acc

def main():
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    full_train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    # Split into train and val (e.g., 50k train, 10k val) for HPO
    train_size = 50000
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    models_to_tune = ["CfC", "AMCNA_v2", "LSTM", "GRU"]
    learning_rates = [1e-2, 3e-3, 1e-3, 3e-4]
    hidden_dims = [64, 96, 128]
    n_seeds = 3
    
    results = {}
    best_configs = {}
    
    outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    for model_name in models_to_tune:
        print(f"--- Tuning {model_name} ---")
        results[model_name] = {}
        best_mean_acc = 0.0
        best_cfg = None
        
        for hd in hidden_dims:
            for lr in learning_rates:
                cfg_name = f"hd_{hd}_lr_{lr}"
                print(f"  Config: {cfg_name}")
                accs = []
                for seed in range(n_seeds):
                    set_seed(42 + seed)
                    model = get_model(model_name, hd)
                    val_acc = train_and_eval(model, train_loader, val_loader, lr)
                    accs.append(val_acc)
                
                mean_acc = sum(accs) / len(accs)
                results[model_name][cfg_name] = {
                    "hidden_dim": hd,
                    "lr": lr,
                    "accs": accs,
                    "mean_acc": mean_acc
                }
                print(f"    Mean Val Acc: {mean_acc:.4f} {accs}")
                
                if mean_acc > best_mean_acc:
                    best_mean_acc = mean_acc
                    best_cfg = results[model_name][cfg_name]
                    
        best_configs[model_name] = best_cfg
        print(f"  Best {model_name}: {best_cfg['hidden_dim']} hd, {best_cfg['lr']} lr (Acc: {best_mean_acc:.4f})")
        
    print("\n=== Best Configurations ===")
    for model_name, cfg in best_configs.items():
        print(f"{model_name:10s} | HD: {cfg['hidden_dim']:3d} | LR: {cfg['lr']:.4e} | Val Acc: {cfg['mean_acc']:.4f}")
        
    with open(os.path.join(outputs_dir, "hpo_results.json"), "w") as f:
        json.dump({"all_results": results, "best_configs": best_configs}, f, indent=4)
        
if __name__ == "__main__":
    main()
