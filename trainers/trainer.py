import torch
import torch.nn as nn
import time
import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import auc

class ExperimentTrainer:
    """
    Trainer con logging completo: convergencia, AUC, FLOPs, latencia,
    memoria y estadísticos para comparación científica.
    """
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 lr=3e-3, epochs=500, patience=50, device="cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.patience = patience
        self.device = device
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=15, min_lr=1e-5)
        self.criterion = nn.BCEWithLogitsLoss()
        
    def _align_shapes(self, out, y):
        if y.dim() == 2:
            y = y.unsqueeze(-1)
        elif y.dim() == 1:
            y = y.view(-1, 1, 1)
        if out.shape != y.shape:
            out = out[:, -y.shape[1]:, :]
        return out, y
        
    def train(self, verbose_every: int = 10):
        best_val_loss = float('inf')
        patience_counter = 0
        best_val_acc = 0.0
        best_epoch = 0
        
        history = {
            "train_loss": [], "val_loss": [],
            "val_acc": [], "epoch_times": [],
            "gains_per_epoch": []  # Para curvas del modulador
        }
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            t0 = time.perf_counter()
            
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(x)
                out, y = self._align_shapes(out, y)
                loss = self.criterion(out, y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item()
                
            epoch_time = time.perf_counter() - t0
            train_loss /= len(self.train_loader)
            val_loss, val_acc = self.evaluate(self.val_loader)
            self.scheduler.step(val_loss)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["epoch_times"].append(epoch_time)
            
            # Registrar ganancias de canal si el modelo las expone
            if hasattr(self.model, 'forward') and 'return_gains' in self.model.forward.__code__.co_varnames:
                self.model.eval()
                with torch.no_grad():
                    try:
                        x_b, _ = next(iter(self.val_loader))
                        x_b = x_b.to(self.device)
                        _, g = self.model(x_b, return_gains=True)
                        avg_g = g.mean(dim=(0, 1)).cpu().numpy()
                        history["gains_per_epoch"].append(avg_g.tolist())
                    except Exception:
                        pass
                self.model.train()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                if verbose_every > 0:
                    print(f"   [Early stop en época {epoch+1}]", flush=True)
                break
                
            if verbose_every > 0 and (epoch + 1) % verbose_every == 0:
                print(f"   Epoch {epoch+1}/{self.epochs} | "
                      f"TL:{train_loss:.4f} VL:{val_loss:.4f} VA:{val_acc:.4f}", flush=True)
                
        # === Métricas finales ===
        test_loss, test_acc = self.evaluate(self.test_loader)
        
        # AUC de la curva de accuracy (eficiencia de aprendizaje)
        epochs_x = list(range(len(history["val_acc"])))
        if len(epochs_x) > 1:
            val_acc_list = history["val_acc"]
            acc_auc = float(sum((val_acc_list[i] + val_acc_list[i+1]) / 2.0 for i in range(len(val_acc_list) - 1))) / (len(epochs_x) - 1)
        else:
            acc_auc = 0.0
        
        # Latencia de inferencia
        lat_ms = self._measure_inference_latency()
        
        return {
            "history": history,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "auc_acc": acc_auc,
            "inference_latency_ms": lat_ms,
            "mean_epoch_time": float(np.mean(history["epoch_times"])),
            "epochs_run": len(history["train_loss"])
        }
        
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                out, y = self._align_shapes(out, y)
                loss = self.criterion(out, y)
                total_loss += loss.item()
                preds = (torch.sigmoid(out) > 0.5).float()
                correct += (preds == y).sum().item()
                total += y.numel()
                
        return total_loss / len(loader), correct / total if total > 0 else 0.0
    
    def _measure_inference_latency(self, n_runs: int = 20) -> float:
        """Mide latencia media de inferencia en ms."""
        self.model.eval()
        x_b, _ = next(iter(self.test_loader))
        x_b = x_b.to(self.device)
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                self.model(x_b)
                times.append((time.perf_counter() - t0) * 1000)
        return float(np.mean(times[5:]))  # Descartar warm-up


def compute_statistics(values: list, reference: list = None) -> dict:
    """
    Calcula estadísticos de un conjunto de valores.
    Si se provee 'reference', calcula Cohen's d y p-value vs referencia.
    """
    arr = np.array(values)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    ci_half = float(1.96 * std / np.sqrt(n)) if n > 1 else 0.0
    
    result = {
        "mean": mean,
        "std": std,
        "ci_95": (mean - ci_half, mean + ci_half),
        "n": n
    }
    
    if reference is not None and len(reference) > 1:
        ref = np.array(reference)
        # Cohen's d
        pooled_std = np.sqrt((arr.std(ddof=1)**2 + ref.std(ddof=1)**2) / 2)
        cohens_d = (mean - ref.mean()) / pooled_std if pooled_std > 0 else 0.0
        
        # Shapiro-Wilk para normalidad (decide entre t-test o Wilcoxon)
        _, p_sw_arr = scipy_stats.shapiro(arr) if n >= 3 else (None, 0.0)
        _, p_sw_ref = scipy_stats.shapiro(ref) if len(ref) >= 3 else (None, 0.0)
        
        if p_sw_arr > 0.05 and p_sw_ref > 0.05:
            _, p_val = scipy_stats.ttest_ind(arr, ref)
            test_used = "t-test"
        else:
            _, p_val = scipy_stats.mannwhitneyu(arr, ref, alternative='two-sided')
            test_used = "Mann-Whitney"
            
        result["cohens_d"] = float(cohens_d)
        result["p_value"] = float(p_val)
        result["test_used"] = test_used
        
    return result
