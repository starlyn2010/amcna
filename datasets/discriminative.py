import torch
import math
from torch.utils.data import Dataset, DataLoader

class DampedOscillationDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int = 50):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.x = torch.zeros((num_samples, seq_len, 1), dtype=torch.float32)
        self.y = torch.zeros((num_samples, 1), dtype=torch.float32)
        
        t = torch.linspace(0, 10, seq_len)
        for i in range(num_samples):
            A = torch.rand(1).item() * 0.5 + 0.5
            gamma = torch.rand(1).item() * 0.2 + 0.1
            omega = torch.rand(1).item() * 2 + 1.0
            phi = torch.rand(1).item() * 2 * math.pi
            
            signal = A * torch.exp(-gamma * t) * torch.sin(omega * t + phi)
            self.x[i, :, 0] = signal
            
            derivative = signal[-1] - signal[-2]
            self.y[i, 0] = 1.0 if derivative > 0 else 0.0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class MultiTimescaleDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int = 60):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.x = torch.zeros((num_samples, seq_len, 1), dtype=torch.float32)
        self.y = torch.zeros((num_samples, 1), dtype=torch.float32)
        
        f1_mean = 1.0 / 40.0
        f2_mean = 1.0 / 5.0
        
        t = torch.arange(seq_len, dtype=torch.float32)
        for i in range(num_samples):
            f1 = f1_mean * (1.0 + torch.rand(1).item() * 0.2 - 0.1)
            f2 = f2_mean * (1.0 + torch.rand(1).item() * 0.2 - 0.1)
            phi1 = torch.rand(1).item() * 2 * math.pi
            phi2 = torch.rand(1).item() * 2 * math.pi
            
            slow_comp = torch.sin(2 * math.pi * f1 * t + phi1)
            fast_comp = torch.sin(2 * math.pi * f2 * t + phi2)
            
            self.x[i, :, 0] = slow_comp + fast_comp
            self.y[i, 0] = 1.0 if slow_comp[-1] > 0 else 0.0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class ChangePointDetectionDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int = 80):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.x = torch.zeros((num_samples, seq_len, 1), dtype=torch.float32)
        self.y = torch.zeros((num_samples, 1), dtype=torch.float32)
        
        for i in range(num_samples):
            A = torch.rand(1).item() * 2 - 1
            B = torch.rand(1).item() * 2 - 1
            while abs(A - B) < 0.5:
                B = torch.rand(1).item() * 2 - 1
                
            t_change = torch.randint(10, seq_len - 20, (1,)).item()
            
            signal = torch.zeros(seq_len)
            signal[:t_change] = A
            signal[t_change:seq_len - 10] = B
            signal[seq_len - 10:] = 0.0
            
            noise = torch.randn(seq_len) * 0.1
            signal += noise
            
            self.x[i, :, 0] = signal
            self.y[i, 0] = 1.0 if B > A else 0.0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def get_dataloaders(dataset_name, batch_size=64, **kwargs):
    if dataset_name == 'oscillation':
        ds = DampedOscillationDataset(**kwargs)
    elif dataset_name == 'multiscale':
        ds = MultiTimescaleDataset(**kwargs)
    elif dataset_name == 'changepoint':
        ds = ChangePointDetectionDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    total = len(ds)
    tr_len = int(0.7 * total)
    val_len = int(0.15 * total)
    te_len = total - tr_len - val_len

    tr_ds, val_ds, te_ds = torch.utils.data.random_split(ds, [tr_len, val_len, te_len])

    return (
        DataLoader(tr_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(te_ds, batch_size=batch_size, shuffle=False)
    )
