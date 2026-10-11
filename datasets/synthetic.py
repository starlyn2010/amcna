import torch
from torch.utils.data import Dataset, DataLoader

class TemporalXORDataset(Dataset):
    """
    Temporal XOR Multi-Pulso: N pulsos discretos (0 o 1) se inyectan
    en posiciones aleatorias de una secuencia de ceros.
    El objetivo es predecir la paridad (XOR de todos los bits) al final.
    """
    def __init__(self, num_samples: int, seq_len: int, num_pulses: int = 2):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_pulses = num_pulses
        
        # X: (N, T, 1), Y: (N, 1)
        self.x = torch.zeros((num_samples, seq_len, 1), dtype=torch.float32)
        self.y = torch.zeros((num_samples, 1), dtype=torch.float32)
        
        for i in range(num_samples):
            positions = torch.randperm(seq_len)[:num_pulses]
            values = torch.randint(0, 2, (num_pulses,)).float()
            for pos, val in zip(positions, values):
                self.x[i, pos, 0] = val
            parity = values.sum().int().item() % 2
            self.y[i, 0] = float(parity)
            
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class DelayedMemoryDataset(Dataset):
    """
    Delayed Memory: Se inyecta 1 bit de información (0 o 1) en t=0.
    El resto son ceros. El modelo debe recordar el bit hasta t=D.
    """
    def __init__(self, num_samples: int, delay: int):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = delay + 1
        
        self.x = torch.zeros((num_samples, self.seq_len, 1), dtype=torch.float32)
        self.y = torch.zeros((num_samples, 1), dtype=torch.float32)
        
        for i in range(num_samples):
            val = float(torch.randint(0, 2, (1,)).item())
            self.x[i, 0, 0] = val
            self.y[i, 0] = val
            
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class SequenceCopyDataset(Dataset):
    """
    Sequence Copy: Una secuencia de S bits seguida de D ceros.
    El objetivo es reproducir los S bits al final.
    """
    def __init__(self, num_samples: int, S: int, D: int):
        super().__init__()
        self.num_samples = num_samples
        self.S = S
        self.D = D
        self.seq_len = S + D + S
        
        # X: (N, seq_len, 2) -> feature 0: bit, feature 1: flag de "inicio de respuesta"
        self.x = torch.zeros((num_samples, self.seq_len, 2), dtype=torch.float32)
        # Y: (N, seq_len, 1) -> target sequence, only last S steps matter
        self.y = torch.zeros((num_samples, self.seq_len, 1), dtype=torch.float32)
        
        for i in range(num_samples):
            bits = torch.randint(0, 2, (S,), dtype=torch.float32)
            self.x[i, :S, 0] = bits
            self.x[i, S+D, 1] = 1.0 
            self.y[i, S+D:, 0] = bits

    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_dataloaders(dataset_name: str, batch_size: int = 64, **kwargs):
    if dataset_name == "xor":
        ds = TemporalXORDataset(**kwargs)
    elif dataset_name == "memory":
        ds = DelayedMemoryDataset(**kwargs)
    elif dataset_name == "copy":
        ds = SequenceCopyDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    # Split 70 / 15 / 15
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
