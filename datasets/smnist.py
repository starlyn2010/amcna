import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

class SequentialMNIST(torch.utils.data.Dataset):
    """
    Sequential MNIST: Each 28x28 image is flattened to a sequence of 784 timesteps,
    each with 1 feature (pixel intensity normalized to [0,1]).
    This is the standard benchmark for evaluating RNNs on long sequences.
    """
    def __init__(self, train=True, permutation=None, data_dir='./data'):
        self.mnist = datasets.MNIST(data_dir, train=train, download=True, 
                                     transform=transforms.ToTensor())
        self.permutation = permutation  # Fixed permutation for psMNIST
        
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        # img: (1, 28, 28) -> flatten to (784,) -> reshape to (784, 1)
        seq = img.view(-1)  # (784,)
        if self.permutation is not None:
            seq = seq[self.permutation]
        seq = seq.unsqueeze(-1)  # (784, 1)
        return seq, label  # label is an integer 0-9


def get_smnist_dataloaders(batch_size=128, permuted=False, data_dir='./data', num_workers=0):
    """
    Returns (train_loader, val_loader, test_loader) for sMNIST or psMNIST.
    Splits the 60k training set into 50k train + 10k validation.
    Test set is the standard 10k.
    """
    perm = None
    if permuted:
        # Fixed permutation, deterministic for reproducibility
        rng = np.random.RandomState(1999)
        perm = torch.from_numpy(rng.permutation(784)).long()
    
    train_full = SequentialMNIST(train=True, permutation=perm, data_dir=data_dir)
    test_ds = SequentialMNIST(train=False, permutation=perm, data_dir=data_dir)
    
    # Split train into train (50k) + val (10k)
    train_ds, val_ds = torch.utils.data.random_split(
        train_full, [50000, 10000],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
