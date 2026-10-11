import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Fija la semilla en todos los módulos para reproducibilidad absoluta."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Configuraciones deterministas
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Utilizar algoritmos deterministas en PyTorch
    torch.use_deterministic_algorithms(True, warn_only=True)
