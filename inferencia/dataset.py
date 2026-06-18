import os
import numpy as np
import torch
from torch.utils.data import Dataset

# Desviaciones estándar teóricas derivadas de las varianzas de Noll para Kolmogorov
NOLL_VARIANCES = {
    "Z1": 1.0,       # Piston (Usamos 1.0 para evitar división por cero, ya que es 0)
    "Z2": 0.448,     # Tip X
    "Z3": 0.448,     # Tilt Y
    "Z4": 0.0232,    # Defocus
    "Z5": 0.0232,    # Astigmatism 45°
    "Z6": 0.0232,    # Astigmatism 0°
    "Z7": 0.00619,   # Coma X
    "Z8": 0.00619,   # Coma Y
    "Z9": 0.00619,   # Trefoil X
    "Z10": 0.00619,  # Trefoil Y
    "Z11": 0.00244,  # Spherical
}

# Crear vector de desviaciones estándar en orden Z1 a Z11
STDS = np.array([np.sqrt(NOLL_VARIANCES[f"Z{i}"]) for i in range(1, 12)], dtype=np.float32)

class PSFDataset(Dataset):
    def __init__(self, data_dir, normalize_labels=True):
        self.data_dir = data_dir
        self.normalize_labels = normalize_labels
        
        # Encontrar todas las muestras indexadas buscando psf_*.npy
        self.sample_indices = sorted([
            int(f.split("_")[1].split(".")[0])
            for f in os.listdir(data_dir)
            if f.startswith("psf_") and f.endswith(".npy")
        ])
        
    def __len__(self):
        return len(self.sample_indices)
        
    def __getitem__(self, idx):
        sample_idx = self.sample_indices[idx]
        
        # Rutas de archivos
        psf_path = os.path.join(self.data_dir, f"psf_{sample_idx:05d}.npy")
        zernike_path = os.path.join(self.data_dir, f"zernike_{sample_idx:05d}.npy")
        
        # Cargar datos
        psf = np.load(psf_path).astype(np.float32)
        zernike = np.load(zernike_path).astype(np.float32)
        
        # Añadir canal para PyTorch: (96, 96) -> (1, 96, 96)
        psf_tensor = torch.from_numpy(psf).unsqueeze(0)
        
        # Normalizar etiquetas dividiendo por su desviación estándar
        if self.normalize_labels:
            zernike_norm = zernike / STDS
            zernike_tensor = torch.from_numpy(zernike_norm)
        else:
            zernike_tensor = torch.from_numpy(zernike)
            
        return psf_tensor, zernike_tensor

def denormalize_predictions(pred_tensor):
    """
    Toma las salidas del modelo normalizadas y las devuelve a sus escalas físicas de Zernike.
    """
    stds_tensor = torch.from_numpy(STDS).to(pred_tensor.device)
    return pred_tensor * stds_tensor
