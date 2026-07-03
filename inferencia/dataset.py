import os
import numpy as np
import torch
from torch.utils.data import Dataset
from prysm.polynomials import noll_to_nm, zernike_nm_sequence
from prysm.propagation import focus

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


class PSFGeneratorDataset(Dataset):
    """
    Dataset de simulación al vuelo (on-the-fly) en memoria RAM.
    Evita generar archivos físicos pesados y permite entrenar con infinitas muestras aleatorias.
    """
    def __init__(self, num_samples, model_type="phase_diversity", size=(256, 256)):
        self.num_samples = num_samples
        self.model_type = model_type  # "phase_diversity" or "resnet10"
        self.size = size
        
        # Inicializar malla y modos ópticos (Caché local)
        W, H = size
        y, x = np.mgrid[-1:1:H*1j, -1:1:W*1j]
        r = np.sqrt(x**2 + y**2)
        self.theta = np.arctan2(y, x)
        self.pupil = r <= 0.92
        
        r_norm = r / 0.92
        nms = [noll_to_nm(i) for i in range(1, 12)]
        self.modes = list(zernike_nm_sequence(nms, r_norm, self.theta, norm=True))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Variar D/r0 aleatoriamente
        if self.model_type == "resnet18":
            # Usar distribución Beta sesgada a la derecha (valores altos de turbulencia)
            # beta(2.0, 1.5) genera valores en [0, 1] acumulados hacia 1.0
            # Mapeamos [0, 1] al rango [0.1, 6.0]
            d_r0 = 0.1 + (6.0 - 0.1) * np.random.beta(a=2.0, b=1.5)
        else:
            d_r0 = np.random.uniform(0.1, 6.0)
            
        factor_kolmogorov = d_r0 ** (5.0 / 3.0)
        
        # 2. Generar coeficientes (Z1=0, Z2..Z11 estocásticos según Kolmogorov)
        coefs = np.zeros(11, dtype=np.float32)
        for i in range(1, 11):  # Z2 a Z11 (índices 1 a 10)
            mode_key = f"Z{i + 1}"
            base_var = NOLL_VARIANCES[mode_key]
            sigma = np.sqrt(base_var * factor_kolmogorov)
            coefs[i] = np.random.normal(0, sigma)
            
        # 3. Calcular fase en la pupila
        phase_data = np.zeros(self.size, dtype=np.float32)
        for coef, mode in zip(coefs, self.modes):
            if coef != 0.0:
                phase_data += coef * mode
                
        # 4. Generar PSF de 2 canales (Phase Diversity)
        # Canal 1: Foco normal
        wf1 = np.exp(1j * phase_data) * self.pupil
        psf1 = np.abs(focus(wf1, Q=2)) ** 2
        
        # Canal 2: Desfocada (añadimos +1.5 rad de Z4)
        # Z4 es el cuarto modo (índice 3 en self.modes)
        phase_defocus = phase_data + (1.5 * self.modes[3])
        wf2 = np.exp(1j * phase_defocus) * self.pupil
        psf2 = np.abs(focus(wf2, Q=2)) ** 2
        
        # Recortar 96x96 central
        H_img, W_img = psf1.shape
        cy, cx = H_img // 2, W_img // 2
        crop_half = 48
        
        psf1_crop = psf1[cy - crop_half:cy + crop_half, cx - crop_half:cx + crop_half]
        psf2_crop = psf2[cy - crop_half:cy + crop_half, cx - crop_half:cx + crop_half]
        
        # Normalizar individualmente
        max1 = np.max(psf1_crop)
        max2 = np.max(psf2_crop)
        psf1_norm = psf1_crop / max1 if max1 > 0 else psf1_crop
        psf2_norm = psf2_crop / max2 if max2 > 0 else psf2_crop
        
        # Aumento de datos (Data Augmentation) al vuelo para modelos profundos (resnet18)
        # Esto reduce el sobreajuste al emular ruido de sensor y fluctuaciones físicas
        if self.model_type == "resnet18":
            # 1. Ruido Gaussiano aditivo leve (sensibilidad del sensor de imagen)
            noise_level = np.random.uniform(0.002, 0.015)
            psf1_norm = psf1_norm + np.random.normal(0, noise_level, psf1_norm.shape)
            psf2_norm = psf2_norm + np.random.normal(0, noise_level, psf2_norm.shape)
            
            # Asegurar límites [0, 1] tras el ruido
            psf1_norm = np.clip(psf1_norm, 0.0, 1.0)
            psf2_norm = np.clip(psf2_norm, 0.0, 1.0)

            # 2. Pequeñas variaciones de escala/intensidad locales aleatorias
            psf1_norm *= np.random.uniform(0.98, 1.02)
            psf2_norm *= np.random.uniform(0.98, 1.02)
            psf1_norm = np.clip(psf1_norm, 0.0, 1.0)
            psf2_norm = np.clip(psf2_norm, 0.0, 1.0)

        psf_tensor = torch.stack([
            torch.from_numpy(psf1_norm.astype(np.float32)),
            torch.from_numpy(psf2_norm.astype(np.float32))
        ], dim=0)

        # Normalizar las etiquetas Zernike
        zernike_norm = coefs / STDS
        zernike_tensor = torch.from_numpy(zernike_norm)
        
        return psf_tensor, zernike_tensor


def denormalize_predictions(pred_tensor):
    """
    Toma las salidas del modelo normalizadas y las devuelve a sus escalas físicas de Zernike.
    """
    stds_tensor = torch.from_numpy(STDS).to(pred_tensor.device)
    return pred_tensor * stds_tensor
