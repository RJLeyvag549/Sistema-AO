import os
import numpy as np
import time
from prysm.polynomials import noll_to_nm, zernike_nm_sequence
from prysm.propagation import focus

# Configuración del dataset
SHARED_DIR = "/app/shared"
DATASET_DIR = os.path.join(SHARED_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

# Crear directorios
for d in [TRAIN_DIR, VAL_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# Varianzas de Noll para Kolmogorov
NOLL_VARIANCES = {
    "Z1": 0.0,       # Piston
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

# Inicializar malla y modos ópticos (Caché local)
def init_grid_and_modes(size=(256, 256)):
    W, H = size
    y, x = np.mgrid[-1:1:H*1j, -1:1:W*1j]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    pupil = r <= 0.92
    
    r_norm = r / 0.92
    nms = [noll_to_nm(i) for i in range(1, 12)]
    modes = list(zernike_nm_sequence(nms, r_norm, theta, norm=True))
    
    return modes, pupil

MODES, PUPIL = init_grid_and_modes()

def generate_sample():
    # Variar la fuerza de la turbulencia D/r0 aleatoriamente entre 0.5 y 2.5
    d_r0 = np.random.uniform(0.5, 2.5)
    factor_kolmogorov = d_r0 ** (5.0 / 3.0)
    
    # Generar coeficientes de Zernike (Z1=0, Z2..Z11 estocásticos según Noll)
    coefs = np.zeros(11, dtype=np.float32)
    for idx in range(1, 11):  # Z2 a Z11 (índices 1 a 10)
        mode_key = f"Z{idx + 1}"
        base_var = NOLL_VARIANCES[mode_key]
        sigma = np.sqrt(base_var * factor_kolmogorov)
        coefs[idx] = np.random.normal(0, sigma)
        
    # Calcular fase en pupila
    phase_data = np.zeros((256, 256), dtype=np.float32)
    for coef, mode in zip(coefs, MODES):
        if coef != 0.0:
            phase_data += coef * mode
            
    # Propagación física al foco
    wf = np.exp(1j * phase_data) * PUPIL
    focal_wf = focus(wf, Q=2)
    psf = np.abs(focal_wf) ** 2
    
    # Recorte 96x96 central
    H, W = psf.shape
    cy, cx = H // 2, W // 2
    crop_half = 48
    psf_crop = psf[cy - crop_half:cy + crop_half, cx - crop_half:cx + crop_half]
    
    # Normalizar PSF a rango [0, 1]
    psf_max = np.max(psf_crop)
    if psf_max > 0:
        psf_norm = psf_crop / psf_max
    else:
        psf_norm = psf_crop
        
    return psf_norm.astype(np.float32), coefs

def build_dataset(num_samples, target_dir):
    print(f"Generando {num_samples} muestras en {target_dir}...", flush=True)
    t0 = time.time()
    for i in range(num_samples):
        psf, coefs = generate_sample()
        
        # Guardar en archivos binarios independientes
        np.save(os.path.join(target_dir, f"psf_{i:05d}.npy"), psf)
        np.save(os.path.join(target_dir, f"zernike_{i:05d}.npy"), coefs)
        
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  -> {i + 1}/{num_samples} completado. Velocidad: {rate:.1f} muestras/seg", flush=True)
            
    total_time = time.time() - t0
    print(f"Completado en {total_time:.1f} segundos.", flush=True)

if __name__ == "__main__":
    print("=== GENERACIÓN DEL DATASET DE ENTRENAMIENTO ===", flush=True)
    # 10,000 para entrenamiento
    build_dataset(10000, TRAIN_DIR)
    # 2,000 para validación
    build_dataset(2000, VAL_DIR)
    print("=== DATASET GENERADO CON ÉXITO ===", flush=True)
