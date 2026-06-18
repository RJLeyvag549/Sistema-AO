from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
import numpy as np
import os
import io
import threading
import time
import requests
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import zoom

# Importar Prysm para la física óptica (API funcional moderna de v0.21+)
from prysm.polynomials import noll_to_nm, zernike_nm_sequence
from prysm.propagation import focus

# ═══════════════════════════════════════════════════════════════
#  CONFIGURACIÓN Y CONSTANTES
# ═══════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)

SHARED_DIR = "/app/shared"
if not os.path.exists(SHARED_DIR):
    os.makedirs(SHARED_DIR)

# Especificaciones Holoeye Pluto 2.1 – 1550 nm
SLM_RES   = (1920, 1080)   # (width, height)
PIXEL_UM  = 8.0            # µm por pixel
LAMBDA_NM = 1550.0         # longitud de onda en nm
PREVIEW   = (640, 360)     # resolución de vista previa

NOLL_VARIANCES = {
    "Z1": 0.0,       # Piston (Usualmente 0 en corrección)
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

# ═══════════════════════════════════════════════════════════════
#  ESTADO GLOBAL DE LA SIMULACIÓN Y METRICAS DE RENDIMIENTO
# ═══════════════════════════════════════════════════════════════

simulation_state = {
    "method": "1",      # "1" = Manual, "2" = Estocástica por modos
    "d_r0": 1.0,        # Relación D/r0 (Fuerza de turbulencia)
    "wind_speed": 0.5,  # Velocidad de evolución temporal (0 a 1)
    "zernikes": {
        "Z1": 0.0,
        "Z2": 0.0,
        "Z3": 0.0,
        "Z4": 0.0,
        "Z5": 0.0,
        "Z6": 0.0,
        "Z7": 0.0,
        "Z8": 0.0,
        "Z9": 0.0,
        "Z10": 0.0,
        "Z11": 0.0,
    }
}

CACHED_DATA = {}

# Métricas para perfilar cuellos de botella de hardware
perf_metrics = {
    "stochastic_loop_count": 0,
    "stochastic_loop_time": 0.0,
    "save_npy_count": 0,
    "save_npy_time": 0.0,
}

# ═══════════════════════════════════════════════════════════════
#  FUNCIONES DE APOYO ÓPTICO Y MATEMÁTICAS
# ═══════════════════════════════════════════════════════════════

def get_cached_grid_and_modes(size):
    if size not in CACHED_DATA:
        W, H = size
        y, x = np.mgrid[-1:1:H*1j, -1:1:W*1j]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        pupil = r <= 0.92
        edge = (r > 0.90) & (r <= 0.92)
        
        r_norm = r / 0.92
        nms = [noll_to_nm(i) for i in range(1, 12)]
        modes = list(zernike_nm_sequence(nms, r_norm, theta, norm=True))
        
        CACHED_DATA[size] = {
            "modes": modes,
            "pupil": pupil,
            "edge": edge,
            "cx": W // 2,
            "cy": H // 2
        }
    return CACHED_DATA[size]


def compute_phase_data(zernikes_dict: dict, size) -> tuple:
    cache = get_cached_grid_and_modes(size)
    modes = cache["modes"]
    coef_list = [zernikes_dict.get(f"Z{i}", 0.0) for i in range(1, 12)]
    
    phase_data = np.zeros((size[1], size[0]), dtype=np.float32)
    for coef, mode in zip(coef_list, modes):
        if coef != 0.0:
            phase_data += coef * mode
            
    return phase_data, cache


def generate_slm_phase_map(zernikes_dict: dict, size=PREVIEW) -> np.ndarray:
    W, H = size
    phase_data, cache = compute_phase_data(zernikes_dict, size)
    pupil = cache["pupil"]
    edge = cache["edge"]
    cx, cy = cache["cx"], cache["cy"]

    # Corrección perfecta es la fase opuesta conjugada
    correction_phase = -phase_data
    phase_wrapped = np.mod(correction_phase, 2 * np.pi)
    gray_val = np.round(phase_wrapped / (2 * np.pi) * 255).astype(np.uint8)

    img_gray = np.zeros((H, W), dtype=np.uint8)
    img_gray[pupil] = gray_val[pupil]

    rgb = np.stack([img_gray, img_gray, img_gray], axis=-1)

    rgb[edge] = [120, 120, 120]
    rgb[cy, :] = [60, 60, 60]
    rgb[:, cx] = [60, 60, 60]

    return rgb


def generate_psf_image(zernikes_dict: dict, size=PREVIEW) -> np.ndarray:
    """
    Genera la PSF (Point Spread Function) real con aberración usando propagación física de Prysm.
    """
    prop_size = (256, 256)
    phase_data, cache = compute_phase_data(zernikes_dict, prop_size)
    pupil = cache["pupil"]
    
    # Wavefunction en el plano de la pupila: amplitud * exp(1j * fase)
    # La amplitud es 1 en la pupila y 0 fuera
    wf = np.exp(1j * phase_data) * pupil
    
    # Propagación al foco con factor de sobremuestreo Q=2
    focal_wf = focus(wf, Q=2)
    
    # Intensidad de la PSF (módulo al cuadrado)
    psf = np.abs(focal_wf) ** 2
    
    # Recortar el centro para hacer zoom sobre el foco
    H, W = psf.shape
    cy, cx = H // 2, W // 2
    crop_half = 48  # Recorte de 96x96 píxeles para excelente visibilidad del spot
    psf_crop = psf[cy - crop_half:cy + crop_half, cx - crop_half:cx + crop_half]
    
    # Normalizar
    psf_max = np.max(psf_crop)
    if psf_max > 0:
        psf_norm = psf_crop / psf_max
    else:
        psf_norm = psf_crop
        
    # Escalar con una potencia (gamma 0.4) para ver los anillos de Airy secundarios
    psf_scaled = np.power(psf_norm, 0.4)
    
    # Mapear a un colormap tipo 'calor/láser' (Rojo -> Naranja -> Blanco)
    r = (psf_scaled * 255).astype(np.uint8)
    g = (np.power(psf_scaled, 2.5) * 255).astype(np.uint8)
    b = (np.power(psf_scaled, 5.0) * 255).astype(np.uint8)
    
    rgb = np.stack([r, g, b], axis=-1)
    
    img = Image.fromarray(rgb)
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    img_resized = img.resize(size, resample)
    
    return np.array(img_resized)




def save_psf_npy():
    """
    Calcula la PSF con aberración actual y la exporta como matriz 96x96 .npy al volumen compartido.
    """
    t_start = time.perf_counter()
    
    prop_size = (256, 256)
    phase_data, cache = compute_phase_data(simulation_state['zernikes'], prop_size)
    pupil = cache["pupil"]
    
    # Propagación física a PSF
    wf = np.exp(1j * phase_data) * pupil
    focal_wf = focus(wf, Q=2)
    psf = np.abs(focal_wf) ** 2
    
    # Recorte central 96x96
    H, W = psf.shape
    cy, cx = H // 2, W // 2
    crop_half = 48
    psf_crop = psf[cy - crop_half:cy + crop_half, cx - crop_half:cx + crop_half]
    
    # Normalización (0.0 a 1.0)
    psf_max = np.max(psf_crop)
    if psf_max > 0:
        psf_norm = psf_crop / psf_max
    else:
        psf_norm = psf_crop

    np.save(os.path.join(SHARED_DIR, "psf.npy"), psf_norm.astype(np.float32))
    
    # Perfilado
    elapsed = time.perf_counter() - t_start
    perf_metrics["save_npy_count"] += 1
    perf_metrics["save_npy_time"] += elapsed
    if perf_metrics["save_npy_count"] % 100 == 0:
        avg_ms = (perf_metrics["save_npy_time"] / 100) * 1000
        print(f"[PERF] Guardar PSF (.npy) en Vol. Compartido (100 escrituras): Promedio {avg_ms:.3f} ms", flush=True)
        perf_metrics["save_npy_time"] = 0.0


# ═══════════════════════════════════════════════════════════════
#  HILO DE SIMULACIÓN EN SEGUNDO PLANO
# ═══════════════════════════════════════════════════════════════

def update_stochastic_turbulence_loop():
    global simulation_state
    while True:
        try:
            if simulation_state.get("method") == "2":
                t_start = time.perf_counter()
                
                d_r0 = simulation_state.get("d_r0", 1.0)
                wind_speed = simulation_state.get("wind_speed", 0.5)
                
                alpha = 1.0 - (wind_speed * 0.30)
                alpha = max(0.5, min(1.0, alpha))
                
                if alpha < 1.0:
                    beta_coeff = np.sqrt(1.0 - alpha**2)
                    factor_kolmogorov = (d_r0) ** (5.0 / 3.0)
                    
                    for k, base_var in NOLL_VARIANCES.items():
                        if k == "Z1":
                            continue
                        sigma = np.sqrt(base_var * factor_kolmogorov)
                        current_val = simulation_state["zernikes"].get(k, 0.0)
                        noise = np.random.normal(0, 1)
                        new_val = alpha * current_val + sigma * beta_coeff * noise
                        simulation_state["zernikes"][k] = float(new_val)
                    
                    save_psf_npy()
                    
                    # Perfilado
                    elapsed = time.perf_counter() - t_start
                    perf_metrics["stochastic_loop_count"] += 1
                    perf_metrics["stochastic_loop_time"] += elapsed
                    if perf_metrics["stochastic_loop_count"] % 100 == 0:
                        avg_ms = (perf_metrics["stochastic_loop_time"] / 100) * 1000
                        print(f"[PERF] Bucle de Turbulencia Estocástica sin sleep (100 iteraciones): Promedio {avg_ms:.3f} ms", flush=True)
                        perf_metrics["stochastic_loop_time"] = 0.0
                        
        except Exception as e:
            print(f"Error en bucle estocástico: {e}")
        time.sleep(0.016)

threading.Thread(target=update_stochastic_turbulence_loop, daemon=True).start()


# ═══════════════════════════════════════════════════════════════
#  RUTAS DE LA API REST
# ═══════════════════════════════════════════════════════════════

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "status":  "online",
        "service": "Simulador de Física Óptica Prysm – 1550 nm",
        "device":  "Holoeye PLUTO 2.1 (TELCO-016) (Simulado)",
        "spec": {
            "resolution":  "1920 x 1080",
            "pixel_pitch": "8.0 µm",
            "wavelength":  "1550 nm",
            "phase_levels": 256,
            "zernike_active": "Z1 a Z11 (Noll Indexing)"
        },
        "state": simulation_state
    })


@app.route('/config', methods=['POST'])
def update_config():
    global simulation_state
    data = request.json
    
    if 'method' in data:
        simulation_state['method'] = str(data['method'])
    if 'd_r0' in data:
        simulation_state['d_r0'] = float(data['d_r0'])
    if 'wind_speed' in data:
        simulation_state['wind_speed'] = float(data['wind_speed'])
        
    if 'zernikes' in data:
        for k, v in data['zernikes'].items():
            if k in simulation_state['zernikes']:
                simulation_state['zernikes'][k] = float(v)

    save_psf_npy()

    return jsonify({
        "message": "Configuración SLM actualizada con éxito",
        "state":   simulation_state
    })



@app.route('/image/psf-raw', methods=['GET'])
def get_psf_raw():
    """
    Devuelve la PSF con turbulencia (aberrada real) como bytes RGB crudos (sin compresión).
    Formato: (H x W x 3) uint8 en orden row-major, 3 bytes por píxel (R, G, B).
    """
    rgb = generate_psf_image(simulation_state['zernikes'])
    rgb_bytes = rgb.tobytes()
    response = make_response(rgb_bytes)
    response.headers['Content-Type']  = 'application/octet-stream'
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    response.headers['Pragma']        = 'no-cache'
    response.headers['Expires']       = '0'
    return response



@app.route('/image/distorted-raw', methods=['GET'])
def get_image_raw():
    """
    Devuelve el mapa de fase SLM como bytes grises crudos (sin codificación JPEG).
    Formato: array de uint8 en orden row-major (H x W), 1 byte por píxel.
    El cliente puede dibujarlo directamente en un canvas HTML5 sin decodificación.
    """
    t_start = time.perf_counter()

    rgb = generate_slm_phase_map(simulation_state['zernikes'])
    # rgb es (H, W, 3) uint8 en escala de grises (R=G=B); extraemos solo el canal R
    gray_bytes = rgb[:, :, 0].tobytes()

    response = make_response(gray_bytes)
    response.headers['Content-Type']  = 'application/octet-stream'
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    response.headers['Pragma']        = 'no-cache'
    response.headers['Expires']       = '0'

    elapsed = time.perf_counter() - t_start
    print(f"[PERF] Raw bytes ({len(gray_bytes)} B): {elapsed*1000:.2f} ms", flush=True) if elapsed > 0.005 else None

    return response


@app.route('/image/cnn-phase-raw', methods=['GET'])
def get_cnn_phase_raw():
    """
    Devuelve el mapa de fase de corrección estimado por la CNN como bytes grises crudos.
    Formato idéntico a /image/distorted-raw: (H x W) uint8, 1 byte por píxel.
    El canvas de la interfaz lo pinta directamente con putImageData (GPU integrada del navegador).
    """
    accuracy = 100.0
    try:
        resp = requests.get("http://ao_inferencia:5000/predict", timeout=1.0)
        if resp.status_code == 200:
            pred_list = resp.json().get("zernike", [])
            cnn_zernikes = {f"Z{i}": pred_list[i-1] for i in range(1, min(12, len(pred_list) + 1))}
            for idx in range(1, 12):
                cnn_zernikes.setdefault(f"Z{idx}", 0.0)
            rgb = generate_slm_phase_map(cnn_zernikes)
            
            # Calcular precisión
            actuals = np.array([simulation_state['zernikes'].get(f"Z{i}", 0.0) for i in range(1, 12)])
            preds = np.array([cnn_zernikes.get(f"Z{i}", 0.0) for i in range(1, 12)])
            mae = np.mean(np.abs(actuals - preds))
            accuracy = float(100.0 * np.exp(-mae))
        else:
            rgb = generate_slm_phase_map({f"Z{i}": 0.0 for i in range(1, 12)})
    except Exception:
        rgb = generate_slm_phase_map({f"Z{i}": 0.0 for i in range(1, 12)})

    gray_bytes = rgb[:, :, 0].tobytes()
    response = make_response(gray_bytes)
    response.headers['Content-Type']  = 'application/octet-stream'
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    response.headers['Pragma']        = 'no-cache'
    response.headers['Expires']       = '0'
    response.headers['Access-Control-Expose-Headers'] = 'X-CNN-Accuracy'
    response.headers['X-CNN-Accuracy'] = f"{accuracy:.2f}"
    return response


@app.route('/image/reconstructed-psf-raw', methods=['GET'])
def get_reconstructed_psf_raw():
    """
    Devuelve la PSF corregida (aberración real - predicción CNN) como bytes RGB crudos.
    Formato: (H x W x 3) uint8 en orden row-major, 3 bytes por píxel (R, G, B).
    El canvas de la interfaz lo reconstruye a RGBA y lo pinta con putImageData.
    """
    accuracy = 100.0
    try:
        resp = requests.get("http://ao_inferencia:5000/predict", timeout=1.0)
        if resp.status_code == 200:
            pred_list = resp.json().get("zernike", [])
            cnn_zernikes = {f"Z{i}": pred_list[i-1] for i in range(1, min(12, len(pred_list) + 1))}
            residual = {}
            for idx in range(1, 12):
                key = f"Z{idx}"
                actual = simulation_state['zernikes'].get(key, 0.0)
                pred   = cnn_zernikes.get(key, 0.0)
                residual[key] = actual - pred
            rgb = generate_psf_image(residual)
            
            # Calcular precisión
            actuals = np.array([simulation_state['zernikes'].get(f"Z{i}", 0.0) for i in range(1, 12)])
            preds = np.array([cnn_zernikes.get(f"Z{i}", 0.0) for i in range(1, 12)])
            mae = np.mean(np.abs(actuals - preds))
            accuracy = float(100.0 * np.exp(-mae))
        else:
            rgb = generate_psf_image({f"Z{i}": 0.0 for i in range(1, 12)})
    except Exception:
        rgb = generate_psf_image({f"Z{i}": 0.0 for i in range(1, 12)})

    rgb_bytes = rgb.tobytes()   # (H, W, 3) → H*W*3 bytes en orden R,G,B por píxel
    response = make_response(rgb_bytes)
    response.headers['Content-Type']  = 'application/octet-stream'
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    response.headers['Pragma']        = 'no-cache'
    response.headers['Expires']       = '0'
    response.headers['Access-Control-Expose-Headers'] = 'X-CNN-Accuracy'
    response.headers['X-CNN-Accuracy'] = f"{accuracy:.2f}"
    return response


# ═══════════════════════════════════════════════════════════════
#  EJECUCIÓN DEL SERVIDOR
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
