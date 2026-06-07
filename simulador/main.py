from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
import numpy as np
import os
import io
import threading
import time
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
    "get_image_count": 0,
    "get_image_time": 0.0,
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

    phase_wrapped = np.mod(phase_data, 2 * np.pi)
    gray_val = np.round(phase_wrapped / (2 * np.pi) * 255).astype(np.uint8)

    img_gray = np.zeros((H, W), dtype=np.uint8)
    img_gray[pupil] = gray_val[pupil]

    rgb = np.stack([img_gray, img_gray, img_gray], axis=-1)

    rgb[edge] = [120, 120, 120]
    rgb[cy, :] = [60, 60, 60]
    rgb[:, cx] = [60, 60, 60]

    return rgb



def generate_under_development(size=PREVIEW) -> np.ndarray:
    """
    Genera una imagen informativa para frentes de onda en desarrollo/reconstrucción.
    """
    W, H = size
    img = Image.new("RGB", (W, H), (18, 18, 20))
    draw = ImageDraw.Draw(img)
    
    for offset in range(-H, W, 40):
        draw.line([offset, 0, offset + H, H], fill=(28, 28, 30), width=3)
        
    box_w, box_h = 420, 80
    bx1 = (W - box_w) // 2
    by1 = (H - box_h) // 2
    bx2 = bx1 + box_w
    by2 = by1 + box_h
    
    draw.rectangle([bx1, by1, bx2, by2], fill=(24, 24, 27), outline=(63, 63, 70), width=1)
    
    text = "FRENTE DE ONDA RECONSTRUIDO"
    subtitle = "Fase de Inferencia y Correccion por CNN"
    
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
        
    draw.text((W // 2 - 80, H // 2 - 15), text, fill=(239, 68, 68), font=font)
    draw.text((W // 2 - 140, H // 2 + 5), subtitle, fill=(161, 161, 170), font=font)
    
    return np.array(img)


def save_wavefront_npy():
    """
    Exporta la matriz actual de frente de onda a un archivo .npy en el volumen compartido.
    """
    t_start = time.perf_counter()
    
    res   = SLM_RES
    size  = (res[0] // 4, res[1] // 4)
    W, H  = size
    
    phase_data, cache = compute_phase_data(simulation_state['zernikes'], size)
    pupil = cache["pupil"]
    
    phase_clean = np.zeros((H, W), dtype=np.float32)
    phase_clean[pupil] = phase_data[pupil]

    np.save(os.path.join(SHARED_DIR, "frente_onda.npy"), phase_clean)
    
    # Perfilado
    elapsed = time.perf_counter() - t_start
    perf_metrics["save_npy_count"] += 1
    perf_metrics["save_npy_time"] += elapsed
    if perf_metrics["save_npy_count"] % 100 == 0:
        avg_ms = (perf_metrics["save_npy_time"] / 100) * 1000
        print(f"[PERF] Guardar Matrix (.npy) en Vol. Compartido (100 escrituras): Promedio {avg_ms:.3f} ms", flush=True)
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
                    
                    save_wavefront_npy()
                    
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

    save_wavefront_npy()

    return jsonify({
        "message": "Configuración SLM actualizada con éxito",
        "state":   simulation_state
    })


@app.route('/image/psf', methods=['GET'])
def get_psf_placeholder():
    t_start = time.perf_counter()
    
    rgb = generate_under_development()

    img    = Image.fromarray(rgb)
    img_io = io.BytesIO()
    img.save(img_io, 'JPEG', quality=85)
    img_io.seek(0)
    
    response = send_file(img_io, mimetype='image/jpeg')
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    
    # Perfilado
    elapsed = time.perf_counter() - t_start
    perf_metrics["get_image_count"] += 1
    perf_metrics["get_image_time"] += elapsed
    if perf_metrics["get_image_count"] % 50 == 0:
        avg_ms = (perf_metrics["get_image_time"] / 50) * 1000
        print(f"[PERF] Generar y Servir Imagen JPEG PSF (50 peticiones): Promedio {avg_ms:.3f} ms", flush=True)
        perf_metrics["get_image_time"] = 0.0
        
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


# ═══════════════════════════════════════════════════════════════
#  EJECUCIÓN DEL SERVIDOR
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
