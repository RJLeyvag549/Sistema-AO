from flask import Flask, request, jsonify, send_file
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

app = Flask(__name__)
CORS(app)

SHARED_DIR = "/app/shared"
if not os.path.exists(SHARED_DIR):
    os.makedirs(SHARED_DIR)

# ────────────────────────────────────────────────────────────────
#  Especificaciones Holoeye Pluto 2.1 – 1550 nm
#  Resolution : 1920 x 1080  |  Pixel pitch : 8.0 µm
#  Phase range: 0 – 2π  |  8-bit grayscale (256 niveles)
#  ────────────────────────────────────────────────────────────────
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


def generate_slm_phase_map(zernikes_dict: dict, size=PREVIEW) -> np.ndarray:
    """
    Genera el mapa de fase (0 a 2pi) para el SLM usando polinomios de Zernike (indexación Noll) optimizado con caché.
    """
    W, H = size
    cache = get_cached_grid_and_modes(size)
    modes = cache["modes"]
    pupil = cache["pupil"]
    edge = cache["edge"]
    cx, cy = cache["cx"], cache["cy"]

    # Extraemos coeficientes de Zernike en radianes y evaluamos
    coef_list = [zernikes_dict.get(f"Z{i}", 0.0) for i in range(1, 12)]
    
    phase_data = np.zeros((H, W), dtype=np.float32)
    for coef, mode in zip(coef_list, modes):
        if coef != 0.0:
            phase_data += coef * mode

    # Envolver la fase en el rango de modulación de fase pura [0, 2*pi]
    phase_wrapped = np.mod(phase_data, 2 * np.pi)
    gray_val = np.round(phase_wrapped / (2 * np.pi) * 255).astype(np.uint8)

    img_gray = np.zeros((H, W), dtype=np.uint8)
    img_gray[pupil] = gray_val[pupil]

    rgb = np.stack([img_gray, img_gray, img_gray], axis=-1)

    # Borde y retículo para la visualización del SLM
    rgb[edge] = [120, 120, 120]
    rgb[cy, :] = [60, 60, 60]
    rgb[:, cx] = [60, 60, 60]

    return rgb


# def generate_psf(zernikes_dict: dict, size=PREVIEW) -> np.ndarray:
#     """
#     Genera la Función de Punto Expandida (PSF) para el sistema AO
#     calculando la difracción física real de Fourier de la pupila
#     aberrada mediante la API funcional de Prysm (v0.21+).
#     """
#     W, H = size
#     samples = 256
#     
#     # Grilla de cálculo óptico físico (coordenadas de pupila normalizadas)
#     y, x = np.mgrid[-1:1:samples*1j, -1:1:samples*1j]
#     r = np.sqrt(x**2 + y**2)
#     theta = np.arctan2(y, x)
#     
#     pupil_mask = r <= 0.92
#     
#     # Generar frente de onda de fase
#     coef_list = [zernikes_dict.get(f"Z{i}", 0.0) for i in range(1, 12)]
#     nms = [noll_to_nm(i) for i in range(1, 12)]
#     
#     r_norm = r / 0.92
#     modes = list(zernike_nm_sequence(nms, r_norm, theta, norm=True))
#     
#     phase = np.zeros_like(r)
#     for coef, mode in zip(coef_list, modes):
#         phase += coef * mode
#     
#     # Crear campo complejo de pupila (A * exp(i * phase))
#     amplitude = np.zeros_like(r, dtype=np.float32)
#     amplitude[pupil_mask] = 1.0
#     
#     wavefunction = amplitude * np.exp(1j * phase)
#     
#     # Propagar al plano focal usando transformada de Fourier rápida
#     # Q=2 provee excelente sobremuestreo para difracción
#     psf_field = focus(wavefunction, Q=2)
#     intensity = np.abs(psf_field) ** 2
#     
#     # Redimensionar la intensidad resultante al tamaño del viewport
#     scale_y = H / intensity.shape[0]
#     scale_x = W / intensity.shape[1]
#     display = zoom(intensity, (scale_y, scale_x), order=1)
#     
#     # Normalizar
#     display = display / (display.max() + 1e-9)
# 
#     # Escala logarítmica para realzar los anillos de difracción
#     display_log = np.log1p(display * 150) / np.log1p(150)
# 
#     # Añadir un speckle sutil de fondo IR para realismo
#     np.random.seed(99)
#     noise = np.abs(np.random.normal(0, 0.012, (H, W)))
#     display_log = np.clip(display_log + noise, 0, 1)
# 
#     # Falso color térmico IR (InGaAs / Cámara térmica)
#     rgb = np.zeros((H, W, 3), dtype=np.float32)
#     rgb[..., 0] = np.clip(display_log * 2.5 - 0.5, 0, 1)  # R
#     rgb[..., 1] = np.clip(display_log * 1.8 - 0.3, 0, 1)  # G
#     rgb[..., 2] = np.clip(display_log * 0.9 + 0.1, 0, 1)  # B
# 
#     rgb = (rgb * 255).astype(np.uint8)
# 
#     # Halo exterior tenue (reflexiones y luz dispersa)
#     y_g, x_g = np.mgrid[-(H//2):(H//2), -(W//2):(W//2)]
#     r_norm_g = np.sqrt(x_g**2 + y_g**2) / (min(W, H) // 2)
#     halo = np.exp(-r_norm_g**2 * 4) * 6
#     rgb = np.clip(rgb.astype(int) + halo[..., np.newaxis].astype(int), 0, 255).astype(np.uint8)
# 
#     return rgb


def save_wavefront_npy():
    res   = SLM_RES
    size  = (res[0] // 4, res[1] // 4)
    W, H  = size
    
    cache = get_cached_grid_and_modes(size)
    modes = cache["modes"]
    pupil = cache["pupil"]
    
    coef_list = [simulation_state['zernikes'].get(f"Z{i}", 0.0) for i in range(1, 12)]
    
    phase_data = np.zeros((H, W), dtype=np.float32)
    for coef, mode in zip(coef_list, modes):
        if coef != 0.0:
            phase_data += coef * mode

    # Filtrar con la máscara de la pupila
    phase_clean = np.zeros((H, W), dtype=np.float32)
    phase_clean[pupil] = phase_data[pupil]

    np.save(os.path.join(SHARED_DIR, "frente_onda.npy"), phase_clean)


def update_stochastic_turbulence_loop():
    global simulation_state
    while True:
        try:
            if simulation_state.get("method") == "2":
                d_r0 = simulation_state.get("d_r0", 1.0)
                wind_speed = simulation_state.get("wind_speed", 0.5)
                
                # Coeficiente de correlación temporal AR(1)
                # wind_speed = 0.0 -> alpha = 1.0 (turbulencia congelada)
                # wind_speed = 1.0 -> alpha = 0.70 (turbulencia de evolución rápida)
                alpha = 1.0 - (wind_speed * 0.30)
                alpha = max(0.5, min(1.0, alpha))
                
                if alpha < 1.0:
                    beta_coeff = np.sqrt(1.0 - alpha**2)
                    # La varianza escala con (D/r0)^(5/3) según Kolmogorov
                    factor_kolmogorov = (d_r0) ** (5.0 / 3.0)
                    
                    for k, base_var in NOLL_VARIANCES.items():
                        # Piston Z1 se puede dejar en 0 o darle una pequeña fluctuación controlada
                        if k == "Z1":
                            continue
                        sigma = np.sqrt(base_var * factor_kolmogorov)
                        current_val = simulation_state["zernikes"].get(k, 0.0)
                        noise = np.random.normal(0, 1)
                        new_val = alpha * current_val + sigma * beta_coeff * noise
                        simulation_state["zernikes"][k] = float(new_val)
                    
                    save_wavefront_npy()
        except Exception as e:
            print(f"Error en bucle estocástico: {e}")
        time.sleep(0.06)  # ~16 FPS

# Iniciar hilo de simulación de turbulencia en segundo plano
threading.Thread(target=update_stochastic_turbulence_loop, daemon=True).start()


# ═══════════════════════════════════════════════════════════════
#  RUTAS DE LA API
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
        
    # Procesar coeficientes Zernike
    if 'zernikes' in data:
        for k, v in data['zernikes'].items():
            if k in simulation_state['zernikes']:
                simulation_state['zernikes'][k] = float(v)

    # Re-generar y guardar mapa de fase actual
    save_wavefront_npy()

    return jsonify({
        "message": "Configuración SLM actualizada con éxito",
        "state":   simulation_state
    })


def generate_under_development(size=PREVIEW) -> np.ndarray:
    W, H = size
    # Imagen oscura estilo industrial / laboratorio
    img = Image.new("RGB", (W, H), (18, 18, 20))
    draw = ImageDraw.Draw(img)
    
    # Líneas de fondo tipo grilla industrial
    for offset in range(-H, W, 40):
        draw.line([offset, 0, offset + H, H], fill=(28, 28, 30), width=3)
        
    # Caja contenedora central
    box_w, box_h = 420, 80
    bx1 = (W - box_w) // 2
    by1 = (H - box_h) // 2
    bx2 = bx1 + box_w
    by2 = by1 + box_h
    
    draw.rectangle([bx1, by1, bx2, by2], fill=(24, 24, 27), outline=(63, 63, 70), width=1)
    
    # Textos descriptivos
    text = "FRENTE DE ONDA RECONSTRUIDO"
    subtitle = "Fase de Inferencia y Correccion por CNN"
    
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
        
    # Dibujar textos
    draw.text((W // 2 - 80, H // 2 - 15), text, fill=(239, 68, 68), font=font)
    draw.text((W // 2 - 140, H // 2 + 5), subtitle, fill=(161, 161, 170), font=font)
    
    return np.array(img)


@app.route('/image/<img_type>', methods=['GET'])
def get_image(img_type):
    zernikes = simulation_state['zernikes']

    if img_type == 'distorted':
        # Mapa de fase del SLM (lo que se muestra en el panel del dispositivo)
        rgb = generate_slm_phase_map(zernikes)
    else:
        # En desarrollo para la fase de reconstrucción
        rgb = generate_under_development()

    img    = Image.fromarray(rgb)
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    
    response = send_file(img_io, mimetype='image/png')
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
