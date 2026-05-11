from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import os
import io
from PIL import Image

app = Flask(__name__)
CORS(app)  # Permitir que React se comunique con Flask

SHARED_DIR = "/app/shared"
if not os.path.exists(SHARED_DIR):
    os.makedirs(SHARED_DIR)

# Estado de la simulación - Especificaciones Holoeye Pluto
simulation_state = {
    "wavelength_nm": 1550.0,
    "resolution": (1920, 1080),
    "pixel_pitch_um": 8.0,
    "piston": 0.0
}

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "status": "online", 
        "service": "Simulador Holoeye Pluto 1550nm",
        "device": "SLM-PLUTO-VIS-016"
    })

@app.route('/config', methods=['POST'])
def update_config():
    global simulation_state
    data = request.json
    simulation_state.update(data)
    
    # Generar máscara de fase compatible con la resolución nativa del SLM (1920x1080)
    width, height = simulation_state['resolution']
    
    # Crear malla de coordenadas espaciales con origen en el centro
    radius_pixels = height // 2  # El radio máximo para que quepa en la pantalla (540 px)
    y, x = np.ogrid[-height//2 : height//2, -width//2 : width//2]
    
    # Definir la máscara circular (Pupila unitaria r <= 1)
    r2 = (x**2 + y**2) / (radius_pixels**2)
    pupil_mask = r2 <= 1.0
    
    # 1. Fase Continua (Inicializada en ceros para toda la matriz)
    fase_continua = np.zeros((height, width), dtype=np.float32)
    
    # -- MODOS DE ZERNIKE --
    # Z0: Piston (Valor constante)
    piston_val = simulation_state.get('piston', 0.0)
    
    # Aplicamos la fase generada *solo* dentro de la pupila
    fase_continua[pupil_mask] += piston_val * np.pi
    
    # 2. Envolver (wrap) la fase entre 0 y 2π
    fase_envuelta = np.mod(fase_continua, 2 * np.pi)
    
    # 3. Mapear de [0, 2π] a escala de grises [0, 255]
    # Invertimos la escala: Fase 0 será 255 (Blanco) y Fase 2π será 0 (Negro).
    # Fuera de la pupila mantenemos fase 0 (Blanco) para que todo el espejo actúe plano.
    fase_slm_8bit = np.full((height, width), 255, dtype=np.uint8)
    fase_slm_8bit[pupil_mask] = 255 - (fase_envuelta[pupil_mask] / (2 * np.pi) * 255).astype(np.uint8)
    
    # Guardar como .npy para la inferencia y .png para validación directa
    np.save(os.path.join(SHARED_DIR, "frente_onda.npy"), fase_slm_8bit)
    Image.fromarray(fase_slm_8bit).save(os.path.join(SHARED_DIR, "fase_slm.png"))
    
    return jsonify({"message": "Configuracion SLM actualizada", "state": simulation_state})

@app.route('/image/<type>', methods=['GET'])
def get_image(type):
    global simulation_state
    # Usamos una resolución reducida para la vista previa web, pero basada en 16:9
    display_res = (640, 360) 
    
    if type == 'distorted':
        # Enviar la máscara de fase REAL generada por el simulador (si existe)
        img_path = os.path.join(SHARED_DIR, "fase_slm.png")
        if os.path.exists(img_path):
            img = Image.open(img_path)
            # Redimensionar para la interfaz web a 640x360 para no saturar la red local
            img = img.resize((display_res[0], display_res[1]), Image.LANCZOS)
            data = np.array(img)
        else:
            # Fallback en caso de que aún no se haya configurado el sistema
            data = np.zeros((display_res[1], display_res[0]), dtype=np.uint8)
    else:
        # IMAGEN RECONSTRUIDA: Punto focal ideal a 1550nm
        data = np.zeros((display_res[1], display_res[0]), dtype=np.uint8)
        center_x, center_y = display_res[0]//2, display_res[1]//2
        y, x = np.ogrid[:display_res[1], :display_res[0]]
        mask = (x - center_x)**2 + (y - center_y)**2 <= 8**2
        data[mask] = 255
        # Añadir halo de difracción
        halo = (x - center_x)**2 + (y - center_y)**2 <= 20**2
        data[halo & ~mask] = 40
        
    img = Image.fromarray(data)
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
