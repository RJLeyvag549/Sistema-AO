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
    "turbulencia": 0.5,
    "viento": 10.0,
    "humedad": 50.0,
    "wavelength_nm": 1550.0,
    "resolution": (1920, 1080),
    "pixel_pitch_um": 8.0
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
    
    # Generar frente de onda compatible con la resolución del SLM
    res = simulation_state['resolution']
    wavefront = np.random.randn(res[1]//4, res[0]//4).astype(np.float32) * simulation_state['turbulencia']
    np.save(os.path.join(SHARED_DIR, "frente_onda.npy"), wavefront)
    
    return jsonify({"message": "Configuracion SLM actualizada", "state": simulation_state})

@app.route('/image/<type>', methods=['GET'])
def get_image(type):
    global simulation_state
    # Usamos una resolución reducida para la vista previa web, pero basada en 16:9
    display_res = (640, 360) 
    
    if type == 'distorted':
        # Simular Mapa de Fase para el SLM Pluto
        noise_level = int(simulation_state['turbulencia'] * 50)
        data = np.zeros((display_res[1], display_res[0]), dtype=np.uint8)
        
        # Generar patrones de interferencia de fase (falsos colores térmicos)
        y, x = np.ogrid[:display_res[1], :display_res[0]]
        for i in range(5):
            cx, cy = np.random.randint(0, display_res[0]), np.random.randint(0, display_res[1])
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            data = (data + (np.sin(dist / (10 + noise_level)) * 127 + 128)).astype(np.uint8)
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
