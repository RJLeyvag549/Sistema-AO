from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time
import requests
import os
import logging
import numpy as np
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from control_vectorial import ZernikeKalmanVectorial

logging.getLogger('werkzeug').setLevel(logging.ERROR)
app = Flask(__name__)
CORS(app)

# Configuraci├│n
INFERENCIA_URL = "http://inferencia:5000/predict"
SIMULADOR_URL = "http://simulador:5000/correct"

@app.route('/status')
def status():
    return jsonify({
        "status": "online", 
        "service": "Controlador AO (Physical Model)"
    })

# ===============================================================
# FILTRO DE KALMAN VECTORIAL MIMO + CONTROLADOR LQG
# ===============================================================
# Reemplaza los 10 filtros AR(1) escalares desacoplados por un
# unico filtro matricial que captura los acoplamientos entre modos
# Zernike inducidos por el viento (Ley de Taylor).

kalman_config = {
    "q_scale":      1.0,    # escala del ruido de proceso (normalizado con la física)
    "cnn_rmse":     0.05,   # RMSE tipico de la CNN en radianes (auto-actualizado segun modelo)
    "delay":        1,      # pasos de anticipacion LQG
    "wind_angle":   0.0,    # angulo del viento en radianes (0 = eje X)
    "d_r0":         1.0,    # fuerza de turbulencia estimada (para Q)
}

# Instancia unica del filtro vectorial MIMO
vectorial_filter = ZernikeKalmanVectorial(
    q_scale=kalman_config["q_scale"],
    cnn_rmse=kalman_config["cnn_rmse"],
    delay=kalman_config["delay"],
)

@app.route('/config', methods=['POST'])
def update_kalman_config():
    global kalman_config, vectorial_filter
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "No data provided"}), 400

    if 'q_scale' in data:
        kalman_config['q_scale'] = float(data['q_scale'])
    if 'cnn_rmse' in data:
        kalman_config['cnn_rmse'] = float(data['cnn_rmse'])
    if 'delay' in data:
        kalman_config['delay'] = int(data['delay'])
    if 'wind_angle' in data:
        kalman_config['wind_angle'] = float(data['wind_angle'])
    if 'd_r0' in data:
        kalman_config['d_r0'] = float(data['d_r0'])

    # Recrear el filtro vectorial con la nueva configuracion y resetear su estado
    vectorial_filter = ZernikeKalmanVectorial(
        q_scale=kalman_config['q_scale'],
        cnn_rmse=kalman_config['cnn_rmse'],
        delay=kalman_config['delay'],
    )

    return jsonify({"status": "success", "config": kalman_config})

@app.route('/process_frame', methods=['POST'])
def process_frame():
    model_name     = request.args.get('model', 'phase_diversity')
    wind_speed     = float(request.args.get('wind_speed', 0.5))
    d_r0           = float(request.args.get('d_r0', kalman_config['d_r0']))
    wind_angle_rad = float(request.args.get('wind_angle', kalman_config['wind_angle']))

    try:
        resp = requests.post(
            f"http://ao_inferencia:5000/predict?model={model_name}",
            data=request.data,
            headers={"Content-Type": "application/octet-stream"},
            timeout=1.5
        )
        if resp.status_code != 200:
            return jsonify({"status": "error", "message": f"Inference server returned status {resp.status_code}"}), resp.status_code

        result = resp.json()
        pred_list = result.get("zernike", [])
    except Exception as e:
        return jsonify({"status": "error", "message": f"Fallo al conectar con inferencia: {str(e)}"}), 500

    # Construir el vector de observacion CNN para los 10 modos (Z2..Z11), ignorando piston (Z1)
    if len(pred_list) < 11:
        return jsonify({"status": "error", "message": "La CNN devolvio menos de 11 coeficientes"}), 500

    # pred_list[0] = Z1 (piston, ignorado), pred_list[1..10] = Z2..Z11
    y_obs = np.array(pred_list[1:11], dtype=np.float64)

    # Mapear perfiles de error tipicos de la red seleccionada para el filtro Kalman
    rmse_profiles = {
        'phase_diversity': 0.12,
        'resnet10':        0.08,
        'resnet18':        0.05
    }
    current_rmse = rmse_profiles.get(model_name, 0.05)
    
    # Sincronizar dinamicamente el filtro con la configuracion y modelo activos
    vectorial_filter.cnn_rmse = current_rmse
    vectorial_filter.q_scale  = kalman_config['q_scale']
    vectorial_filter.delay    = kalman_config['delay']

    # Ejecutar un unico paso del filtro Kalman Vectorial MIMO
    x_current, x_predicted = vectorial_filter.update(
        y=y_obs,
        wind_speed=wind_speed,
        d_r0=d_r0,
        wind_angle_rad=wind_angle_rad,
    )

    # Reconstruir listas de 11 elementos (Z1=piston=0 siempre)
    cnn_zernikes     = [pred_list[0]] + pred_list[1:11]
    kalman_current   = [0.0] + x_current.tolist()
    control_zernikes = [0.0] + x_predicted.tolist()

    return jsonify({
        "cnn_zernikes":     cnn_zernikes,
        "kalman_current":   kalman_current,   # estimacion filtrada del frame actual
        "control_zernikes": control_zernikes, # prediccion LQG del proximo frame
        "uncertainty":      vectorial_filter.uncertainty,
    })


# Configuraci├│n InfluxDB
DB_URL = "http://ao_database:8086"
DB_TOKEN = os.environ.get("INFLUXDB_TOKEN", "token_provisorio_de_github")
DB_ORG = os.environ.get("INFLUXDB_ORG", "organizacion_ao")
DB_BUCKET = os.environ.get("INFLUXDB_BUCKET", "telemetria_bucket")

@app.route('/calibrate', methods=['POST'])
def calibrate():
    print("\n[CONTROLADOR] Iniciando ciclo de calibracion...", flush=True)
    try:
        # 1. Obtener prediccion de la IA
        response = requests.get("http://ao_inferencia:5000/predict")
        data = response.json()
        zernike_val = data['zernike'][0]
        
        # 2. Registrar Telemetria en InfluxDB
        with InfluxDBClient(url=DB_URL, token=DB_TOKEN, org=DB_ORG) as client:
            write_api = client.write_api(write_options=SYNCHRONOUS)
            point = Point("zernike_regression") \
                .tag("device", "holoeye_pluto") \
                .field("z0_piston", float(zernike_val)) \
                .field("wavelength_nm", 1550.0) \
                .time(time.time_ns(), WritePrecision.NS)
            write_api.write(bucket=DB_BUCKET, record=point)
        
        print(f"[CONTROLADOR] Calibracion exitosa. Z0: {zernike_val:.4f}", flush=True)
        return jsonify({"status": "success", "z0": zernike_val})
        
    except Exception as e:
        print(f"[CONTROLADOR] Fallo en ciclo de calibracion: {e}", flush=True)
        return jsonify({"status": "error", "message": str(e)}), 500

def logic_loop():
    print("--- CONTROLADOR AO: REGISTRO DE TELEMETR├ìA DE TURBULENCIA INICIADO ---", flush=True)
    while True:
        try:
            # 1. Obtener el estado actual de la turbulencia del simulador
            response = requests.get("http://ao_simulador:5000/status", timeout=2)
            if response.status_code == 200:
                state_data = response.json()
                state = state_data.get("state", {})
                zernikes = state.get("zernikes", {})
                d_r0 = state.get("d_r0", 1.0)
                method = state.get("method", "1")
                
                # 2. Registrar en InfluxDB
                with InfluxDBClient(url=DB_URL, token=DB_TOKEN, org=DB_ORG) as client:
                    write_api = client.write_api(write_options=SYNCHRONOUS)
                    
                    point = Point("turbulence_telemetry") \
                        .tag("device", "holoeye_pluto") \
                        .tag("method", method) \
                        .field("d_r0", float(d_r0))
                    
                    # A├▒adir cada uno de los 11 coeficientes a la telemetr├¡a
                    for z_key, z_val in zernikes.items():
                        point.field(z_key.lower(), float(z_val))
                        
                    point.time(time.time_ns(), WritePrecision.NS)
                    write_api.write(bucket=DB_BUCKET, record=point)
            
            # Registrar cada 500 ms (frecuencia de 2 Hz para no saturar y tener buena resoluci├│n)
            time.sleep(0.5)
        except Exception as e:
            print(f"[CONTROLADOR] Error en ciclo de registro de telemetr├¡a: {e}", flush=True)
            time.sleep(3)

if __name__ == '__main__':
    # Iniciar el ciclo de control en un hilo separado
    threading.Thread(target=logic_loop, daemon=True).start()
    
    # Iniciar servidor API
    app.run(host='0.0.0.0', port=5000, debug=False)
