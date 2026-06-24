from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time
import requests
import os
import logging

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

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

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
