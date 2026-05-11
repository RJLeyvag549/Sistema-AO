from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time
import requests
import os

app = Flask(__name__)
CORS(app)

# Configuración
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

# Configuración InfluxDB
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
    print("--- CONTROLADOR AO: CICLO DE CORRECCION INICIADO ---")
    while True:
        try:
            # 1. El controlador podría estar esperando una señal o monitoreando el volumen
            # Por ahora, simulamos el flujo cada 2 segundos
            
            # 2. Pedir predicción a la Inferencia
            # (En un caso real, enviaríamos el estado actual del frente de onda)
            # response = requests.get(INFERENCIA_URL)
            # zernike = response.json().get('zernike')
            
            # 3. Aplicar Modelo Físico y enviar corrección al Simulador
            # print(f"Aplicando corrección basada en Zernike")
            
            time.sleep(2)
        except Exception as e:
            print(f" Error en ciclo de control: {e}")
            time.sleep(5)

if __name__ == '__main__':
    # Iniciar el ciclo de control en un hilo separado
    threading.Thread(target=logic_loop, daemon=True).start()
    
    # Iniciar servidor API
    app.run(host='0.0.0.0', port=5000, debug=True)
