from flask import Flask, jsonify
import numpy as np
import threading
import time
import os

app = Flask(__name__)

SHARED_DIR = "/app/shared"

@app.route('/status')
def status():
    return jsonify({
        "status": "online", 
        "service": "Inferencia CNN",
        "shared_volume": os.path.exists(SHARED_DIR)
    })

@app.route('/predict', methods=['GET'])
def predict():
    # Simular la inferencia de la CNN
    # En un caso real, leería la imagen del SHARED_DIR
    zernike = np.random.uniform(-1, 1, 36).tolist()
    return jsonify({
        "zernike": zernike,
        "timestamp": time.time()
    })

if __name__ == '__main__':
    print("--- INFERENCIA CNN: MODELO CARGADO ---")
    app.run(host='0.0.0.0', port=5000, debug=True)
