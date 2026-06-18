from flask import Flask, jsonify
import numpy as np
import torch
import os
import time

from model import BaselineCNN
from dataset import denormalize_predictions

app = Flask(__name__)

SHARED_DIR = "/app/shared"
MODEL_PATH = os.path.join(SHARED_DIR, "custom_cnn.pth")
PSF_PATH = os.path.join(SHARED_DIR, "psf.npy")

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFERENCIA] Usando dispositivo: {device}")

# Cargar modelo de forma global al iniciar
model = BaselineCNN()
model_loaded = False

if os.path.exists(MODEL_PATH):
    try:
        # Cargar pesos en el dispositivo adecuado
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        model_loaded = True
        print(f"[INFERENCIA] Modelo cargado con éxito desde {MODEL_PATH}")
    except Exception as e:
        print(f"[INFERENCIA] Error al cargar el modelo: {e}")
else:
    print(f"[INFERENCIA] ADVERTENCIA: Pesos del modelo no encontrados en {MODEL_PATH}. Corriendo con pesos aleatorios.")
    model.to(device)
    model.eval()

@app.route('/status')
def status():
    return jsonify({
        "status": "online", 
        "service": "Inferencia CNN (PyTorch)",
        "shared_volume": os.path.exists(SHARED_DIR),
        "model_loaded": model_loaded,
        "device": str(device)
    })

@app.route('/predict', methods=['GET'])
def predict():
    # Intentar leer la PSF actual del volumen compartido
    if os.path.exists(PSF_PATH):
        try:
            # 1. Cargar matriz de PSF
            psf_data = np.load(PSF_PATH).astype(np.float32) # Debería ser 96x96
            
            # Verificar dimensiones correctas
            if psf_data.shape != (96, 96):
                # Si el simulador guardó otra resolución, intentamos redimensionar o recortar
                print(f"[INFERENCIA] Advertencia: Tamaño de PSF incorrecto {psf_data.shape}. Requerido: (96, 96).")
                return jsonify({"error": f"Dimensiones incorrectas {psf_data.shape}"}), 400
                
            # 2. Preprocesar para PyTorch: (96, 96) -> (1, 1, 96, 96)
            psf_tensor = torch.from_numpy(psf_data).unsqueeze(0).unsqueeze(0).to(device)
            
            # 3. Ejecutar inferencia
            with torch.no_grad():
                pred_norm = model(psf_tensor)
                # Denormalizar los coeficientes para recuperar sus escalas físicas
                pred_physical = denormalize_predictions(pred_norm)
                
            # Convertir a lista de floats estándar de Python
            zernike_predictions = pred_physical.squeeze(0).cpu().numpy().tolist()
            
            return jsonify({
                "zernike": zernike_predictions,
                "timestamp": time.time(),
                "source": "pytorch_cnn"
            })
            
        except Exception as e:
            print(f"[INFERENCIA] Error durante la inferencia: {e}")
            return jsonify({"error": f"Fallo en inferencia: {str(e)}"}), 500
    else:
        # Fallback si el archivo psf.npy no existe
        return jsonify({
            "error": "psf.npy no encontrado en el volumen compartido",
            "timestamp": time.time()
        }), 404

if __name__ == '__main__':
    print("--- INFERENCIA CNN: SERVIDOR ACTIVO ---")
    app.run(host='0.0.0.0', port=5000, debug=True)
