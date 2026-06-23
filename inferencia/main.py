from flask import Flask, jsonify, request
import numpy as np
import torch
import os
import time

from model import BaselineCNN
from model_resnet import ResNet10
from dataset import denormalize_predictions
# from jit_resnet import load_resnet10_for_inference

app = Flask(__name__)

SHARED_DIR = "/app/shared"
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PSF_PATH = os.path.join(SHARED_DIR, "psf.npy")

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFERENCIA] Usando dispositivo: {device}")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

# Diccionario para almacenar los modelos cargados
models = {}
models_loaded = {
    "phase_diversity": False,
    "resnet10": False
}
resnet_jit_meta = {}
resnet_uses_jit = False


def _load_resnet10_eager(resnet_path: str):
    """Fallback: modelo eager si TorchScript falla."""
    global resnet_jit_meta, resnet_uses_jit
    model = ResNet10(in_channels=2).to(device)
    if os.path.exists(resnet_path):
        model.load_state_dict(
            torch.load(resnet_path, map_location=device, weights_only=True)
        )
        models_loaded["resnet10"] = True
    else:
        models_loaded["resnet10"] = False
    model.eval()
    models["resnet10"] = model
    resnet_uses_jit = False
    resnet_jit_meta = {"backend": "eager", "weights_path": resnet_path}


def _load_resnet10():
    resnet_path = os.path.join(MODEL_DIR, "resnet10_cnn.pth")
    _load_resnet10_eager(resnet_path)


# 0. Cargar Modelo ResNet-10 (2 Canales) directamente desde el .pth
_load_resnet10()

# 1. Cargar Modelo Phase Diversity (2 Canales)
models["phase_diversity"] = BaselineCNN(in_channels=2)
pd_path = os.path.join(MODEL_DIR, "phase_diversity_cnn.pth")
if os.path.exists(pd_path):
    try:
        models["phase_diversity"].load_state_dict(torch.load(pd_path, map_location=device))
        models["phase_diversity"].to(device)
        models["phase_diversity"].eval()
        models_loaded["phase_diversity"] = True
        print(f"[INFERENCIA] Modelo Phase Diversity cargado con éxito desde {pd_path}")
    except Exception as e:
        print(f"[INFERENCIA] Error al cargar modelo Phase Diversity: {e}")
else:
    print(f"[INFERENCIA] Modelo Phase Diversity no cargado (esperando entrenamiento).")


@app.route('/status')
def status():
    return jsonify({
        "status": "online", 
        "service": "Inferencia CNN (PyTorch ResNet-10)",
        "shared_volume": os.path.exists(SHARED_DIR),
        "models_loaded": models_loaded,
        "device": str(device),
        "resnet10_jit": resnet_jit_meta,
        "resnet10_uses_jit": resnet_uses_jit,
    })

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    model_name = request.args.get("model", "phase_diversity")
    
    if model_name not in models:
        return jsonify({"error": f"Modelo '{model_name}' no soportado"}), 400
        
    # Verificar si el modelo solicitado está cargado. Si no, intentar recargarlo.
    if not models_loaded[model_name]:
        path_map = {
            "phase_diversity": os.path.join(MODEL_DIR, "phase_diversity_cnn.pth"),
            "resnet10": os.path.join(MODEL_DIR, "resnet10_cnn.pth")
        }
        full_path = path_map[model_name]
        if os.path.exists(full_path):
            try:
                if model_name == "resnet10":
                    _load_resnet10()
                else:
                    models[model_name].load_state_dict(
                        torch.load(full_path, map_location=device, weights_only=True)
                    )
                    models[model_name].to(device)
                    models[model_name].eval()
                    models_loaded[model_name] = True
                print(f"[INFERENCIA] Modelo {model_name} recargado con éxito desde {full_path}")
            except Exception as e:
                print(f"[INFERENCIA] Error al recargar modelo {model_name}: {e}")
                
    active_model = models[model_name]
    
    try:
        if request.method == 'POST':
            psf_data = np.frombuffer(request.data, dtype=np.float32).reshape(2, 96, 96)
        else:
            if os.path.exists(PSF_PATH):
                psf_data = np.load(PSF_PATH).astype(np.float32)
            else:
                return jsonify({"error": "psf.npy no encontrado en el volumen compartido"}), 404
            
        # Ambos modelos esperan 2 canales: shape (2, 96, 96) -> preprocesar a (1, 2, 96, 96)
        if psf_data.shape != (2, 96, 96):
            return jsonify({"error": f"Modelo requiere PSF de 2 canales. Recibido: {psf_data.shape}"}), 400
        psf_tensor = torch.from_numpy(psf_data).unsqueeze(0).to(device)
            
        # Ejecutar inferencia
        with torch.no_grad():
            pred_norm = active_model(psf_tensor)
            pred_physical = denormalize_predictions(pred_norm)
            
        zernike_predictions = pred_physical.squeeze(0).cpu().numpy().tolist()
        
        return jsonify({
            "zernike": zernike_predictions,
            "timestamp": time.time(),
            "model_active": model_name,
            "source": "pytorch_cnn"
        })
        
    except Exception as e:
        print(f"[INFERENCIA] Error durante inferencia: {e}")
        return jsonify({"error": f"Fallo en inferencia: {str(e)}"}), 500

if __name__ == '__main__':
    print("--- INFERENCIA CNN MULTI-MODELO DOBLE CANAL: SERVIDOR ACTIVO ---")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
