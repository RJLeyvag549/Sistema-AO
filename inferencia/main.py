from flask import Flask, jsonify, request
import numpy as np
import torch
import torch._dynamo
import glob
import os
import time
import logging

from model import BaselineCNN
from model_resnet import ResNet10, ResNet18
from dataset import denormalize_predictions
# from jit_resnet import load_resnet10_for_inference

logging.basicConfig(
    level=logging.WARNING,
    format='[INFERENCIA] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
logging.getLogger('werkzeug').setLevel(logging.WARNING)

SHARED_DIR = "/app/shared"
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(MODEL_DIR, "models")  # inferencia/models/ — pesos versionados en el repo
PSF_PATH = os.path.join(SHARED_DIR, "psf.npy")

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.warning(f"[DEVICE] Usando: {device.type.upper()}")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    # TF32: usa Tensor Cores de Ampere+ para matmul float32 (~2x más rápido, sin cambios en código)
    torch.set_float32_matmul_precision('high')
    gpu_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.warning(f"[GPU] {gpu_name} | VRAM total: {vram_total:.1f} GB")
else:
    logger.warning("[ADVERTENCIA] No se detectó CUDA. Inferencia en CPU — rendimiento limitado.")


if device.type == "cuda":
    _fix_wsl2_libcuda = None  # Fix gestionado por entrypoint.sh (antes de que Python arranque)


# Diccionario para almacenar los modelos cargados
models = {}
models_loaded = {
    "phase_diversity": False,
    "resnet10": False,
    "resnet18": False,
}
resnet_jit_meta = {}
resnet_uses_jit = False


def _compile_model(model, name: str):
    """
    Intenta torch.compile() con reduce-overhead.
    El test de compilación real se hace aquí con un forward pass dummy:
    inductor compila de forma lazy en el primer forward, no en la llamada a torch.compile().
    Si falla (ej: libcuda.so no encontrado), revierte a modo eager limpio.
    """
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        # IMPORTANTE: el error de inductor ocurre aquí (compilación lazy), no en torch.compile()
        dummy = torch.zeros(1, 2, 96, 96, device=device)
        with torch.no_grad():
            compiled(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        logger.warning(f"[COMPILE] {name}: torch.compile activo (reduce-overhead + TF32)")
        return compiled
    except Exception as e:
        logger.warning(f"[COMPILE] torch.compile falló para {name} ({type(e).__name__}). Revirtiendo a eager.")
        logger.warning(f"[COMPILE] Causa: {str(e)[:120]}")
        torch._dynamo.reset()  # Limpia el estado de dynamo para evitar recompilaciones fallidas
        return model  # Retorna el modelo original sin compilar


def _warmup_model(model, name: str):
    """Warm-up adicional: el primer forward ya se hizo en _compile_model, estos son extra."""
    dummy = torch.zeros(1, 2, 96, 96, device=device)
    with torch.no_grad():
        for _ in range(2):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    logger.warning(f"[WARMUP] {name} listo")


def _resolve_pth(filename):
    """
    Resuelve la ruta de los pesos en orden de prioridad:
      1. /app/shared/  — salida de train.py (re-entrenamiento en vivo tiene precedencia)
      2. /app/models/  — pesos versionados en el repositorio (arranque en frío sin volumen)
      3. /app/         — compatibilidad con instalaciones antiguas
    """
    shared = os.path.join(SHARED_DIR, filename)
    if os.path.exists(shared):
        return shared
    versioned = os.path.join(MODELS_DIR, filename)
    if os.path.exists(versioned):
        return versioned
    return os.path.join(MODEL_DIR, filename)


def _load_resnet10_eager(resnet_path: str):
    """Fallback: modelo eager si TorchScript falla."""
    global resnet_jit_meta, resnet_uses_jit
    model = ResNet10(in_channels=2).to(device)
    if os.path.exists(resnet_path):
        try:
            model.load_state_dict(
                torch.load(resnet_path, map_location=device, weights_only=True)
            )
            models_loaded["resnet10"] = True
        except Exception as e:
            models_loaded["resnet10"] = False
            logger.warning(f"[LOAD] Error al cargar pesos de ResNet-10 ({e}). Iniciando modelo vacio para permitir entrenamiento.")
    else:
        models_loaded["resnet10"] = False
    model.eval()
    model = _compile_model(model, "ResNet10")
    _warmup_model(model, "ResNet10")
    models["resnet10"] = model
    resnet_uses_jit = False
    resnet_jit_meta = {"backend": "eager+compiled", "weights_path": resnet_path}


def _load_resnet10():
    # Buscar primero en SHARED_DIR (donde train.py guarda), luego en MODEL_DIR
    resnet_path = _resolve_pth("resnet10_cnn.pth")
    _load_resnet10_eager(resnet_path)


def _load_resnet18_eager(resnet_path: str):
    """Carga ResNet-18 en modo eager con torch.compile si CUDA disponible."""
    model = ResNet18(in_channels=2).to(device)
    if os.path.exists(resnet_path):
        try:
            model.load_state_dict(
                torch.load(resnet_path, map_location=device, weights_only=True)
            )
            models_loaded["resnet18"] = True
            logger.warning(f"[LOAD] ResNet-18 cargado desde {resnet_path}")
        except Exception as e:
            models_loaded["resnet18"] = False
            logger.warning(f"[LOAD] Error al cargar pesos de ResNet-18 ({e}). Iniciando modelo vacio para permitir entrenamiento.")
    else:
        models_loaded["resnet18"] = False
        logger.warning(f"[LOAD] ResNet-18 no encontrado en {resnet_path} — modelo no cargado.")
    model.eval()
    model = _compile_model(model, "ResNet18")
    _warmup_model(model, "ResNet18")
    models["resnet18"] = model


def _load_resnet18():
    # Buscar primero en SHARED_DIR (donde train.py guarda), luego en MODEL_DIR
    resnet_path = _resolve_pth("resnet18_cnn.pth")
    _load_resnet18_eager(resnet_path)


# 0. Cargar Modelo ResNet-10 (2 Canales) directamente desde el .pth
_load_resnet10()

# 0b. Cargar Modelo ResNet-18 si existe el .pth entrenado
_load_resnet18()

# 1. Cargar Modelo Phase Diversity (2 Canales)
models["phase_diversity"] = BaselineCNN(in_channels=2)
pd_path = _resolve_pth("phase_diversity_cnn.pth")
if os.path.exists(pd_path):
    try:
        models["phase_diversity"].load_state_dict(torch.load(pd_path, map_location=device))
        models["phase_diversity"].to(device)
        models["phase_diversity"].eval()
        models["phase_diversity"] = _compile_model(models["phase_diversity"], "BaselineCNN")
        _warmup_model(models["phase_diversity"], "BaselineCNN")
        models_loaded["phase_diversity"] = True
        logger.warning(f"[LOAD] Modelo Phase Diversity cargado desde {pd_path}")
    except Exception as e:
        logger.error(f"Error al cargar modelo Phase Diversity: {e}")
else:
    logger.warning("Modelo Phase Diversity no cargado (esperando entrenamiento).")


@app.route('/status')
def status():
    return jsonify({
        "status": "online", 
        "service": "Inferencia CNN (PyTorch ResNet-10 / ResNet-18)",
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
            "phase_diversity": _resolve_pth("phase_diversity_cnn.pth"),
            "resnet10": _resolve_pth("resnet10_cnn.pth"),
            "resnet18": _resolve_pth("resnet18_cnn.pth"),
        }
        full_path = path_map[model_name]
        if os.path.exists(full_path):
            try:
                if model_name == "resnet10":
                    _load_resnet10()
                elif model_name == "resnet18":
                    _load_resnet18()
                else:
                    models[model_name].load_state_dict(
                        torch.load(full_path, map_location=device, weights_only=True)
                    )
                    models[model_name].to(device)
                    models[model_name].eval()
                    models_loaded[model_name] = True
                logger.debug(f"Modelo {model_name} recargado con éxito desde {full_path}")
            except Exception as e:
                logger.error(f"Error al recargar modelo {model_name}: {e}")
                
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
        psf_tensor = torch.from_numpy(psf_data).unsqueeze(0).to(device, non_blocking=True)
            
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
        logger.error(f"Error durante inferencia: {e}")
        return jsonify({"error": f"Fallo en inferencia: {str(e)}"}), 500

if __name__ == '__main__':
    logger.warning("--- INFERENCIA CNN MULTI-MODELO DOBLE CANAL: SERVIDOR ACTIVO ---")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
