"""
Compilación TorchScript de ResNet-10 para inferencia en tiempo real.
Entrada fija: (batch=1, canales=2, 96, 96).
"""
import os
import time
import torch

from model_resnet import ResNet10

INPUT_SHAPE = (1, 2, 96, 96)
JIT_SUFFIX = "_jit.pt"


def jit_path_for_weights(weights_path: str) -> str:
    base, _ = os.path.splitext(weights_path)
    return f"{base}{JIT_SUFFIX}"


def _example_input(device: torch.device) -> torch.Tensor:
    return torch.randn(INPUT_SHAPE, device=device)


def trace_resnet10(model: ResNet10, device: torch.device) -> torch.jit.ScriptModule:
    """Traza ResNet-10 en eval mode. No usa freeze: BatchNorm diverge con freeze."""
    model = model.to(device)
    model.eval()

    example = _example_input(device)
    with torch.no_grad():
        eager_out = model(example)
        traced = torch.jit.trace(model, example, check_trace=False, strict=False)
        jit_out = traced(example)

    max_diff = (eager_out - jit_out).abs().max().item()
    if max_diff > 1e-4:
        raise RuntimeError(
            f"TorchScript diverge del modelo eager (max_diff={max_diff:.2e})"
        )

    return traced


def save_traced_resnet10(traced: torch.jit.ScriptModule, jit_path: str) -> None:
    os.makedirs(os.path.dirname(jit_path) or ".", exist_ok=True)
    traced.save(jit_path)


def load_traced_resnet10(jit_path: str, device: torch.device) -> torch.jit.ScriptModule:
    traced = torch.jit.load(jit_path, map_location=device)
    traced.eval()
    return traced


def _jit_cache_is_fresh(jit_path: str, weights_path: str) -> bool:
    if not os.path.exists(jit_path) or not os.path.exists(weights_path):
        return False
    return os.path.getmtime(jit_path) >= os.path.getmtime(weights_path)


def load_resnet10_for_inference(
    weights_path: str,
    device: torch.device,
    *,
    force_recompile: bool = False,
) -> tuple[torch.jit.ScriptModule, dict]:
    """
    Carga ResNet-10 optimizado con TorchScript.
    Reutiliza el artefacto .pt en caché si es más reciente que los pesos .pth.
  """
    jit_path = jit_path_for_weights(weights_path)
    meta = {
        "backend": "torchscript",
        "jit_path": jit_path,
        "weights_path": weights_path,
        "weights_loaded": False,
        "compiled_at_startup": False,
        "compile_ms": 0.0,
    }

    if not force_recompile and _jit_cache_is_fresh(jit_path, weights_path):
        t0 = time.perf_counter()
        traced = load_traced_resnet10(jit_path, device)
        meta["weights_loaded"] = True
        meta["compile_ms"] = (time.perf_counter() - t0) * 1000
        return traced, meta

    model = ResNet10(in_channels=2)
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        meta["weights_loaded"] = True

    t0 = time.perf_counter()
    traced = trace_resnet10(model, device)
    meta["compiled_at_startup"] = True
    meta["compile_ms"] = (time.perf_counter() - t0) * 1000

    if meta["weights_loaded"]:
        save_traced_resnet10(traced, jit_path)

    return traced, meta


def export_resnet10_jit(weights_path: str, device: torch.device | None = None) -> str:
    """Exporta TorchScript tras entrenamiento. Devuelve la ruta del artefacto .pt."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, meta = load_resnet10_for_inference(
        weights_path,
        device,
        force_recompile=True,
    )
    return meta["jit_path"]
