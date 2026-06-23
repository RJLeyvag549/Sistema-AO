import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

from model import BaselineCNN
from model_resnet import ResNet10
from dataset import PSFGeneratorDataset
from jit_resnet import export_resnet10_jit

# Configuración de rutas
SHARED_DIR = "/app/shared"

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento al vuelo de la CNN de Óptica Adaptativa")
    parser.add_argument("--model", type=str, default="phase_diversity", choices=["phase_diversity", "resnet10"],
                        help="Tipo de modelo a entrenar: phase_diversity, resnet10")
    parser.add_argument("--samples", type=int, default=100000,
                        help="Número de muestras equivalentes para entrenar")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Número de épocas de entrenamiento")
    args = parser.parse_args()

    model_type = args.model
    samples = args.samples
    epochs = args.epochs

    MODEL_SAVE_PATH = os.path.join(SHARED_DIR, f"{model_type}_cnn.pth")
    PLOT_SAVE_PATH = os.path.join(SHARED_DIR, f"{model_type}_loss_curve.png")

    # Hiperparámetros
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001

    # 1. Configurar dispositivo (GPU o CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Iniciando Entrenamiento para: {model_type.upper()} ===")
    print(f"Dispositivo: {device}")
    if device.type == "cuda":
        print(f"  -> Tarjeta gráfica: {torch.cuda.get_device_name(0)}")
    print(f"Muestras de entrenamiento: {samples}")
    print(f"Épocas: {epochs}\n")

    # 2. Cargar Datasets al Vuelo
    print("Inicializando generador de datos al vuelo...", flush=True)
    train_dataset = PSFGeneratorDataset(num_samples=samples, model_type=model_type)
    # 2,000 muestras fijas de validación para mantener rapidez
    val_dataset = PSFGeneratorDataset(num_samples=2000, model_type=model_type)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=(device.type == "cuda")
    )

    # 3. Inicializar Modelo (siempre 2 canales para phase_diversity o resnet10)
    in_channels = 2
    if model_type == "resnet10":
        model = ResNet10(in_channels=in_channels).to(device)
    else:
        model = BaselineCNN(in_channels=in_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # 4. Definir Pérdida Ponderada (Weighted MSE Loss)
    loss_weights = torch.ones(11, dtype=torch.float32).to(device)
    loss_weights[1] = 10.0  # Z2 (Tip)
    loss_weights[2] = 10.0  # Z3 (Tilt)

    def weighted_mse_loss(pred, target):
        squared_errors = (pred - target) ** 2
        weighted_errors = squared_errors * loss_weights
        return torch.mean(weighted_errors)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Bucle de entrenamiento por lotes
        for psfs, targets in train_loader:
            psfs, targets = psfs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(psfs)
            loss = weighted_mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * psfs.size(0)
            
        epoch_train_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)
        
        # Validación
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for psfs, targets in val_loader:
                psfs, targets = psfs.to(device), targets.to(device)
                outputs = model(psfs)
                loss = weighted_mse_loss(outputs, targets)
                running_val_loss += loss.item() * psfs.size(0)
                
        epoch_val_loss = running_val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)
        
        scheduler.step(epoch_val_loss)
        
        # Guardar el mejor modelo
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Época {epoch+1:02d}/{epochs} -> ¡Guardado Mejor Modelo! Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}", flush=True)
        else:
            print(f"Época {epoch+1:02d}/{epochs} -> Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}", flush=True)

    total_time = time.time() - t_start
    print(f"=== Entrenamiento {model_type.upper()} completado en {total_time/60:.2f} minutos ===")
    print(f"Mejor Pérdida de Validación: {best_val_loss:.6f}")

    if model_type == "resnet10" and os.path.exists(MODEL_SAVE_PATH):
        try:
            jit_path = export_resnet10_jit(MODEL_SAVE_PATH, device)
            print(f"TorchScript exportado: {jit_path}", flush=True)
        except Exception as e:
            print(f"Advertencia: no se pudo exportar TorchScript: {e}", flush=True)

    # 5. Generar y guardar la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Pérdida Entrenamiento", color="#EF4444", lw=2)
    plt.plot(range(1, epochs + 1), val_losses, label="Pérdida Validación", color="#3B82F6", lw=2)
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida Ponderada (MSE)")
    plt.title(f"Pérdida durante Entrenamiento ({model_type.upper()})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(PLOT_SAVE_PATH, dpi=150)
    plt.close()
    print(f"Gráfica guardada en: {PLOT_SAVE_PATH}\n")

if __name__ == "__main__":
    main()
