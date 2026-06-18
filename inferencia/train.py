import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

from model import BaselineCNN
from dataset import PSFDataset, denormalize_predictions

# Configuración de rutas
DATASET_DIR = "/app/shared/dataset"
MODEL_SAVE_PATH = "/app/shared/custom_cnn.pth"
PLOT_SAVE_PATH = "/app/shared/loss_curve.png"

# Hiperparámetros
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 25

def main():
    # 1. Configurar dispositivo (GPU RTX 3050 o CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Dispositivo detectado para entrenamiento: {device} ===")
    if device.type == "cuda":
        print(f"  -> Tarjeta gráfica: {torch.cuda.get_device_name(0)}")

    # 2. Cargar Datasets y DataLoaders
    train_path = os.path.join(DATASET_DIR, "train")
    val_path = os.path.join(DATASET_DIR, "val")
    
    if not os.path.exists(train_path) or not os.listdir(train_path):
        print(f"ERROR: No se encontraron datos en {train_path}. Por favor ejecuta generar_dataset.py primero.")
        return

    print("Cargando datasets...", flush=True)
    train_dataset = PSFDataset(train_path, normalize_labels=True)
    val_dataset = PSFDataset(val_path, normalize_labels=True)

    # Nota: num_workers=2 para carga eficiente. pin_memory=True si usamos GPU.
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

    print(f"  -> Muestras de entrenamiento: {len(train_dataset)}")
    print(f"  -> Muestras de validación: {len(val_dataset)}")

    # 3. Inicializar Modelo, Optimizador y Scheduler
    model = BaselineCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # 4. Definir Pérdida Ponderada (Weighted MSE Loss)
    # Índices: Z1=0 (Piston), Z2=1 (Tip X), Z3=2 (Tilt Y), Z4-Z11 = 3-10
    # Priorizamos Z2 (Tip) y Z3 (Tilt) multiplicando sus pérdidas por 10.0
    loss_weights = torch.ones(11, dtype=torch.float32).to(device)
    loss_weights[1] = 10.0  # Z2 (Tip)
    loss_weights[2] = 10.0  # Z3 (Tilt)

    def weighted_mse_loss(pred, target):
        squared_errors = (pred - target) ** 2
        weighted_errors = squared_errors * loss_weights
        return torch.mean(weighted_errors)

    # Historial de métricas
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    print("Comenzando el bucle de entrenamiento...", flush=True)
    t_start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        # Bucle de entrenamiento por lotes (batches)
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
        
        # Bucle de validación
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
        
        # Ajustar tasa de aprendizaje según la pérdida de validación
        scheduler.step(epoch_val_loss)
        
        # Guardar el mejor modelo
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Época {epoch+1:02d}/{EPOCHS} -> Guardado Mejor Modelo! Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}", flush=True)
        else:
            print(f"Época {epoch+1:02d}/{EPOCHS} -> Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}", flush=True)

    total_time = time.time() - t_start
    print(f"=== Entrenamiento completado en {total_time/60:.2f} minutos ===")
    print(f"Mejor Pérdida de Validación Ponderada: {best_val_loss:.6f}")

    # 5. Generar y guardar la gráfica de pérdidas históricas
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Pérdida Entrenamiento", color="#EF4444", lw=2)
    plt.plot(range(1, EPOCHS + 1), val_losses, label="Pérdida Validación", color="#3B82F6", lw=2)
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida Ponderada (MSE)")
    plt.title("Evolución de la Pérdida durante el Entrenamiento de la CNN Baseline")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(PLOT_SAVE_PATH, dpi=150)
    plt.close()
    print(f"Gráfica de pérdidas guardada en: {PLOT_SAVE_PATH}")

if __name__ == "__main__":
    main()
