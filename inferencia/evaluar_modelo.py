import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import BaselineCNN
from dataset import PSFDataset

def evaluate():
    # Detectar si estamos dentro de Docker o en Windows Host
    # Dentro de Docker /app/shared es la ruta estándar. En Windows local, si no existe,
    # el usuario puede indicar la ruta local.
    is_docker = os.path.exists("/app/shared")
    
    if is_docker:
        val_path = "/app/shared/dataset/val"
        model_path = "/app/shared/custom_cnn.pth"
        output_plot_path = "/app/shared/zernike_correlation.png"
    else:
        # Rutas por defecto en el host local si el volumen compartido no está montado directamente
        # (Se puede ajustar si el usuario tiene una copia local del dataset)
        val_path = "./simulador/shared/dataset/val"
        model_path = "./simulador/shared/custom_cnn.pth"
        output_plot_path = "./zernike_correlation.png"
        
    print(f"Buscando datos en: {val_path}")
    print(f"Buscando modelo en: {model_path}")
    
    if not os.path.exists(val_path):
        print(f"Error: Dataset de validación no encontrado en {val_path}")
        print("Sugerencia: Ejecuta el script dentro del contenedor con:")
        print("  docker exec ao_inferencia python evaluar_modelo.py")
        return
        
    if not os.path.exists(model_path):
        print(f"Error: Modelo no encontrado en {model_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluando en dispositivo: {device}")
    
    dataset = PSFDataset(val_path, normalize_labels=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    model = BaselineCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for psfs, targets in loader:
            psfs, targets = psfs.to(device), targets.to(device)
            preds = model(psfs)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calcular métricas globales
    maes = np.mean(np.abs(all_preds - all_targets), axis=0)
    mean_mae = np.mean(maes)
    accuracy = 100.0 * np.exp(-mean_mae)
    
    print("\n=== METRICAS DEL MODELO ACTUAL ===")
    print(f"Total muestras evaluadas: {len(all_preds)}")
    print(f"MAE Normalizado promedio: {mean_mae:.6f}")
    print(f"Precisión global (100 * exp(-MAE)): {accuracy:.2f}%")
    print("===================================\n")
    
    # Generar la "Matriz de Correlación" (Actual vs Predicho para cada uno de los 11 Zernikes)
    # Como es un problema de regresión continua, la matriz de confusión equivalente es el gráfico
    # de dispersión (Scatter Plot) de Valores Reales vs Predichos para cada variable.
    
    print(f"Generando gráficos de correlación (Actual vs Predicho) en {output_plot_path}...")
    fig, axes = plt.subplots(4, 3, figsize=(15, 18))
    axes = axes.ravel()
    
    zernike_names = ["Z1 (Piston)", "Z2 (Tip X)", "Z3 (Tilt Y)", "Z4 (Defocus)", 
                     "Z5 (Astig 45)", "Z6 (Astig 0)", "Z7 (Coma X)", "Z8 (Coma Y)", 
                     "Z9 (Trefoil X)", "Z10 (Trefoil Y)", "Z11 (Spherical)"]
    
    for i in range(11):
        ax = axes[i]
        y_true = all_targets[:, i]
        y_pred = all_preds[:, i]
        
        # Gráfico de dispersión
        ax.scatter(y_true, y_pred, alpha=0.3, color="#3B82F6", s=10)
        
        # Línea de identidad ideal (y = x)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ideal (y=x)")
        
        # Calcular R2 o correlación de Pearson
        r = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_true) > 0 and np.std(y_pred) > 0 else 0.0
        mae_mode = maes[i]
        
        ax.set_title(f"{zernike_names[i]}\nCorr: {r:.3f} | MAE: {mae_mode:.3f}")
        ax.set_xlabel("Valor Real (Normalizado)")
        ax.set_ylabel("Valor Predicho (Normalizado)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        
    # Ocultar el 12º subplot vacío
    axes[11].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=150)
    plt.close()
    print("¡Gráficos generados con éxito!")

if __name__ == "__main__":
    evaluate()
