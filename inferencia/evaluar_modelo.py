import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import BaselineCNN
from dataset import PSFGeneratorDataset

def evaluate():
    parser = argparse.ArgumentParser(description="Evaluación de modelos de Óptica Adaptativa")
    parser.add_argument("--model", type=str, default="phase_diversity", choices=["phase_diversity", "resnet10"],
                        help="Modelo a evaluar: phase_diversity, resnet10")
    args = parser.parse_args()
    model_type = args.model

    is_docker = os.path.exists("/app/shared")
    
    if is_docker:
        model_path = f"/app/shared/{model_type}_cnn.pth"
        output_plot_path = f"/app/shared/{model_type}_correlation.png"
    else:
        model_path = f"./simulador/shared/{model_type}_cnn.pth"
        output_plot_path = f"./{model_type}_correlation.png"
        
    print(f"Evaluando modelo: {model_type.upper()}")
    print(f"Buscando pesos del modelo en: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Modelo no encontrado en {model_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluando en dispositivo: {device}")
    
    # Generar 2000 muestras al vuelo de validación
    dataset = PSFGeneratorDataset(num_samples=2000, model_type=model_type)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    in_channels = 2
    if model_type == "resnet10":
        from model_resnet import ResNet10
        model = ResNet10(in_channels=in_channels).to(device)
    else:
        model = BaselineCNN(in_channels=in_channels).to(device)
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
    
    print("\n=== METRICAS DEL MODELO EVALUADO ===")
    print(f"Total muestras evaluadas: {len(all_preds)}")
    print(f"MAE Normalizado promedio: {mean_mae:.6f}")
    print(f"Precisión global (100 * exp(-MAE)): {accuracy:.2f}%")
    print("===================================\n")
    
    # Generar gráficos de correlación
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
        
        # Calcular correlación
        r = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_true) > 0 and np.std(y_pred) > 0 else 0.0
        mae_mode = maes[i]
        
        ax.set_title(f"{zernike_names[i]}\nCorr: {r:.3f} | MAE: {mae_mode:.3f}")
        ax.set_xlabel("Valor Real (Normalizado)")
        ax.set_ylabel("Valor Predicho (Normalizado)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        
    axes[11].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=150)
    plt.close()
    print("¡Gráficos generados con éxito!")

if __name__ == "__main__":
    evaluate()
