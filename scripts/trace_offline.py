import sys
from pathlib import Path
import numpy as np

# Añadir script root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.benchmark_controlador import simulate_turbulence, simulate_cnn_observations
from controlador.control_vectorial import ZernikeKalmanVectorial, N_MODES

def generate_short_trace():
    print('--- INICIANDO TRAZA CORTA MATRIZ A MATRIZ (SIMULADOR ONLINE OFFLINE) ---')
    
    wind_speed = 20.0
    d_r0 = 5.0
    cnn_rmse = 0.120 # PHASE_DIVERSITY
    delay = 1
    
    truth = simulate_turbulence(n_frames=35, wind_speed=wind_speed, d_r0=d_r0, seed=42)
    obs_cnn = simulate_cnn_observations(truth, cnn_rmse=cnn_rmse, seed=42)
    
    kf = ZernikeKalmanVectorial(q_scale=1.0, cnn_rmse=cnn_rmse, delay=delay)
    
    print(f'\nEscenario: Viento = {wind_speed} m/s, D/r0 = {d_r0:.1f} (Ruido CNN: {cnn_rmse} rad)')
    print(f'{"Frame":<6} | {"MAE CNN (rad)":<15} | {"MAE LQG (rad)":<15} | {"Z4 Real (Desfoco)":<20} | {"Z4 CNN (Desfoco)":<20} | {"Z4 LQG Predicho":<20}')
    print('-' * 115)
    
    for t in range(30):
        y_t = obs_cnn[t]
        x_tp1 = truth[t + delay] 
        
        x_current, x_predicted = kf.update(y=y_t, wind_speed=wind_speed, d_r0=d_r0)
        
        # Saltamos el warmup
        if t < 5:
            continue
            
        mae_cnn = np.mean(np.abs(x_tp1 - y_t))
        mae_lqg = np.mean(np.abs(x_tp1 - x_predicted))
        
        z4_real = x_tp1[2] # Z4 es el indice 2 porque empieza en Z2
        z4_cnn = y_t[2]
        z4_lqg = x_predicted[2]
        
        print(f'{t:<6} | {mae_cnn:<15.4f} | {mae_lqg:<15.4f} | {z4_real:>10.4f}         | {z4_cnn:>10.4f}         | {z4_lqg:>10.4f}')

if __name__ == '__main__':
    generate_short_trace()
