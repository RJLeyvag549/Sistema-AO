import time
import requests
import numpy as np

def run_trace(duration=10.0, interval=0.045):
    print(f"Iniciando captura de precisiones online por {duration} segundos...")
    start = time.time()
    
    records = []
    
    while (time.time() - start) < duration:
        try:
            resp = requests.get("http://localhost:5000/status", timeout=1.0)
            if resp.status_code == 200:
                data = resp.json()
                state = data.get("state", {})
                real_z = state.get("zernikes", {})
                cnn_z = state.get("cnn_zernikes", {})
                lqg_z = state.get("control_zernikes", {})
                
                # Z2 a Z11 (10 modos)
                reals = [real_z.get(f"Z{i}", 0.0) for i in range(2, 12)]
                cnns  = [cnn_z.get(f"Z{i}", 0.0) for i in range(2, 12)]
                lqgs  = [lqg_z.get(f"Z{i}", 0.0) for i in range(2, 12)]
                
                mae_cnn = np.mean(np.abs(np.array(reals) - np.array(cnns)))
                mae_lqg = np.mean(np.abs(np.array(reals) - np.array(lqgs)))
                
                records.append({
                    "time": time.time() - start,
                    "mae_cnn": mae_cnn,
                    "mae_lqg": mae_lqg,
                    "reals": reals,
                    "cnns": cnns,
                    "lqgs": lqgs
                })
        except Exception as e:
            pass
        time.sleep(interval)
        
    print(f"\nTraza capturada: {len(records)} frames.")
    
    # Formateo de salida en consola (mostramos los primeros 5 modos por espacio, 
    # pero guardamos todo el conjunto de los 10 modos)
    print("\nMuestra de la prediccion en conjunto (ultimos 15 frames) para Z2 a Z11:")
    print("--------------------------------------------------------------------------------")
    
    for r in records[-15:]:
        # Redondear y unir para mostrar como array de numpy
        r_str = np.array2string(np.array(r['reals']), formatter={'float_kind':lambda x: f"{x:6.2f}"}, separator=',')
        c_str = np.array2string(np.array(r['cnns']), formatter={'float_kind':lambda x: f"{x:6.2f}"}, separator=',')
        l_str = np.array2string(np.array(r['lqgs']), formatter={'float_kind':lambda x: f"{x:6.2f}"}, separator=',')
        
        print(f"T: {r['time']:5.2f}s | MAE CNN: {r['mae_cnn']:.3f} | MAE LQG: {r['mae_lqg']:.3f}")
        print(f"  REAL: {r_str}")
        print(f"  CNN:  {c_str}")
        print(f"  LQG:  {l_str}")
        print()
        
    # Guardar en un CSV
    import csv
    with open('scripts/trace_online_10s.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ['time', 'mae_cnn', 'mae_lqg']
        for i in range(2, 12): headers.append(f'real_Z{i}')
        for i in range(2, 12): headers.append(f'cnn_Z{i}')
        for i in range(2, 12): headers.append(f'lqg_Z{i}')
        writer.writerow(headers)
        
        for r in records:
            row = [r['time'], r['mae_cnn'], r['mae_lqg']] + r['reals'] + r['cnns'] + r['lqgs']
            writer.writerow(row)
            
    print("Traza completa de los 10 modos guardada en: scripts/trace_online_10s.csv")

if __name__ == "__main__":
    run_trace()
