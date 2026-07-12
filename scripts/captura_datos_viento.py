"""
captura_datos_viento.py
=======================
Script de captura de datos de evolucion de viento para analisis del modelo Kalman/LQG.

MODO DE USO:
    python scripts/captura_datos_viento.py

El script corre indefinidamente mientras el sistema AO esta activo.
Mientras el script captura, ve a la interfaz y mueve libremente los sliders
de velocidad de viento y fuerza de turbulencia.

Presiona Ctrl+C para detener. El CSV se guarda automaticamente al salir.

REQUISITOS:
    - Todos los contenedores Docker arriba (docker-compose up -d)
    - Simulador corriendo en modo estocastico (Metodo 2) para que los coeficientes evolucionen
    - pip install requests (ya instalado en el entorno base)

PUERTOS EXPUESTOS (docker-compose):
    ao_simulador -> http://localhost:5000

CONTENIDO DEL CSV:
    Columnas de contexto:
        frame              : numero de muestra secuencial
        timestamp_s        : tiempo UNIX en segundos (precision de ms)
        elapsed_s          : segundos desde inicio de captura
        method             : "1" = Manual, "2" = Estocastico
        wind_speed         : velocidad de viento configurada [0.0, 1.0]
        d_r0               : fuerza de turbulencia D/r0 [0.5, 6.0]
        active_model       : modelo CNN activo (phase_diversity / resnet10 / resnet18)
        control_mode       : modo de control (direct / kalman_lqg)
        kalman_uncertainty : traza promedio de P / N_MODES (incertidumbre del filtro)
        kalman_q           : parametro q_scale configurado
        kalman_r           : parametro cnn_rmse configurado
        kalman_delay       : pasos de anticipacion LQG configurados

    Coeficientes REALES del simulador (verdad absoluta de la turbulencia):
        z1_real .. z11_real

    Coeficientes predichos por la CNN pura (ANTES del filtro Kalman):
        z1_cnn .. z11_cnn

    Estado estimado ACTUAL del filtro Kalman (frame t, ya filtrado):
        z1_kalman .. z11_kalman

    Prediccion LQG (lo que se envia al SLM como correccion, frame t+delay):
        z1_control .. z11_control

    Errores derivados por modo (calculados aqui para facilitar analisis posterior):
        err_z1_cnn .. err_z11_cnn         : error CNN vs Real (residual que Kalman debe reducir)
        err_z1_control .. err_z11_control : residual despues de la correccion Kalman+LQG
        delta_z1_kalman_vs_cnn ..         : cuanto corrige el filtro vs la CNN pura
        delta_z1_control_vs_kalman ..     : anticipacion adicional del LQG sobre el estado filtrado
"""

import requests
import csv
import time
import os
import sys
import signal
from datetime import datetime
import threading

# ---------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------

SIMULADOR_URL = "http://localhost:5000/status"  # Puerto expuesto del simulador en el host
POLL_RATE_HZ  = 22                              # Muestreo a 22 Hz para alineacion 1-a-1 con el simulador (45ms)
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "resultados")
ZERNIKE_MODES = [f"Z{i}" for i in range(1, 12)]   # Z1..Z11 (Noll)

# ---------------------------------------------------------------
# ESTADO GLOBAL DEL SCRIPT
# ---------------------------------------------------------------

records       = []   # Lista de dicts, cada uno es una fila del CSV
frame_counter = 0
t_start       = None
running       = True


def signal_handler(sig, frame):
    """Captura Ctrl+C para salida limpia con guardado del CSV."""
    global running
    running = False
    print("\n\n[CAPTURA] Ctrl+C recibido. Guardando CSV...", flush=True)


signal.signal(signal.SIGINT, signal_handler)


def fetch_state() -> dict | None:
    """
    Llama a GET /status del simulador y devuelve el campo 'state' completo.
    El campo 'state' contiene: method, wind_speed, d_r0, active_model,
    control_mode, kalman_uncertainty, zernikes (real), cnn_zernikes,
    kalman_current, control_zernikes, camera_*.
    Retorna None si hay error de conexion.
    """
    try:
        resp = requests.get(SIMULADOR_URL, timeout=1.0)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("state", None)
    except requests.exceptions.ConnectionError:
        print(
            "[CAPTURA] ERROR: No se puede conectar con el simulador. "
            "Verifica que los contenedores esten arriba (docker-compose up -d).",
            flush=True
        )
    except requests.exceptions.Timeout:
        print("[CAPTURA] ADVERTENCIA: Timeout al consultar el simulador.", flush=True)
    except Exception as e:
        print(f"[CAPTURA] ERROR inesperado: {e}", flush=True)
    return None


def state_to_record(state: dict, frame: int, elapsed: float) -> dict:
    """
    Convierte el diccionario 'state' del simulador en una fila plana del CSV.
    Captura TODAS las variables disponibles sin omitir ninguna.
    """
    z_real    = state.get("zernikes",         {})
    z_cnn     = state.get("cnn_zernikes",     {})
    z_kalman  = state.get("kalman_current",   {})
    z_control = state.get("control_zernikes", {})

    row = {
        # ---- Contexto del sistema ----------------------------------------
        "frame":              frame,
        "timestamp_s":        time.time(),
        "elapsed_s":          round(elapsed, 4),
        "method":             state.get("method",             "?"),
        "wind_speed":         state.get("wind_speed",         float("nan")),
        "d_r0":               state.get("d_r0",               float("nan")),
        "active_model":       state.get("active_model",       "?"),
        "control_mode":       state.get("control_mode",       "?"),
        "kalman_uncertainty": state.get("kalman_uncertainty",  float("nan")),

        # ---- Parametros Kalman configurados (propagados desde interfaz) ---
        "kalman_q":           state.get("kalman_q",           float("nan")),
        "kalman_r":           state.get("kalman_r",           float("nan")),
        "kalman_delay":       state.get("kalman_delay",       float("nan")),
    }

    # ---- Coeficientes Zernike por fuente, y errores derivados --------
    for z in ZERNIKE_MODES:
        z_lower = z.lower()   # "z1", "z2", ..., "z11"

        real_val    = float(z_real   .get(z, float("nan")))
        cnn_val     = float(z_cnn    .get(z, float("nan")))
        kalman_val  = float(z_kalman .get(z, float("nan")))
        control_val = float(z_control.get(z, float("nan")))

        # -- Valores brutos de cada fuente --
        row[f"{z_lower}_real"]    = real_val
        row[f"{z_lower}_cnn"]     = cnn_val
        row[f"{z_lower}_kalman"]  = kalman_val
        row[f"{z_lower}_control"] = control_val

    return row


def save_csv(records: list, output_dir: str) -> str:
    """Guarda la lista de registros como CSV con nombre timestamped."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"captura_viento_{ts}.csv")

    if not records:
        print("[CAPTURA] No hay datos para guardar.", flush=True)
        return ""

    # ---- POST-PROCESAMIENTO: Calculo de Errores con Desplazamiento Temporal ----
    def _safe_sub(a, b):
        return a - b if (a == a and b == b) else float("nan")

    for i in range(len(records)):
        delay = int(records[i].get("kalman_delay", 1) or 1)
        
        for z in ZERNIKE_MODES:
            z_lower = z.lower()
            cnn_val     = records[i].get(f"{z_lower}_cnn", float("nan"))
            kalman_val  = records[i].get(f"{z_lower}_kalman", float("nan"))
            control_val = records[i].get(f"{z_lower}_control", float("nan"))
            real_val_t  = records[i].get(f"{z_lower}_real", float("nan"))
            
            # Error CNN: contemporaneo (la prediccion de la foto t vs la turbulencia t)
            records[i][f"err_{z_lower}_cnn"] = _safe_sub(cnn_val, real_val_t)
            
            # Error CNN lag 1: prediccion t contra turbulencia t+1 (comparacion justa con LQG)
            if i + 1 < len(records):
                real_val_t_next = records[i + 1].get(f"{z_lower}_real", float("nan"))
                records[i][f"err_{z_lower}_cnn_lag1"] = _safe_sub(cnn_val, real_val_t_next)
            else:
                records[i][f"err_{z_lower}_cnn_lag1"] = float("nan")
            
            # Deltas internos
            records[i][f"delta_{z_lower}_kalman_vs_cnn"] = _safe_sub(kalman_val, cnn_val)
            records[i][f"delta_{z_lower}_control_vs_kalman"] = _safe_sub(control_val, kalman_val)
            
            # Error Control LQG: se compara lo mandado al SLM en t contra la realidad en t+delay
            if i + delay < len(records):
                real_val_t_next = records[i + delay].get(f"{z_lower}_real", float("nan"))
                records[i][f"err_{z_lower}_control"] = _safe_sub(control_val, real_val_t_next)
            else:
                records[i][f"err_{z_lower}_control"] = float("nan") # No se puede calcular para los ultimos frames
    # ----------------------------------------------------------------------------

    # Para que las columnas siempre aparezcan en orden, usamos las keys del primer dict
    fieldnames = list(records[0].keys())
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return filepath


def print_live_summary(record: dict):
    """
    Imprime un resumen de una linea por muestra para confirmar que la
    captura esta activa mientras se usa la interfaz.
    Muestra: frame, tiempo, wind, d/r0, Z2 y Z3 (Tip/Tilt) desde las 4 fuentes,
    e incertidumbre de Kalman.
    """
    def fmt(v):
        return f"{v:+7.4f}" if v == v else "   nan "

    v    = record["wind_speed"]
    d    = record["d_r0"]
    unc  = record["kalman_uncertainty"]
    z2r  = record.get("z2_real",    float("nan"))
    z2c  = record.get("z2_cnn",     float("nan"))
    z2k  = record.get("z2_kalman",  float("nan"))
    z2ct = record.get("z2_control", float("nan"))
    z3r  = record.get("z3_real",    float("nan"))
    z3c  = record.get("z3_cnn",     float("nan"))

    print(
        f"[F{record['frame']:05d} | t={record['elapsed_s']:7.2f}s] "
        f"v={v:.2f}  d/r0={d:.1f}  "
        f"| Z2: real={fmt(z2r)} cnn={fmt(z2c)} kal={fmt(z2k)} ctrl={fmt(z2ct)} "
        f"| Z3: real={fmt(z3r)} cnn={fmt(z3c)} "
        f"| unc={unc:.5f}",
        flush=True
    )


# ---------------------------------------------------------------
# SECUENCIA AUTOMATIZADA
# ---------------------------------------------------------------

def set_simulator_config(wind_speed, d_r0):
    try:
        requests.post(
            "http://localhost:5000/config", 
            json={
                "wind_speed": wind_speed, 
                "d_r0": d_r0,
                "method": "2"  # Fuerza el modo estocastico pase lo que pase en la UI
            }, 
            timeout=1.0
        )
    except Exception as e:
        print(f"[AUTOMATION] Error conectando al simulador: {e}")

def automation_loop():
    turbulences = [1.0, 3.0, 4.5, 6.0]
    winds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    time.sleep(2) # Esperar a que inicie la captura principal
    print("\n" + "="*60)
    print("  [AUTOMATION] Iniciando secuencia automatizada...")
    print("  Se probaran 4 turbulencias x 10 vientos (20s c/u)")
    print("  Tiempo estimado: ~13 minutos. Presiona Ctrl+C para abortar.")
    print("="*60 + "\n", flush=True)
    
    for d_r0 in turbulences:
        for w in winds:
            if not running:
                return
            print(f"\n---> [AUTOMATION] Nuevo Regimen -> Turbulencia (D/r0): {d_r0:.1f} | Viento: {w:.1f} <---", flush=True)
            set_simulator_config(w, d_r0)
            
            # Esperar 20 segundos divididos en pasos cortos para respuesta rapida a Ctrl+C
            for _ in range(200):
                if not running:
                    return
                time.sleep(0.1)
                
    print("\n[AUTOMATION] SECUENCIA COMPLETADA. Presiona Ctrl+C para guardar el CSV y salir.", flush=True)


# ---------------------------------------------------------------
# BUCLE PRINCIPAL
# ---------------------------------------------------------------

def main():
    global frame_counter, t_start, running

    poll_interval = 1.0 / POLL_RATE_HZ

    print("=" * 80, flush=True)
    print("  CAPTURA DE DATOS DE VIENTO - Sistema AO", flush=True)
    print("=" * 80, flush=True)
    print(f"  URL simulador : {SIMULADOR_URL}", flush=True)
    print(f"  Frecuencia    : {POLL_RATE_HZ} Hz  ({poll_interval*1000:.0f} ms por muestra)", flush=True)
    print(f"  Salida CSV    : {OUTPUT_DIR}/captura_viento_YYYYMMDD_HHMMSS.csv", flush=True)
    print(f"  Columnas      : frame, timestamp, contexto (12), "
          f"zernike raw (44), errores derivados (44)  ->  total ~101 columnas", flush=True)
    print("", flush=True)
    print("  INSTRUCCIONES:", flush=True)
    print("  1. Asegurate de tener el simulador en Metodo 2 (Estocastico)", flush=True)
    print("  2. Abre la interfaz web y mueve los sliders de viento y turbulencia", flush=True)
    print("  3. Cubre regimenes: viento bajo (0.1-0.3), medio (0.4-0.6), alto (0.7-1.0)", flush=True)
    print("  4. Presiona Ctrl+C cuando hayas cubierto los escenarios deseados", flush=True)
    print("=" * 80, flush=True)
    print("", flush=True)

    # Verificar conexion antes de empezar
    print("[CAPTURA] Verificando conexion con el simulador...", flush=True)
    test = fetch_state()
    if test is None:
        print("[CAPTURA] FATAL: No se pudo conectar. Abortando.", flush=True)
        print("  Comprueba: docker-compose ps  y  docker-compose up -d", flush=True)
        sys.exit(1)

    print("[CAPTURA] Conexion OK. Iniciando captura...\n", flush=True)
    t_start = time.perf_counter()

    # Iniciar el hilo de automatizacion
    threading.Thread(target=automation_loop, daemon=True).start()

    while running:
        t_loop = time.perf_counter()

        state = fetch_state()

        if state is not None:
            elapsed = time.perf_counter() - t_start
            record  = state_to_record(state, frame_counter, elapsed)
            records.append(record)
            print_live_summary(record)
            frame_counter += 1

        # Mantener la tasa objetivo sin deriva acumulada
        elapsed_loop = time.perf_counter() - t_loop
        sleep_time   = max(0.0, poll_interval - elapsed_loop)
        time.sleep(sleep_time)

    # ---- Guardar CSV al salir ----------------------------------------
    filepath = save_csv(records, OUTPUT_DIR)
    if filepath:
        total_time = time.perf_counter() - t_start
        print(f"\n[CAPTURA] OK  {frame_counter} muestras en {total_time:.1f} s  "
              f"({frame_counter/max(total_time,1):.1f} Hz real)", flush=True)
        print(f"[CAPTURA] OK  CSV guardado en: {filepath}", flush=True)
    else:
        print("[CAPTURA] No se generaron datos.", flush=True)


if __name__ == "__main__":
    main()
