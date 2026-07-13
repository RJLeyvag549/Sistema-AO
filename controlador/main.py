from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time
import requests
import os
import logging
import csv
from datetime import datetime
import numpy as np
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from control_vectorial import ZernikeKalmanVectorial

logging.getLogger('werkzeug').setLevel(logging.ERROR)
app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------
# CONFIGURACIÓN DE SERVICIOS
# ---------------------------------------------------------------
INFERENCIA_URL = os.environ.get("INFERENCIA_URL", "http://ao_inferencia:5000").rstrip('/') + "/predict"
SIMULADOR_URL  = os.environ.get("SIMULADOR_URL",  "http://ao_simulador:5000").rstrip('/')

# ---------------------------------------------------------------
# CONFIGURACIÓN DE INFLUXDB
# ---------------------------------------------------------------
DB_URL    = os.environ.get("DATABASE_URL",    "http://ao_database:8086")
DB_TOKEN  = os.environ.get("INFLUXDB_TOKEN",  "token_provisorio_de_github")
DB_ORG    = os.environ.get("INFLUXDB_ORG",    "organizacion_ao")
DB_BUCKET = os.environ.get("INFLUXDB_BUCKET", "telemetria_bucket")

# ---------------------------------------------------------------
# FILTRO KALMAN VECTORIAL MIMO + LQG
# ---------------------------------------------------------------
kalman_config = {
    "q_scale":    1.0,
    "cnn_rmse":   0.5,
    "delay":      1,
    "wind_angle": 0.0,
    "d_r0":       1.0,
}

vectorial_filter = ZernikeKalmanVectorial(
    q_scale=kalman_config["q_scale"],
    cnn_rmse=kalman_config["cnn_rmse"],
    delay=kalman_config["delay"],
)

# ---------------------------------------------------------------
# RUTAS PRINCIPALES
# ---------------------------------------------------------------

@app.route('/status')
def status():
    return jsonify({
        "status": "online",
        "service": "Controlador AO (Physical Model)"
    })


@app.route('/config', methods=['POST'])
def update_kalman_config():
    global kalman_config, vectorial_filter
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "No data provided"}), 400

    if 'q_scale'    in data: kalman_config['q_scale']    = float(data['q_scale'])
    if 'cnn_rmse'   in data: kalman_config['cnn_rmse']   = float(data['cnn_rmse'])
    if 'delay'      in data: kalman_config['delay']       = int(data['delay'])
    if 'wind_angle' in data: kalman_config['wind_angle']  = float(data['wind_angle'])
    if 'd_r0'       in data: kalman_config['d_r0']        = float(data['d_r0'])

    vectorial_filter = ZernikeKalmanVectorial(
        q_scale=kalman_config['q_scale'],
        cnn_rmse=kalman_config['cnn_rmse'],
        delay=kalman_config['delay'],
    )
    return jsonify({"status": "success", "config": kalman_config})


@app.route('/process_frame', methods=['POST'])
def process_frame():
    model_name     = request.args.get('model', 'phase_diversity')
    wind_speed     = float(request.args.get('wind_speed', 0.5))
    d_r0           = float(request.args.get('d_r0', kalman_config['d_r0']))
    wind_angle_rad = float(request.args.get('wind_angle', kalman_config['wind_angle']))

    try:
        resp = requests.post(
            f"{INFERENCIA_URL}?model={model_name}",
            data=request.data,
            headers={"Content-Type": "application/octet-stream"},
            timeout=1.5
        )
        if resp.status_code != 200:
            return jsonify({"status": "error", "message": f"Inference returned {resp.status_code}"}), resp.status_code
        result = resp.json()
        pred_list = result.get("zernike", [])
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error inferencia: {str(e)}"}), 500

    if len(pred_list) < 11:
        return jsonify({"status": "error", "message": "CNN devolvió menos de 11 coeficientes"}), 500

    y_obs = np.array(pred_list[1:11], dtype=np.float64)

    rmse_profiles = {'phase_diversity': 0.12, 'resnet10': 0.08, 'resnet18': 0.05}
    vectorial_filter.cnn_rmse = rmse_profiles.get(model_name, 0.05)
    vectorial_filter.q_scale  = kalman_config['q_scale']
    vectorial_filter.delay    = kalman_config['delay']

    x_current, x_predicted = vectorial_filter.update(
        y=y_obs,
        wind_speed=wind_speed,
        d_r0=d_r0,
        wind_angle_rad=wind_angle_rad,
    )

    return jsonify({
        "cnn_zernikes":     [pred_list[0]] + pred_list[1:11],
        "kalman_current":   [0.0] + x_current.tolist(),
        "control_zernikes": [0.0] + x_predicted.tolist(),
        "uncertainty":      vectorial_filter.uncertainty,
    })


# ---------------------------------------------------------------
# TELEMETRÍA: INFLUXDB — CONSULTA DE ESTADÍSTICAS
# ---------------------------------------------------------------

@app.route('/telemetry/stats', methods=['GET'])
def get_telemetry_stats():
    """
    Consulta los últimos 60 puntos de telemetría en InfluxDB y devuelve
    estadísticas RMSE y serie temporal para el dashboard de analíticas.
    """
    try:
        query = f'''
        from(bucket: "{DB_BUCKET}")
          |> range(start: -5m)
          |> filter(fn: (r) => r["_measurement"] == "turbulence_telemetry")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> sort(columns: ["_time"])
          |> limit(n: 60)
        '''

        stats_list = []
        with InfluxDBClient(url=DB_URL, token=DB_TOKEN, org=DB_ORG) as client:
            query_api = client.query_api()
            tables = query_api.query(query)

            for table in tables:
                for record in table.records:
                    row = dict(record.values)
                    if "_time" in row:
                        row["time"] = str(row["_time"])
                    # Limpiar claves internas de InfluxDB
                    row = {k: v for k, v in row.items() if not k.startswith("_")}
                    stats_list.append(row)

        # Calcular RMSE acumulado CNN vs Kalman+LQG
        summary = {"rmse_cnn": 0.0, "rmse_control": 0.0, "improvement": 0.0}
        if stats_list:
            cnn_sq, ctrl_sq = [], []
            for row in stats_list:
                for i in range(2, 12):
                    real    = float(row.get(f"z{i}",         0.0) or 0.0)
                    cnn     = float(row.get(f"z{i}_cnn",     0.0) or 0.0)
                    control = float(row.get(f"z{i}_control", 0.0) or 0.0)
                    cnn_sq.append((cnn - real) ** 2)
                    ctrl_sq.append((control - real) ** 2)

            if cnn_sq:
                rmse_cnn  = float(np.sqrt(np.mean(cnn_sq)))
                rmse_ctrl = float(np.sqrt(np.mean(ctrl_sq)))
                improvement = ((rmse_cnn - rmse_ctrl) / rmse_cnn * 100) if rmse_cnn > 0 else 0.0
                summary = {
                    "rmse_cnn":     round(rmse_cnn, 4),
                    "rmse_control": round(rmse_ctrl, 4),
                    "improvement":  round(improvement, 2),
                }

        return jsonify({"status": "success", "summary": summary, "history": stats_list})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ---------------------------------------------------------------
# TELEMETRÍA: CSV — CONTROL DE GRABACIÓN
# ---------------------------------------------------------------

CSV_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "csv_logs")
CSV_MODES      = [f"Z{i}" for i in range(1, 12)]

CSV_FIELDNAMES = [
    "frame", "timestamp_s", "elapsed_s", "method",
    "wind_speed", "d_r0", "active_model", "control_mode",
    "kalman_uncertainty", "kalman_q", "kalman_r", "kalman_delay",
]
for prefix in ["real", "cnn", "kalman", "control"]:
    for z in CSV_MODES:
        CSV_FIELDNAMES.append(f"{z.lower()}_{prefix}")
for z in CSV_MODES:
    CSV_FIELDNAMES.append(f"err_{z.lower()}_cnn")
    CSV_FIELDNAMES.append(f"delta_{z.lower()}_kalman_vs_cnn")

csv_file_lock         = threading.Lock()
csv_file_handle       = None
csv_file_writer       = None
csv_recording_active  = False
csv_file_path         = None
csv_frame_counter     = 0
csv_recording_start   = None
_loop_count           = 0

# Estado de telemetría bajo demanda
telemetry_active      = False


def _build_csv_row(state: dict, frame: int, elapsed: float) -> dict:
    row = {
        "frame":              frame,
        "timestamp_s":        time.time(),
        "elapsed_s":          round(elapsed, 4),
        "method":             state.get("method",             "?"),
        "wind_speed":         state.get("wind_speed",         float("nan")),
        "d_r0":               state.get("d_r0",               float("nan")),
        "active_model":       state.get("active_model",       "?"),
        "control_mode":       state.get("control_mode",       "?"),
        "kalman_uncertainty": state.get("kalman_uncertainty",  float("nan")),
        "kalman_q":           state.get("kalman_q",           float("nan")),
        "kalman_r":           state.get("kalman_r",           float("nan")),
        "kalman_delay":       state.get("kalman_delay",       float("nan")),
    }
    z_real    = state.get("zernikes",         {})
    z_cnn     = state.get("cnn_zernikes",     {})
    z_kalman  = state.get("kalman_current",   {})
    z_control = state.get("control_zernikes", {})

    for z in CSV_MODES:
        zl = z.lower()
        rv = float(z_real   .get(z, float("nan")))
        cv = float(z_cnn    .get(z, float("nan")))
        kv = float(z_kalman .get(z, float("nan")))
        xv = float(z_control.get(z, float("nan")))
        row[f"{zl}_real"]    = rv
        row[f"{zl}_cnn"]     = cv
        row[f"{zl}_kalman"]  = kv
        row[f"{zl}_control"] = xv
        row[f"err_{zl}_cnn"]             = cv - rv if (cv == cv and rv == rv) else float("nan")
        row[f"delta_{zl}_kalman_vs_cnn"] = kv - cv if (kv == kv and cv == cv) else float("nan")

    return row


def _open_csv_writer() -> tuple:
    global csv_file_handle, csv_file_writer, csv_recording_active
    global csv_file_path, csv_frame_counter, csv_recording_start

    with csv_file_lock:
        if csv_recording_active:
            return False, "La grabación ya está activa."
        os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file_path   = os.path.join(CSV_OUTPUT_DIR, f"telemetry_{ts}.csv")
        csv_file_handle = open(csv_file_path, "w", newline="", encoding="utf-8")
        csv_file_writer = csv.DictWriter(csv_file_handle, fieldnames=CSV_FIELDNAMES)
        csv_file_writer.writeheader()
        csv_file_handle.flush()
        csv_recording_active = True
        csv_frame_counter    = 0
        csv_recording_start  = time.time()

    return True, csv_file_path


def _close_csv_writer() -> tuple:
    global csv_file_handle, csv_file_writer, csv_recording_active
    global csv_file_path, csv_frame_counter, csv_recording_start

    with csv_file_lock:
        if not csv_recording_active:
            return False, "No hay grabación activa."
        try:
            csv_file_handle.flush()
            csv_file_handle.close()
        except Exception as e:
            return False, f"Error al cerrar CSV: {e}"
        finally:
            csv_file_handle      = None
            csv_file_writer      = None
            csv_recording_active = False
            csv_frame_counter    = 0
            csv_recording_start  = None
            saved = csv_file_path
            csv_file_path        = None

    return True, f"Grabación detenida. Archivo: {saved}"


@app.route('/telemetry/csv/start', methods=['POST'])
def start_csv():
    ok, msg = _open_csv_writer()
    return jsonify({"status": "success" if ok else "error", "message": msg}), (200 if ok else 400)


@app.route('/telemetry/csv/stop', methods=['POST'])
def stop_csv():
    ok, msg = _close_csv_writer()
    return jsonify({"status": "success" if ok else "error", "message": msg}), (200 if ok else 400)


@app.route('/telemetry/csv/status', methods=['GET'])
def csv_status():
    return jsonify({
        "recording":     csv_recording_active,
        "file_path":     csv_file_path or "",
        "frames_written": csv_frame_counter,
        "elapsed_s":     (time.time() - csv_recording_start) if csv_recording_active else 0.0,
    })


@app.route('/telemetry/session/start', methods=['POST'])
def session_start():
    global telemetry_active
    telemetry_active = True
    return jsonify({
        "status": "success",
        "telemetry_active": telemetry_active,
        "message": "Telemetría hacia InfluxDB activada. CSV no iniciado."
    })


@app.route('/telemetry/session/stop', methods=['POST'])
def session_stop():
    global telemetry_active
    telemetry_active = False
    return jsonify({
        "status": "success",
        "telemetry_active": telemetry_active,
        "message": "Telemetría hacia InfluxDB desactivada. CSV no afectado."
    })


# ---------------------------------------------------------------
# LOOP DE TELEMETRÍA (hilo daemon)
# ---------------------------------------------------------------

def logic_loop():
    global csv_recording_active, csv_frame_counter, csv_recording_start
    global csv_file_handle, csv_file_writer, csv_file_path, _loop_count

    # Construir URL del simulador (sin sufijo /correct ni /status)
    sim_base   = SIMULADOR_URL.rstrip('/')
    if sim_base.endswith('/correct'):
        sim_base = sim_base[:-len('/correct')]
    status_url = f"{sim_base}/status"

    print(f"[CONTROLADOR] Loop de telemetría en espera (bajo demanda).", flush=True)
    print(f"  Simulador : {status_url}", flush=True)
    print(f"  InfluxDB  : {DB_URL}", flush=True)
    print(f"  CSV dir   : {CSV_OUTPUT_DIR}", flush=True)

    # Esperar al simulador antes de estar listo
    for intento in range(30):
        try:
            r = requests.get(status_url, timeout=2)
            if r.status_code == 200:
                print(f"[CONTROLADOR] Conexión inicial simulador OK.", flush=True)
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        print("[CONTROLADOR] Simulador no disponible al iniciar.", flush=True)

    influx_errors = 0

    while True:
        try:
            if not telemetry_active:
                time.sleep(0.5)
                continue

            state = {}
            resp = requests.get(status_url, timeout=2)
            if resp.status_code == 200:
                state = resp.json().get("state", {})
                zernikes         = state.get("zernikes",         {})
                cnn_zernikes     = state.get("cnn_zernikes",     {})
                control_zernikes = state.get("control_zernikes", {})
                kalman_unc       = state.get("kalman_uncertainty", 0.0)
                d_r0             = state.get("d_r0",  1.0)
                method           = state.get("method", "1")

                # Log periódico cada ~30s
                _loop_count += 1
                if _loop_count % 60 == 0:
                    csv_info = f"frame={csv_frame_counter}" if csv_recording_active else "INACTIVA"
                    print(f"[CONTROLADOR] ciclo={_loop_count} d/r0={d_r0:.1f} "
                          f"Z2_real={zernikes.get('Z2',0):.4f} "
                          f"Z2_cnn={cnn_zernikes.get('Z2',0):.4f} CSV:{csv_info}", flush=True)

                # --- Escribir a InfluxDB ---
                try:
                    with InfluxDBClient(url=DB_URL, token=DB_TOKEN, org=DB_ORG) as client:
                        write_api = client.write_api(write_options=SYNCHRONOUS)
                        point = (
                            Point("turbulence_telemetry")
                            .tag("device", "holoeye_pluto")
                            .tag("method", method)
                            .field("d_r0", float(d_r0))
                            .field("kalman_uncertainty", float(kalman_unc))
                        )
                        for k, v in zernikes.items():
                            point.field(k.lower(), float(v))
                        for k, v in cnn_zernikes.items():
                            point.field(f"{k.lower()}_cnn", float(v))
                        for k, v in control_zernikes.items():
                            point.field(f"{k.lower()}_control", float(v))
                        point.time(time.time_ns(), WritePrecision.NS)
                        write_api.write(bucket=DB_BUCKET, record=point)
                        influx_errors = 0
                except Exception as ie:
                    influx_errors += 1
                    if influx_errors <= 3 or influx_errors % 30 == 0:
                        print(f"[CONTROLADOR] InfluxDB error ({influx_errors}): {ie}", flush=True)

            # --- Escribir al CSV (independiente de InfluxDB) ---
            if csv_recording_active and state:
                with csv_file_lock:
                    try:
                        row = _build_csv_row(state, csv_frame_counter,
                                             time.time() - csv_recording_start)
                        csv_file_writer.writerow(row)
                        csv_file_handle.flush()
                        csv_frame_counter += 1
                    except Exception as ce:
                        print(f"[CONTROLADOR] Error CSV: {ce}", flush=True)

            time.sleep(0.5)

        except requests.exceptions.ConnectionError:
            print(f"[CONTROLADOR] Sin conexión al simulador. Reintentando...", flush=True)
            time.sleep(3)
        except Exception as e:
            print(f"[CONTROLADOR] Error loop: {e}", flush=True)
            time.sleep(3)


# ---------------------------------------------------------------
# ARRANQUE
# ---------------------------------------------------------------

if __name__ == '__main__':
    threading.Thread(target=logic_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)
