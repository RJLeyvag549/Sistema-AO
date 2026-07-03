"""
benchmark_controlador.py
========================
Evaluacion standalone del Controlador Kalman Vectorial MIMO.

No requiere Docker. Importa directamente control_vectorial.py del controlador.

Modelo de evaluacion correcto (con delay de un ciclo):
    - En t: la CNN predice y_t = x_t + ruido_CNN
    - La correccion se aplica al SLM, que actua en t+1
    - Escenario SIN predictor LQG: se aplica y_t al frame t+1 → residual = x_{t+1} - y_t
    - Escenario CON predictor LQG: se aplica x̂_{t+1|t} al frame t+1 → residual = x_{t+1} - x̂_{t+1|t}

La ventaja del LQG es visible especialmente a altas velocidades de viento,
donde la turbulencia cambia significativamente entre frames consecutivos.

Uso:
    python scripts/benchmark_controlador.py

Resultados:
    - Tabla de métricas por escenario (consola)
    - scripts/resultados/benchmark_controlador_<timestamp>.csv
    - scripts/resultados/benchmark_controlador_<timestamp>.png
"""

import sys
import os
import numpy as np
import json
import csv
from datetime import datetime
from pathlib import Path

# ── Añadir controlador/ al path para importar control_vectorial ─────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'controlador'))

from control_vectorial import ZernikeKalmanVectorial, NOLL_VARIANCES, N_MODES

# ── Dependencias opcionales para plots ─────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARNING] matplotlib no disponible. Solo se generara el CSV.")

# ============================================================
# CONSTANTES DEL SISTEMA
# ============================================================

# RMSE tipico de la CNN segun el modelo seleccionado
CNN_RMSE_PROFILES = {
    'phase_diversity':  0.12,  # MAE ~0.30 → rmse estimado
    'resnet10':         0.08,  # Mas preciso
    'resnet18':         0.05,  # Val Loss 0.1286, rmse estimado
}

# Grilla de escenarios: viento × turbulencia
WIND_SPEEDS  = [0.1, 0.3, 0.5, 0.7, 0.9]          # [0=estatico, 1=rapido]
D_R0_VALUES  = [0.5, 1.0, 2.0, 3.5, 6.0]          # [debil → muy fuerte]

N_FRAMES     = 800    # frames por escenario (excl. warm-up)
WARMUP       = 50     # frames de calentamiento del filtro
DELAY        = 1      # latencia SLM (ciclos)
N_SEEDS      = 5      # repeticiones para reducir varianza estadistica


# ============================================================
# MODELOS DE SIMULACION
# ============================================================

def simulate_turbulence(n_frames: int, wind_speed: float, d_r0: float, seed: int = 0) -> np.ndarray:
    """
    Genera la evolucion temporal de los 10 modos Zernike (Z2..Z11)
    usando el modelo AR(1) de Kolmogorov, identico al simulador/main.py.

    alpha = 1 - wind_speed * 0.3   (decorrelacion temporal)
    sigma_i = sqrt(var_Noll_i * (D/r0)^(5/3)) * beta

    Retorna: np.ndarray (n_frames, N_MODES=10)
    """
    rng = np.random.default_rng(seed)

    alpha      = np.clip(1.0 - wind_speed * 0.30, 0.5, 1.0)
    beta       = np.sqrt(1.0 - alpha ** 2)
    factor_k   = d_r0 ** (5.0 / 3.0)
    sigmas     = np.sqrt(NOLL_VARIANCES * factor_k)

    x       = np.zeros(N_MODES)
    history = np.zeros((n_frames, N_MODES))

    for t in range(n_frames):
        noise    = rng.standard_normal(N_MODES)
        x        = alpha * x + sigmas * beta * noise
        history[t] = x

    return history


def simulate_cnn_observations(truth: np.ndarray, cnn_rmse: float, seed: int = 0) -> np.ndarray:
    """
    Simula las predicciones de la CNN sumando ruido gaussiano al ground truth.
    El ruido es independiente por modo y por frame.
    """
    rng = np.random.default_rng(seed + 1000)
    return truth + rng.normal(0.0, cnn_rmse, truth.shape)


# ============================================================
# EVALUACION DE UN ESCENARIO
# ============================================================

def evaluate_scenario(
    wind_speed: float,
    d_r0: float,
    cnn_rmse: float = 0.05,
    n_frames: int = N_FRAMES,
    warmup: int = WARMUP,
    delay: int = DELAY,
    seed: int = 0,
) -> dict:
    """
    Evalua el controlador LQG para un par (wind_speed, d_r0).

    Escenarios de corrección comparados:
    ─────────────────────────────────────
    A) Sin corrección:
       residual_t = x_t  (error sin aplicar ninguna correccion)

    B) CNN directa sin delay (irrealista, limite superior):
       residual_t = x_t - y_t  (CNN del mismo frame)

    C) CNN con delay 1 ciclo (realista, sin predictor):
       residual_t = x_t - y_{t-1}  (CNN del frame anterior)

    D) LQG con delay 1 ciclo (realista, con predictor Kalman):
       residual_t = x_t - x̂_{t|t-1}  (prediccion LQG calculada en t-1)

    La comparacion justa es C vs D, porque ambas aplican
    una correccion calculada un ciclo antes.
    """
    # Generar turbulencia real y observaciones CNN
    n_total = n_frames + delay + warmup
    truth   = simulate_turbulence(n_total, wind_speed, d_r0, seed)
    obs_cnn = simulate_cnn_observations(truth, cnn_rmse, seed)

    # Inicializar filtro Kalman vectorial
    kf = ZernikeKalmanVectorial(q_scale=1.0, cnn_rmse=cnn_rmse, delay=delay)

    # Buffers de metricas (post warm-up)
    mae_no_corr   = []
    mae_cnn_ideal = []   # sin delay (limite superior)
    mae_cnn_delay = []   # con delay (linea base realista)
    mae_lqg_delay = []   # con delay + Kalman LQG (objetivo)

    # Estado previo para el modelo de delay
    prev_cnn_obs  = np.zeros(N_MODES)  # y_{t-1}: CNN del frame anterior
    prev_lqg_pred = np.zeros(N_MODES)  # x̂_{t|t-1}: prediccion LQG del frame anterior

    for t in range(n_total - delay):
        y_t   = obs_cnn[t]       # observacion CNN en t
        x_tp1 = truth[t + delay] # estado REAL en t+1 (el que corriges)

        # Paso Kalman: procesa y_t, predice estado en t+delay
        x_current, x_predicted = kf.update(
            y=y_t,
            wind_speed=wind_speed,
            d_r0=d_r0,
        )

        if t >= warmup:
            # A) Sin correccion: el frente de onda en t+1 sin correccion alguna
            mae_no_corr.append(np.mean(np.abs(x_tp1)))

            # B) CNN directa ideal (sin delay, irrealista):
            #    CNN captura y corrige en el MISMO frame → residual = x_{t+1} - y_{t+1}
            y_tp1 = obs_cnn[t + delay]
            mae_cnn_ideal.append(np.mean(np.abs(x_tp1 - y_tp1)))

            # C) CNN con delay 1 ciclo (REALISTA, sin predictor):
            #    En el ciclo t la CNN predice y_t, el SLM lo aplica en t+1.
            #    residual = x_{t+1} - y_t
            #    (NO usar prev_cnn_obs: ese seria delay de 2 ciclos)
            mae_cnn_delay.append(np.mean(np.abs(x_tp1 - y_t)))

            # D) LQG con delay 1 ciclo (REALISTA, con predictor Kalman):
            #    kf.update(y_t, delay=1) ya devuelve x_predicted = A @ x̂_t,
            #    que es la prediccion de x_{t+1} calculada en este mismo ciclo t.
            #    residual = x_{t+1} - x̂_{t+1|t}
            #    (NO usar prev_lqg_pred: ese seria delay de 2 ciclos)
            mae_lqg_delay.append(np.mean(np.abs(x_tp1 - x_predicted)))

    def safe_mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    mae_a = safe_mean(mae_no_corr)
    mae_b = safe_mean(mae_cnn_ideal)
    mae_c = safe_mean(mae_cnn_delay)
    mae_d = safe_mean(mae_lqg_delay)

    # Mejora del LQG sobre CNN con delay (la comparacion justa)
    mejora_lqg_pct = 100.0 * (mae_c - mae_d) / mae_c if mae_c > 0 else 0.0

    # Ratio de Strehl aproximado: exp(-sigma^2), sigma ≈ MAE
    def strehl(mae):
        return float(np.exp(-(mae ** 2)))

    return {
        'wind_speed':          wind_speed,
        'd_r0':                d_r0,
        'cnn_rmse':            cnn_rmse,
        'mae_no_correction':   mae_a,
        'mae_cnn_ideal':       mae_b,
        'mae_cnn_delayed':     mae_c,
        'mae_lqg':             mae_d,
        'mejora_lqg_pct':      mejora_lqg_pct,
        'strehl_no_corr':      strehl(mae_a),
        'strehl_cnn_ideal':    strehl(mae_b),
        'strehl_cnn_delayed':  strehl(mae_c),
        'strehl_lqg':          strehl(mae_d),
    }


def evaluate_scenario_multiseed(wind_speed, d_r0, cnn_rmse, n_seeds=N_SEEDS) -> dict:
    """Promedia evaluate_scenario sobre multiples seeds para mayor robustez estadistica."""
    results = [evaluate_scenario(wind_speed, d_r0, cnn_rmse, seed=s) for s in range(n_seeds)]

    keys_float = [
        'mae_no_correction', 'mae_cnn_ideal', 'mae_cnn_delayed', 'mae_lqg',
        'mejora_lqg_pct',
        'strehl_no_corr', 'strehl_cnn_ideal', 'strehl_cnn_delayed', 'strehl_lqg',
    ]
    aggregated = {k: float(np.mean([r[k] for r in results])) for k in keys_float}
    aggregated['wind_speed'] = wind_speed
    aggregated['d_r0']       = d_r0
    aggregated['cnn_rmse']   = cnn_rmse
    return aggregated


# ============================================================
# ANALISIS TEMPORAL PARA UN ESCENARIO CRITICO
# ============================================================

def temporal_trace(wind_speed: float, d_r0: float, cnn_rmse: float, n_frames: int = 300):
    """
    Genera la traza temporal de los residuales para el escenario critico
    (wind_speed alto, d_r0 alto). Util para visualizacion.
    """
    n_total = n_frames + DELAY + WARMUP
    truth   = simulate_turbulence(n_total, wind_speed, d_r0, seed=0)
    obs_cnn = simulate_cnn_observations(truth, cnn_rmse, seed=0)

    kf = ZernikeKalmanVectorial(q_scale=1.0, cnn_rmse=cnn_rmse, delay=DELAY)

    trace_no_corr   = []
    trace_cnn_delay = []
    trace_lqg       = []
    trace_frames    = []

    prev_cnn_obs  = np.zeros(N_MODES)
    prev_lqg_pred = np.zeros(N_MODES)

    for t in range(n_total - DELAY):
        y_t   = obs_cnn[t]
        x_tp1 = truth[t + DELAY]

        _, x_predicted = kf.update(y=y_t, wind_speed=wind_speed, d_r0=d_r0)

        if t >= WARMUP:
            trace_no_corr.append(np.mean(np.abs(x_tp1)))
            trace_cnn_delay.append(np.mean(np.abs(x_tp1 - y_t)))
            trace_lqg.append(np.mean(np.abs(x_tp1 - x_predicted)))
            trace_frames.append(t - WARMUP)

    return np.array(trace_frames), np.array(trace_no_corr), np.array(trace_cnn_delay), np.array(trace_lqg)


# ============================================================
# GENERACION DE PLOTS
# ============================================================

def generate_plots(all_results: list, output_path: Path, cnn_model: str):
    """Genera el panel de figuras del benchmark."""
    if not HAS_MATPLOTLIB:
        return

    wind_speeds = sorted(set(r['wind_speed'] for r in all_results))
    d_r0_values = sorted(set(r['d_r0']       for r in all_results))

    # ── Matrices para heatmaps ──────────────────────────────
    def make_matrix(key):
        m = np.zeros((len(d_r0_values), len(wind_speeds)))
        for r in all_results:
            i = d_r0_values.index(r['d_r0'])
            j = wind_speeds.index(r['wind_speed'])
            m[i, j] = r[key]
        return m

    mat_mejora   = make_matrix('mejora_lqg_pct')
    mat_mae_lqg  = make_matrix('mae_lqg')
    mat_mae_cnn  = make_matrix('mae_cnn_delayed')
    mat_strehl   = make_matrix('strehl_lqg')

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0f0f14')

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    CMAP_GOOD  = 'RdYlGn'
    CMAP_ERR   = 'YlOrRd'
    CMAP_STR   = 'viridis'
    XT_LABELS  = [f'{v:.1f}' for v in wind_speeds]
    YT_LABELS  = [f'{v:.1f}' for v in d_r0_values]

    def style_ax(ax, title):
        ax.set_facecolor('#1a1a2e')
        ax.set_title(title, color='white', fontsize=10, pad=8, fontweight='bold')
        ax.set_xlabel('Velocidad de Viento', color='#94a3b8', fontsize=8)
        ax.set_ylabel('Fuerza Turbulencia (D/r₀)', color='#94a3b8', fontsize=8)
        ax.tick_params(colors='#94a3b8', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')

    def heatmap(ax, data, cmap, title, fmt='.1f', vmin=None, vmax=None):
        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax, origin='lower')
        ax.set_xticks(range(len(wind_speeds)))
        ax.set_xticklabels(XT_LABELS)
        ax.set_yticks(range(len(d_r0_values)))
        ax.set_yticklabels(YT_LABELS)
        style_ax(ax, title)
        for i in range(len(d_r0_values)):
            for j in range(len(wind_speeds)):
                ax.text(j, i, f'{data[i,j]:{fmt}}',
                        ha='center', va='center', fontsize=7.5,
                        color='white', fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors='#94a3b8', labelsize=6)

    # 1. Mejora LQG (%) sobre CNN con delay
    ax1 = fig.add_subplot(gs[0, 0])
    heatmap(ax1, mat_mejora, CMAP_GOOD,
            'Mejora LQG vs CNN+Delay (%)\n(+ es mejor)', fmt='.1f', vmin=0)

    # 2. MAE LQG
    ax2 = fig.add_subplot(gs[0, 1])
    heatmap(ax2, mat_mae_lqg, CMAP_ERR,
            'MAE Residual — Predictor LQG (rad)', fmt='.3f')

    # 3. MAE CNN con delay
    ax3 = fig.add_subplot(gs[0, 2])
    heatmap(ax3, mat_mae_cnn, CMAP_ERR,
            'MAE Residual — CNN+Delay (rad)', fmt='.3f')

    # 4. Strehl ratio LQG
    ax4 = fig.add_subplot(gs[1, 0])
    heatmap(ax4, mat_strehl, CMAP_STR,
            'Strehl Ratio — Predictor LQG\n(1.0 = corrección perfecta)', fmt='.3f', vmin=0, vmax=1)

    # 5. Corte por viento a D/r0 alto (peor turbulencia)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor('#1a1a2e')
    max_dr0 = max(d_r0_values)
    subset = [r for r in all_results if r['d_r0'] == max_dr0]
    subset.sort(key=lambda r: r['wind_speed'])
    ws_vals   = [r['wind_speed'] for r in subset]
    mae_cnn_v = [r['mae_cnn_delayed'] for r in subset]
    mae_lqg_v = [r['mae_lqg']        for r in subset]
    ax5.plot(ws_vals, mae_cnn_v, 'o-', color='#f59e0b', lw=2, label='CNN + delay')
    ax5.plot(ws_vals, mae_lqg_v, 's-', color='#10b981', lw=2, label='Predictor LQG')
    ax5.fill_between(ws_vals, mae_lqg_v, mae_cnn_v, alpha=0.15, color='#10b981')
    ax5.set_xlabel('Velocidad de Viento', color='#94a3b8', fontsize=8)
    ax5.set_ylabel('MAE residual (rad)', color='#94a3b8', fontsize=8)
    ax5.legend(fontsize=7, facecolor='#0f0f14', labelcolor='white', framealpha=0.8)
    ax5.tick_params(colors='#94a3b8', labelsize=7)
    ax5.grid(True, color='#334155', alpha=0.4)
    style_ax(ax5, f'Corte a D/r₀={max_dr0} — MAE vs Velocidad Viento')

    # 6. Traza temporal escenario critico (viento alto, turbulencia alta)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor('#1a1a2e')
    frames, tr_no, tr_cnn, tr_lqg = temporal_trace(
        max(wind_speeds), max_dr0,
        cnn_rmse=CNN_RMSE_PROFILES.get(cnn_model, 0.05),
        n_frames=200
    )
    ax6.plot(frames, tr_no,  '-', color='#ef4444', lw=1.2, alpha=0.7, label='Sin corrección')
    ax6.plot(frames, tr_cnn, '-', color='#f59e0b', lw=1.5, label='CNN + delay')
    ax6.plot(frames, tr_lqg, '-', color='#10b981', lw=2.0, label='Predictor LQG')
    ax6.set_xlabel('Frame', color='#94a3b8', fontsize=8)
    ax6.set_ylabel('MAE residual instantáneo (rad)', color='#94a3b8', fontsize=8)
    ax6.legend(fontsize=7, facecolor='#0f0f14', labelcolor='white', framealpha=0.8)
    ax6.tick_params(colors='#94a3b8', labelsize=7)
    ax6.grid(True, color='#334155', alpha=0.4)
    style_ax(ax6, f'Traza Temporal — Escenario Crítico\n(viento={max(wind_speeds):.1f}, D/r₀={max_dr0:.1f})')

    # Título global
    cnn_rmse_used = CNN_RMSE_PROFILES.get(cnn_model, 0.05)
    fig.suptitle(
        f'Benchmark Controlador LQG — Modelo: {cnn_model.upper()}   '
        f'| CNN RMSE={cnn_rmse_used:.3f} rad   '
        f'| {N_FRAMES} frames · {N_SEEDS} seeds · delay={DELAY} ciclo',
        color='white', fontsize=11, fontweight='bold', y=0.98
    )

    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0f0f14')
    plt.close(fig)
    print(f"[PLOT] Guardado en: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir   = Path(__file__).parent / 'resultados'
    out_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("  BENCHMARK CONTROLADOR KALMAN/LQG — Sistema AO")
    print(f"  Grilla: {len(WIND_SPEEDS)} velocidades × {len(D_R0_VALUES)} turbulencias")
    print(f"  Seeds:  {N_SEEDS}   Frames/escenario: {N_FRAMES}   Delay: {DELAY} ciclo")
    print("=" * 70)

    # Evaluar cada modelo CNN
    for cnn_model, cnn_rmse in CNN_RMSE_PROFILES.items():
        print(f"\n{'-'*70}")
        print(f"  Modelo CNN: {cnn_model.upper()}   (RMSE estimado = {cnn_rmse:.3f} rad)")
        print(f"{'-'*70}")
        print(f"  {'Viento':>7}  {'D/r0':>6}  {'MAE_nada':>9}  {'MAE_CNN+d':>9}  {'MAE_LQG':>9}  {'Mejora%':>8}  {'Strehl_LQG':>10}")
        print(f"  {'-'*7:7}  {'-'*6:6}  {'-'*9:9}  {'-'*9:9}  {'-'*9:9}  {'-'*8:8}  {'-'*10:10}")

        all_results = []

        for d_r0 in D_R0_VALUES:
            for ws in WIND_SPEEDS:
                r = evaluate_scenario_multiseed(ws, d_r0, cnn_rmse)
                all_results.append(r)

                sign = '+' if r['mejora_lqg_pct'] >= 0 else ''
                print(
                    f"  {ws:>7.2f}  {d_r0:>6.1f}  "
                    f"{r['mae_no_correction']:>9.4f}  "
                    f"{r['mae_cnn_delayed']:>9.4f}  "
                    f"{r['mae_lqg']:>9.4f}  "
                    f"{sign}{r['mejora_lqg_pct']:>7.1f}%  "
                    f"{r['strehl_lqg']:>10.4f}"
                )

        # Guardar CSV
        csv_path = out_dir / f'benchmark_controlador_{cnn_model}_{timestamp}.csv'
        fieldnames = list(all_results[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n  [CSV] Guardado en: {csv_path}")

        # Guardar JSON
        json_path = out_dir / f'benchmark_controlador_{cnn_model}_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        # Generar plots
        if HAS_MATPLOTLIB:
            png_path = out_dir / f'benchmark_controlador_{cnn_model}_{timestamp}.png'
            generate_plots(all_results, png_path, cnn_model)

        # Resumen del escenario mas critico
        worst = max(all_results, key=lambda r: r['d_r0'] + r['wind_speed'])
        print(f"\n  [ESCENARIO CRITICO] viento={worst['wind_speed']:.1f}, D/r0={worst['d_r0']:.1f}")
        print(f"    MAE sin corrección : {worst['mae_no_correction']:.4f} rad")
        print(f"    MAE CNN + delay    : {worst['mae_cnn_delayed']:.4f} rad")
        print(f"    MAE LQG predictor  : {worst['mae_lqg']:.4f} rad")
        print(f"    Mejora LQG vs CNN  : {worst['mejora_lqg_pct']:+.1f}%")
        print(f"    Strehl LQG         : {worst['strehl_lqg']:.4f}")

    print("\n" + "=" * 70)
    print("  Benchmark completado.")
    print("=" * 70)


if __name__ == '__main__':
    main()
