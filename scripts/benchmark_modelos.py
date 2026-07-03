#!/usr/bin/env python3
"""
benchmark_modelos.py
====================
Compara los 3 modelos CNN del sistema de Óptica Adaptativa:
  - Modelo A    (phase_diversity)
  - ResNet-10
  - ResNet-18

En 9 condiciones: D/r0 ∈ {0.5, 2.0, 4.0} × Viento ∈ {0.1, 0.5, 0.9}

Requiere los contenedores Docker corriendo:
    docker-compose up -d   (desde c:\\Sistema-AO)

Dependencias externas (instalar si faltan):
    pip install requests numpy pandas openpyxl matplotlib seaborn
"""

import time
import sys
import numpy as np
import requests
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime

# ── Seaborn es opcional; se usa solo para heatmaps ───────────────
try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False
    print("[AVISO] seaborn no instalado — heatmaps desactivados. "
          "Instala con: pip install seaborn")

matplotlib.rcParams["font.family"] = "DejaVu Sans"

# ── Configuración general ─────────────────────────────────────────
SIM_URL = "http://localhost:5000"   # Simulador Prysm
INF_URL = "http://localhost:5001"   # Servicio de inferencia CNN

MODELS = ["phase_diversity", "resnet10", "resnet18"]
MODEL_LABELS = {
    "phase_diversity": "Modelo A\n(Phase Diversity)",
    "resnet10":        "ResNet-10",
    "resnet18":        "ResNet-18",
}
MODEL_COLORS = {
    "phase_diversity": "#818cf8",   # índigo claro
    "resnet10":        "#34d399",   # verde esmeralda
    "resnet18":        "#fb923c",   # naranja
}

# Condiciones del benchmark
D_R0_VALUES  = [0.5, 2.0, 4.0]   # Turbulencia: Baja / Media / Alta
WIND_SPEEDS  = [0.1, 0.5, 0.9]   # Viento:      Bajo / Medio / Alto
TURB_LABELS  = {0.5: "Baja (0.5)", 2.0: "Media (2.0)", 4.0: "Alta (4.0)"}
WIND_LABELS  = {0.1: "Bajo (0.1)", 0.5: "Medio (0.5)", 0.9: "Alto (0.9)"}

N_SAMPLES     = 25    # Muestras por condición × modelo
WARMUP_S      = 3.0   # Segundos de estabilización al cambiar condición
SAMPLE_WAIT_S = 0.15  # Pausa entre muestras (> 45ms del bucle estocástico)

# Directorio de salida
OUTPUT_DIR = Path(__file__).parent / "benchmark_results"
OUTPUT_DIR.mkdir(exist_ok=True)
TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")


# ════════════════════════════════════════════════════════════════
#  UTILIDADES DE RED
# ════════════════════════════════════════════════════════════════

def check_services():
    """Verifica que el simulador e inferencia sean accesibles.
    Además muestra qué modelos están cargados en inferencia."""
    print("\nVerificando servicios Docker...")
    for name, url in [("Simulador ", SIM_URL), ("Inferencia", INF_URL)]:
        try:
            r = requests.get(f"{url}/status", timeout=4)
            r.raise_for_status()
            print(f"  [OK] {name} -> {url}")
        except Exception as exc:
            print(f"  [ERROR] {name} NO DISPONIBLE -> {exc}")
            sys.exit(1)

    # Mostrar estado de modelos cargados
    try:
        r = requests.get(f"{INF_URL}/status", timeout=4)
        loaded = r.json().get("models_loaded", {})
        print("\n  Estado de modelos en inferencia:")
        for m in MODELS:
            estado = "[OK] cargado" if loaded.get(m) else "[ERR] NO CARGADO (pesos no encontrados)"
            print(f"    {m:25s} {estado}")
        not_loaded = [m for m in MODELS if not loaded.get(m)]
        if not_loaded:
            print(f"\n  [ADVERTENCIA] {not_loaded} no estan cargados.")
            print("  El benchmark correra igualmente, pero esos modelos daran errores altos.")
    except Exception:
        pass


def set_condition(d_r0: float, wind_speed: float):
    """Configura el simulador en modo estocástico con los parámetros dados."""
    requests.post(
        f"{SIM_URL}/config",
        json={"method": "2", "d_r0": d_r0, "wind_speed": wind_speed},
        timeout=4,
    )


def restore_simulador():
    """Restaura el simulador a modo manual al terminar el benchmark."""
    try:
        requests.post(f"{SIM_URL}/config", json={"method": "1"}, timeout=4)
    except Exception:
        pass


def get_ground_truth() -> dict | None:
    """Devuelve los coeficientes Zernike reales del simulador."""
    try:
        r = requests.get(f"{SIM_URL}/status", timeout=2)
        if r.status_code == 200:
            return r.json().get("state", {}).get("zernikes")
    except Exception:
        pass
    return None


def get_prediction(model: str) -> list | None:
    """Llama a GET /predict en inferencia (lee el psf.npy del volumen compartido)."""
    try:
        r = requests.get(f"{INF_URL}/predict?model={model}", timeout=4)
        if r.status_code == 200:
            return r.json().get("zernike")
    except Exception:
        pass
    return None


def compute_errors(truth: dict, pred: list) -> dict:
    """
    Calcula MAE, RMSE y error máximo entre Zernikes reales y predichos.
    Excluye Z1 (pistón) que no contribuye a la corrección.
    """
    t = np.array([truth.get(f"Z{i}", 0.0) for i in range(2, 12)])  # Z2..Z11
    p = np.array(pred[1:11])                                          # ídem
    diff = t - p
    return {
        "mae":     float(np.mean(np.abs(diff))),
        "rmse":    float(np.sqrt(np.mean(diff ** 2))),
        "max_err": float(np.max(np.abs(diff))),
    }


# ════════════════════════════════════════════════════════════════
#  BENCHMARK PRINCIPAL
# ════════════════════════════════════════════════════════════════

def run_benchmark() -> pd.DataFrame:
    records = []
    n_cond  = len(D_R0_VALUES) * len(WIND_SPEEDS)
    idx     = 0

    total_est = n_cond * (WARMUP_S + N_SAMPLES * (SAMPLE_WAIT_S + 0.07) * len(MODELS))
    print(f"\nTiempo estimado: ~{total_est / 60:.1f} minutos\n")
    print("-" * 62)

    for d_r0 in D_R0_VALUES:
        for wind_speed in WIND_SPEEDS:
            idx += 1
            turb_str = TURB_LABELS[d_r0]
            wind_str = WIND_LABELS[wind_speed]
            print(f"\n[{idx}/{n_cond}] Turbulencia: {turb_str}  |  Viento: {wind_str}")
            print(f"  Configurando simulador y esperando {WARMUP_S:.0f}s...", end="", flush=True)
            set_condition(d_r0, wind_speed)
            time.sleep(WARMUP_S)
            print(" listo")

            # Buffers de errores por modelo
            buffers: dict[str, dict[str, list]] = {
                m: {"mae": [], "rmse": [], "max_err": []} for m in MODELS
            }

            for s in range(N_SAMPLES):
                # Obtener estado real (ground truth)
                truth = get_ground_truth()
                if truth is None:
                    time.sleep(SAMPLE_WAIT_S)
                    continue

                # Pequeño gap para que psf.npy esté actualizado tras el /status
                time.sleep(0.04)

                # Obtener predicciones de los 3 modelos
                for model in MODELS:
                    pred = get_prediction(model)
                    if pred is not None and len(pred) >= 11:
                        errs = compute_errors(truth, pred)
                        for k in ("mae", "rmse", "max_err"):
                            buffers[model][k].append(errs[k])

                time.sleep(SAMPLE_WAIT_S)
                print(f"  Muestra {s + 1:>2}/{N_SAMPLES}", end="\r", flush=True)

            print()  # salto de línea tras el \r

            # Consolidar estadísticas por modelo
            for model in MODELS:
                n = len(buffers[model]["mae"])
                if n == 0:
                    print(f"  {model}: SIN DATOS (modelo no cargado)")
                    continue

                mae_arr  = np.array(buffers[model]["mae"])
                rmse_arr = np.array(buffers[model]["rmse"])
                maxe_arr = np.array(buffers[model]["max_err"])

                records.append({
                    "d_r0":         d_r0,
                    "turbulencia":  turb_str,
                    "wind_speed":   wind_speed,
                    "viento":       wind_str,
                    "modelo":       model,
                    "label":        MODEL_LABELS[model].replace("\n", " "),
                    "n_muestras":   n,
                    "mae_mean":     float(np.mean(mae_arr)),
                    "mae_std":      float(np.std(mae_arr)),
                    "rmse_mean":    float(np.mean(rmse_arr)),
                    "rmse_std":     float(np.std(rmse_arr)),
                    "max_err_mean": float(np.mean(maxe_arr)),
                })

                label_short = MODEL_LABELS[model].replace("\n", " ")
                print(f"  {label_short:30s}  "
                      f"MAE={np.mean(mae_arr):.4f}+-{np.std(mae_arr):.4f}  "
                      f"RMSE={np.mean(rmse_arr):.4f}")

    restore_simulador()
    return pd.DataFrame(records)


# ================================================================
#  EXPORTAR EXCEL
# ================================================================

def export_excel(df: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / f"benchmark_{TIMESTAMP}.xlsx"

    with pd.ExcelWriter(path, engine="openpyxl") as writer:

        # -- Hoja 1: MAE pivot ----------------------------------
        pivot_mae = df.pivot_table(
            index=["turbulencia", "viento"],
            columns="label",
            values="mae_mean",
        ).round(4)
        pivot_mae.index.names = ["Turbulencia (D/r0)", "Velocidad Viento"]
        pivot_mae.columns.name = "MAE medio (rad)  menor = mejor"
        pivot_mae.to_excel(writer, sheet_name="MAE (menor = mejor)")

        # -- Hoja 2: RMSE pivot ---------------------------------
        pivot_rmse = df.pivot_table(
            index=["turbulencia", "viento"],
            columns="label",
            values="rmse_mean",
        ).round(4)
        pivot_rmse.index.names = ["Turbulencia (D/r0)", "Velocidad Viento"]
        pivot_rmse.columns.name = "RMSE medio (rad)  menor = mejor"
        pivot_rmse.to_excel(writer, sheet_name="RMSE (menor = mejor)")

        # -- Hoja 3: Mejor modelo por condicion ----------------
        best = df.loc[df.groupby(["d_r0", "wind_speed"])["mae_mean"].idxmin()][
            ["turbulencia", "viento", "label", "mae_mean", "rmse_mean"]
        ].rename(columns={
            "turbulencia": "Turbulencia",
            "viento":      "Viento",
            "label":       "Mejor Modelo",
            "mae_mean":    "MAE (rad)",
            "rmse_mean":   "RMSE (rad)",
        })
        best.to_excel(writer, sheet_name="Mejor Modelo por Condicion", index=False)

        # -- Hoja 4: Datos completos ----------------------------
        df.to_excel(writer, sheet_name="Datos Completos", index=False)

    print(f"\n  [OK] Excel guardado: {path}")
    return path



# ════════════════════════════════════════════════════════════════
#  GRÁFICAS
# ════════════════════════════════════════════════════════════════

BG_DARK  = "#0d0d0d"
BG_PANEL = "#1a1a1a"
TXT_CLR  = "#e4e4e7"
GRID_CLR = "#2a2a2a"

TURB_COLORS = {
    "Baja (0.5)":  "#4ade80",
    "Media (2.0)": "#facc15",
    "Alta (4.0)":  "#f87171",
}


def _style_ax(ax):
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=TXT_CLR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.grid(axis="y", color=GRID_CLR, linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)


def plot_results(df: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(20, 18))
    fig.patch.set_facecolor(BG_DARK)

    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.55, wspace=0.38,
        left=0.06, right=0.97, top=0.92, bottom=0.05,
    )

    # ── Fila 0: Barras MAE por modelo (un subplot por modelo) ──
    wind_vals  = WIND_SPEEDS
    wind_ticks = [WIND_LABELS[w].split()[0] for w in wind_vals]
    x          = np.arange(len(wind_vals))
    bar_w      = 0.24

    for col, model in enumerate(MODELS):
        ax = fig.add_subplot(gs[0, col])
        _style_ax(ax)
        sub = df[df["modelo"] == model]

        for i, d_r0 in enumerate(D_R0_VALUES):
            turb_key = TURB_LABELS[d_r0]
            vals, errs = [], []
            for wv in wind_vals:
                row = sub[(sub["d_r0"] == d_r0) & (sub["wind_speed"] == wv)]
                if len(row):
                    vals.append(row["mae_mean"].values[0])
                    errs.append(row["mae_std"].values[0])
                else:
                    vals.append(0); errs.append(0)

            offset = (i - 1) * bar_w
            ax.bar(
                x + offset, vals, bar_w,
                label=f"D/r0 = {d_r0}",
                color=TURB_COLORS[turb_key],
                alpha=0.85,
                yerr=errs, capsize=3,
                error_kw={"ecolor": "#ffffff55", "elinewidth": 1},
            )

        ax.set_title(
            MODEL_LABELS[model].replace("\n", " - "),
            color=MODEL_COLORS[model], fontsize=11, fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"Viento\n{t}" for t in wind_ticks], color=TXT_CLR, fontsize=8)
        ax.set_ylabel("MAE (rad)", color=TXT_CLR, fontsize=9)
        ax.legend(
            fontsize=7.5, labelcolor=TXT_CLR,
            facecolor="#222", edgecolor="#444", framealpha=0.85,
        )

    # ── Fila 1: Heatmaps MAE — LOS 3 MODELOS ─────────────────────
    if _HAS_SNS:
        for col, model in enumerate(MODELS):   # <-- ahora los 3 modelos
            ax = fig.add_subplot(gs[1, col])
            sub = df[df["modelo"] == model].copy()
            sub["turb_short"] = sub["turbulencia"].apply(lambda s: s.split()[0])
            sub["wind_short"] = sub["viento"].apply(lambda s: s.split()[0])

            pivot = sub.pivot_table(
                index="turb_short", columns="wind_short", values="mae_mean"
            ).reindex(
                index=["Baja", "Media", "Alta"],
                columns=["Bajo", "Medio", "Alto"],
            )

            # Escala de color individual por modelo para máxima claridad
            sns.heatmap(
                pivot, ax=ax,
                annot=True, fmt=".3f",
                cmap="YlOrRd",
                linewidths=0.5, linecolor="#333",
                annot_kws={"size": 9, "weight": "bold", "color": "#111"},
                cbar_kws={"label": "MAE (rad)", "shrink": 0.85},
            )
            ax.set_title(
                f"Heatmap MAE\n{MODEL_LABELS[model].replace(chr(10), ' ')}",
                color=MODEL_COLORS[model], fontsize=10, fontweight="bold",
            )
            ax.set_xlabel("Velocidad de Viento", color=TXT_CLR, fontsize=9)
            ax.set_ylabel("Turbulencia (D/r0)", color=TXT_CLR, fontsize=9)
            ax.tick_params(colors=TXT_CLR, labelsize=8)
            ax.figure.axes[-1].tick_params(colors=TXT_CLR, labelsize=7)
            ax.figure.axes[-1].yaxis.label.set_color(TXT_CLR)

    # ── Fila 2: Líneas MAE vs D/r0 a lo ancho (3 columnas) ───────
    ax_line = fig.add_subplot(gs[2, :])   # span de las 3 columnas
    _style_ax(ax_line)
    ax_line.grid(axis="both", color=GRID_CLR, linewidth=0.5, linestyle="--")

    for model in MODELS:
        sub = df[df["modelo"] == model]
        grp = sub.groupby("d_r0").agg(
            mae_m=("mae_mean", "mean"),
            mae_s=("mae_std",  "mean"),
        ).reset_index()

        ax_line.plot(
            grp["d_r0"], grp["mae_m"],
            marker="o", lw=2.5,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model].replace("\n", " "),
            zorder=3,
        )
        ax_line.fill_between(
            grp["d_r0"],
            grp["mae_m"] - grp["mae_s"],
            grp["mae_m"] + grp["mae_s"],
            color=MODEL_COLORS[model], alpha=0.12,
        )

    ax_line.set_title(
        "MAE Promedio vs Fuerza de Turbulencia  (promedio sobre velocidades de viento)",
        color=TXT_CLR, fontsize=11, fontweight="bold",
    )
    ax_line.set_xlabel("D/r0  (fuerza de turbulencia)", color=TXT_CLR, fontsize=9)
    ax_line.set_ylabel("MAE promedio (rad)", color=TXT_CLR, fontsize=9)
    ax_line.legend(
        fontsize=9, labelcolor=TXT_CLR,
        facecolor="#222", edgecolor="#444", framealpha=0.85,
        loc="upper left",
    )
    ax_line.set_xticks(D_R0_VALUES)

    # ── Titulo global ─────────────────────────────────────────────
    fig.suptitle(
        "Comparacion de Modelos CNN — Sistema de Optica Adaptativa\n"
        f"D/r0 = {D_R0_VALUES}  |  Viento = {WIND_SPEEDS}  |  "
        f"{N_SAMPLES} muestras/condicion  |  {TIMESTAMP}",
        color=TXT_CLR, fontsize=12, fontweight="bold", y=0.97,
    )

    path = OUTPUT_DIR / f"benchmark_{TIMESTAMP}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close()
    print(f"  [OK] Grafico guardado: {path}")
    return path





def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 62)
    print("  RESUMEN GLOBAL  (MAE promedio sobre todas las condiciones)")
    print("=" * 62)
    summary = (
        df.groupby("modelo")
        .agg(
            MAE_global=("mae_mean", "mean"),
            RMSE_global=("rmse_mean", "mean"),
            Mejor_en=("mae_mean", lambda s: s.idxmin()),
        )
        .sort_values("MAE_global")
    )
    for model, row in summary.iterrows():
        star = " <-- mejor" if row["MAE_global"] == summary["MAE_global"].min() else ""
        print(f"  {MODEL_LABELS[model].replace(chr(10),' '):30s}  "
              f"MAE={row['MAE_global']:.4f}  RMSE={row['RMSE_global']:.4f}{star}")
    print("=" * 62)


# ════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("+----------------------------------------------------------+")
    print("|   BENCHMARK COMPARATIVO -- MODELOS CNN AO                |")
    print("+----------------------------------------------------------+")

    check_services()

    n_cond = len(D_R0_VALUES) * len(WIND_SPEEDS)
    total_est = n_cond * (WARMUP_S + N_SAMPLES * (SAMPLE_WAIT_S + 0.07) * len(MODELS))
    print(f"\n  Condiciones:   D/r0 = {D_R0_VALUES}")
    print(f"  Viento:        {WIND_SPEEDS}")
    print(f"  Modelos:       {MODELS}")
    print(f"  Muestras/cond: {N_SAMPLES}  (total inferencias ~ {n_cond * N_SAMPLES * len(MODELS)})")
    print(f"  Tiempo est.:   ~{total_est / 60:.1f} min")
    print(f"  Salida:        {OUTPUT_DIR}")

    try:
        df = run_benchmark()
    except KeyboardInterrupt:
        print("\n\n[INTERRUMPIDO] Restaurando simulador a modo manual...")
        restore_simulador()
        sys.exit(0)

    if df.empty:
        print("\n[ERROR] No se recopilaron datos. Verifica los contenedores.")
        sys.exit(1)

    print("\nExportando resultados...")
    excel_path = export_excel(df)
    plot_path  = plot_results(df)
    print_summary(df)

    print(f"\n  Excel:   {excel_path}")
    print(f"  Grafico: {plot_path}\n")
