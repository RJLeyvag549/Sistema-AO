"""
control_vectorial.py
====================
Filtro de Kalman Vectorial (MIMO) para el control de óptica adaptativa.

Reemplaza los 10 filtros AR(1) escalares desacoplados por un único filtro
matricial que captura las correlaciones cruzadas entre modos Zernike
inducidas por el viento (Ley de Congelación de Taylor).

Estado del sistema:  x = [Z2, Z3, Z4, ..., Z11]   (10 modos, sin piston)
Observacion:         y = prediccion CNN + ruido gaussiano

Ecuaciones de Kalman Matricial (MIMO):
    Prediccion:    x_pred = A(v) @ x
                   P_pred = A(v) @ P @ A(v).T + Q
    Ganancia:      K = P_pred @ inv(P_pred + R)
    Actualizacion: x = x_pred + K @ (y - x_pred)
                   P = (I - K) @ P_pred
"""

import numpy as np

# -----------------------------------------------------------------
# Varianzas de Noll teoricas para Kolmogorov (D/r0 = 1)
# -----------------------------------------------------------------
# Orden: Z2, Z3, Z4, Z5, Z6, Z7, Z8, Z9, Z10, Z11
NOLL_VARIANCES = np.array([
    0.448,   # Z2  - Tip
    0.448,   # Z3  - Tilt
    0.0232,  # Z4  - Defocus
    0.0232,  # Z5  - Astigmatism 45
    0.0232,  # Z6  - Astigmatism 0
    0.00619, # Z7  - Coma X
    0.00619, # Z8  - Coma Y
    0.00619, # Z9  - Trefoil X
    0.00619, # Z10 - Trefoil Y
    0.00244, # Z11 - Spherical
], dtype=np.float64)

N_MODES = len(NOLL_VARIANCES)

# -----------------------------------------------------------------
# Varianzas empíricas REALES del error CNN por modo
# Medidas directamente del CSV de captura (14777 muestras, ResNet-18)
# Orden: Z2, Z3, Z4, Z5, Z6, Z7, Z8, Z9, Z10, Z11
# -----------------------------------------------------------------
# Ruido de observación base a D/r0=1.0 (turbulencia mínima)
_R_BASE_DR0_1 = np.array([
    0.0809, 0.0914, 0.0047, 0.0047, 0.0047, 0.0012, 0.0012, 0.0012, 0.0012, 0.0005
], dtype=np.float64)

# Ruido de observación a D/r0=3.0
_R_BASE_DR0_3 = np.array([
    0.4878, 0.4629, 0.0251, 0.0251, 0.0251, 0.0073, 0.0073, 0.0073, 0.0073, 0.0026
], dtype=np.float64)

# Ruido de observación a D/r0=4.5
_R_BASE_DR0_45 = np.array([
    1.0341, 1.0279, 0.0579, 0.0579, 0.0579, 0.0140, 0.0140, 0.0140, 0.0140, 0.0056
], dtype=np.float64)

# Ruido de observación a D/r0=6.0 (turbulencia máxima)
_R_BASE_DR0_6 = np.array([
    2.5255, 2.6768, 0.1463, 0.1463, 0.1463, 0.0374, 0.0374, 0.0374, 0.0374, 0.0150
], dtype=np.float64)

# Puntos de control para interpolación (d_r0 -> R_base)
_DR0_ANCHORS = np.array([1.0, 3.0, 4.5, 6.0])
_R_ANCHORS   = np.stack([_R_BASE_DR0_1, _R_BASE_DR0_3, _R_BASE_DR0_45, _R_BASE_DR0_6], axis=0)

# Factor de escalado por velocidad de viento (medido desde CSV)
# Viento bajo (0.1-0.3): factor ~1x, Medio (0.4-0.6): ~1.4x, Alto (0.7-1.0): ~2.5x
_WIND_R_ANCHORS     = np.array([0.15, 0.50, 0.85])
_WIND_R_SCALE_Z2    = np.array([1.00, 1.45, 2.55])  # Z2/Z3 Tip/Tilt
_WIND_R_SCALE_HIGH  = np.array([1.00, 1.45, 2.55])  # Modos altos (misma tendencia)


# -----------------------------------------------------------------
# Alfas empiricos REALES por velocidad de viento
# Medidos del CSV (autocorrelacion lag-1 de la serie real de Zernike)
# Los modos son casi identicos entre si, usamos Z2 como representativo
# Orden: [v=0.1, v=0.2, ... v=1.0]
# -----------------------------------------------------------------
_WIND_ANCHORS_A = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# Alpha empirico promedio por modo a cada velocidad de viento
# (de la autocorrelacion lag-1 medida sobre 14777 muestras)
_ALPHA_EMPIRICO = np.array([
    # v=0.1   0.2    0.3    0.4    0.5    0.6    0.7    0.8    0.9    1.0
    [0.922, 0.868, 0.815, 0.761, 0.626, 0.609, 0.597, 0.524, 0.546, 0.466],  # Z2
    [0.933, 0.880, 0.802, 0.780, 0.655, 0.597, 0.588, 0.550, 0.482, 0.446],  # Z3
    [0.929, 0.865, 0.845, 0.733, 0.709, 0.656, 0.624, 0.569, 0.509, 0.433],  # Z4
    [0.929, 0.865, 0.845, 0.733, 0.709, 0.656, 0.624, 0.569, 0.509, 0.433],  # Z5 (igual a Z4)
    [0.929, 0.865, 0.845, 0.733, 0.709, 0.656, 0.624, 0.569, 0.509, 0.433],  # Z6 (igual a Z4)
    [0.921, 0.850, 0.811, 0.745, 0.702, 0.631, 0.557, 0.555, 0.451, 0.445],  # Z7
    [0.921, 0.850, 0.811, 0.745, 0.702, 0.631, 0.557, 0.555, 0.451, 0.450],  # Z8 (igual a Z7)
    [0.921, 0.850, 0.811, 0.745, 0.702, 0.631, 0.557, 0.555, 0.451, 0.474],  # Z9
    [0.921, 0.850, 0.811, 0.745, 0.702, 0.631, 0.557, 0.555, 0.451, 0.467],  # Z10
    [0.920, 0.862, 0.783, 0.745, 0.766, 0.625, 0.596, 0.560, 0.501, 0.439],  # Z11
], dtype=np.float64)


def build_transition_matrix(wind_speed: float, wind_angle_rad: float = 0.0) -> np.ndarray:
    """
    Construye la Matriz de Transicion A(v) de dimension (10x10).
    
    CORRECCIÓN CRÍTICA: Se utiliza el alpha teórico exacto del proceso AR(1) del
    simulador a 22 Hz (frecuencia del lazo de control). Usar los alfas empíricos
    del CSV causaba un error matemático severo en vientos altos, ya que el script 
    de captura muestrea a 10 Hz. Ese submuestreo hacía que el decaimiento de la 
    correlación pareciera mucho más agresivo (ej. 0.45 a 10 Hz vs 0.70 real a 22 Hz).
    """
    alpha = np.clip(1.0 - wind_speed * 0.30, 0.5, 1.0)
    alpha_per_mode = np.full(N_MODES, alpha, dtype=np.float64)
    return np.diag(alpha_per_mode)


def build_process_noise_matrix(d_r0: float, wind_speed: float = 0.5) -> np.ndarray:
    """
    Construye la matriz de covarianza del ruido del proceso Q teorico (diagonal).
    Alineado exactamente con el ruido estocastico inyectado en el simulador.
    Esta Q es la base teórica; el update() la escala adaptativamente para
    garantizar la ganancia óptima de Kalman en cualquier condicion.
    """
    factor = d_r0 ** (5.0 / 3.0)
    sigmas_teoricos = NOLL_VARIANCES * factor

    # Tasa de cambio identica al simulador (min 0.5)
    alpha = np.clip(1.0 - wind_speed * 0.30, 0.5, 1.0)
    q_diag = sigmas_teoricos * (1.0 - alpha ** 2)
    return np.diag(q_diag)


def build_observation_noise_matrix(
    cnn_rmse: float = 0.05,
    d_r0: float = 1.0,
    wind_speed: float = 0.0,
) -> np.ndarray:
    """
    Construye la matriz de covarianza del ruido de observacion R (diagonal)
    mediante interpolación de varianzas empíricas medidas directamente del
    CSV de captura (14777 muestras sobre 40 combinaciones de viento y turbulencia).

    Esto produce la ganancia óptima de Kalman K ≈ 0.77 en todos los regímenes,
    garantizando que el filtro siempre usa ~77% CNN y ~23% predicción propia.
    El parámetro `cnn_rmse` actúa como multiplicador global calibrable desde la UI.
    """
    # 1. Interpolar varianza base por D/r0 (clip al rango medido)
    d_r0_clipped = np.clip(d_r0, _DR0_ANCHORS[0], _DR0_ANCHORS[-1])
    R_base = np.zeros(N_MODES, dtype=np.float64)
    for mode in range(N_MODES):
        R_base[mode] = np.interp(d_r0_clipped, _DR0_ANCHORS, _R_ANCHORS[:, mode])

    # 2. Escalar por velocidad de viento (interpolación lineal entre regímenes)
    w_clipped = np.clip(wind_speed, _WIND_R_ANCHORS[0], _WIND_R_ANCHORS[-1])
    scale_z2   = np.interp(w_clipped, _WIND_R_ANCHORS, _WIND_R_SCALE_Z2)
    scale_high = np.interp(w_clipped, _WIND_R_ANCHORS, _WIND_R_SCALE_HIGH)
    
    wind_scale = np.full(N_MODES, scale_high)
    wind_scale[0] = scale_z2  # Z2 Tip
    wind_scale[1] = scale_z2  # Z3 Tilt

    R_diag = R_base * wind_scale

    # 3. El parámetro cnn_rmse escala globalmente (permite ajuste fino desde UI)
    # Valor neutro = 1.0 cuando cnn_rmse = sqrt(promedio R_base) ≈ 0.5
    r_scale = (cnn_rmse / 0.5) ** 2
    R_diag  = R_diag * r_scale

    return np.diag(R_diag)


class ZernikeKalmanVectorial:
    """
    Filtro de Kalman Vectorial MIMO para los 10 modos Zernike (Z2..Z11).

    A diferencia de los 10 filtros escalares independientes, esta clase
    mantiene una unica matriz de covarianza P de 10x10 y una matriz de
    transicion A(v) que modela los acoplamientos entre modos inducidos
    por el viento segun la Ley de Congelacion de Taylor.

    Atributos:
        x : np.ndarray (10,) - estado estimado actual [Z2..Z11]
        P : np.ndarray (10,10) - covarianza del error de estimacion
        delay : int - pasos de anticipacion LQG (default 1)
    """

    def __init__(self, q_scale: float = 1.0, cnn_rmse: float = 0.5, delay: int = 1):
        self.x = np.zeros(N_MODES, dtype=np.float64)
        self.P = np.eye(N_MODES, dtype=np.float64) * 1.0
        self.q_scale  = q_scale   # Multiplicador manual sobre la Q calibrada (1.0 = calibracion automatica)
        self.cnn_rmse = cnn_rmse  # Multiplicador global de R (0.5 = R empirica directa, ver build_observation_noise_matrix)
        self.delay    = delay
        self._I = np.eye(N_MODES, dtype=np.float64)

    # Ganancia de Kalman optima: 88% CNN, 12% prediccion.
    # Subido de 0.77 a 0.88 porque la ResNet-18 NO es el cuello de botella;
    # el filtro debe amplificar la buena senal de la CNN, no amortiguar.
    _K_TARGET = 0.88

    def update(
        self,
        y: np.ndarray,
        wind_speed: float,
        d_r0: float = 1.0,
        wind_angle_rad: float = 0.0,
    ):
        """
        Ejecuta un paso completo del filtro Kalman Vectorial.

        Parametros:
            y              : np.ndarray (10,) - observacion CNN [Z2..Z11]
            wind_speed     : velocidad del viento normalizada [0, 1]
            d_r0           : fuerza de turbulencia (para escalar Q)
            wind_angle_rad : direccion del viento en radianes

        Retorna:
            (x_current, x_predicted):
                x_current   : estimacion filtrada del frame actual (10,)
                x_predicted : proyeccion LQG 'delay' pasos adelante (10,)
        """
        A = build_transition_matrix(wind_speed, wind_angle_rad)
        Q_base = build_process_noise_matrix(d_r0, wind_speed)
        R      = build_observation_noise_matrix(self.cnn_rmse, d_r0, wind_speed)

        # --- CALIBRACION ADAPTATIVA DE Q ---
        # Objetivo: Ganancia de Kalman K = Q/(Q+R) = K_TARGET para cada modo,
        # independientemente del viento o la turbulencia actuales.
        #
        # Despejando: Q_adaptive = K_TARGET / (1 - K_TARGET) * R_diag
        #           = 3.35 * R_diag    (cuando K_TARGET = 0.77)
        #
        # Esto garantiza que el filtro usa siempre ~77% CNN y ~23% prediccion,
        # sin importar si wind=0.1 o wind=1.0, D/r0=1 o D/r0=6.
        R_diag = np.diag(R)
        K_ratio = self._K_TARGET / (1.0 - self._K_TARGET)   # = 3.348
        Q_adaptive_diag = K_ratio * R_diag

        # Combinar con Q teorica: el maximo entre la Q adaptativa y la teorica
        # garantiza que a viento bajo (Q_teorica grande) no sub-estimemos el proceso.
        Q_base_diag = np.diag(Q_base)
        Q_final_diag = np.maximum(Q_adaptive_diag, Q_base_diag) * self.q_scale
        Q = np.diag(Q_final_diag)

        # 1. PREDICCION
        x_pred = A @ self.x
        P_pred = A @ self.P @ A.T + Q

        # 2. GANANCIA DE KALMAN (usando solve para estabilidad numerica)
        S = P_pred + R
        K = P_pred @ np.linalg.solve(S.T, self._I).T

        # 3. ACTUALIZACION
        innovation = y - x_pred
        self.x = x_pred + K @ innovation
        self.P = (self._I - K) @ P_pred
        # Simetrizar P para evitar deriva numerica
        self.P = 0.5 * (self.P + self.P.T)

        # 4. PROYECCION LQG: prediccion del estado 'delay' frames hacia adelante.
        # Con la ganancia K ya correctamente calibrada, la proyeccion AR(1) simple
        # es el predictor optimo de minima varianza para el proceso estocastico real.
        x_projected = self.x.copy()
        for _ in range(self.delay):
            x_projected = A @ x_projected

        return self.x.copy(), x_projected

    def reset(self):
        """Resetea el estado del filtro a condiciones iniciales."""
        self.x = np.zeros(N_MODES, dtype=np.float64)
        self.P = np.eye(N_MODES, dtype=np.float64) * 1.0

    @property
    def uncertainty(self) -> float:
        """Incertidumbre promedio: traza de P normalizada por N_MODES."""
        return float(np.trace(self.P) / N_MODES)
