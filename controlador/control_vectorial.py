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


def build_transition_matrix(wind_speed: float, wind_angle_rad: float = 0.0) -> np.ndarray:
    """
    Construye la Matriz de Transicion A(v, theta) de dimension (10x10).
    Alineado con el simulador: alpha = 1 - wind_speed * 0.3 (limites [0.5, 1.0])
    """
    radial_orders = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3, 4], dtype=np.float64)
    
    # Coeficiente AR(1) base del simulador
    alpha_base = np.clip(1.0 - wind_speed * 0.30, 0.5, 1.0)
    
    # Escalado por orden radial para decaimientos mas rapidos en frecuencias espaciales altas
    # Z2, Z3 decaen a ritmo alpha_base, Z4..Z6 mas rapido (alpha_base^radial_order)
    alpha_diag = alpha_base ** (radial_orders / 2.0)
    A = np.diag(alpha_diag)

    # Acoplamiento fuera de la diagonal por el viento (Ley de Taylor)
    coupling_strength = wind_speed * 0.05

    # Par Tip-Tilt: el viento proyecta energia entre Z2 y Z3 segun su angulo
    A[0, 1] +=  coupling_strength * np.cos(wind_angle_rad)
    A[1, 0] += -coupling_strength * np.cos(wind_angle_rad)

    # Par Astigmatismo: Z5 <-> Z6
    A[3, 4] += coupling_strength * np.sin(2 * wind_angle_rad) * 0.3
    A[4, 3] += coupling_strength * np.sin(2 * wind_angle_rad) * 0.3

    # Par Coma: Z7 <-> Z8
    A[5, 6] += coupling_strength * np.cos(wind_angle_rad) * 0.2
    A[6, 5] += coupling_strength * np.cos(wind_angle_rad) * 0.2

    return A


def build_process_noise_matrix(d_r0: float, wind_speed: float = 0.5) -> np.ndarray:
    """
    Construye la matriz de covarianza del ruido del proceso Q (diagonal).
    Q = sigmas^2 * (1 - alpha^2)

    Parametros:
        d_r0       : fuerza de turbulencia, tipicamente en [0.5, 6.0]
        wind_speed : velocidad de viento para estimar la tasa de cambio de turbulencia
    """
    factor = d_r0 ** (5.0 / 3.0)
    sigmas_teoricos = NOLL_VARIANCES * factor

    # Tasa de cambio de la turbulencia
    radial_orders = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3, 4], dtype=np.float64)
    alpha_base = np.clip(1.0 - wind_speed * 0.30, 0.5, 1.0)
    alpha_diag = alpha_base ** (radial_orders / 2.0)
    
    # Ruido del proceso teorico inducido por la transicion temporal AR(1)
    q_diag = sigmas_teoricos * (1.0 - alpha_diag ** 2)
    return np.diag(q_diag)


def build_observation_noise_matrix(cnn_rmse: float = 0.05, d_r0: float = 1.0) -> np.ndarray:
    """
    Construye la matriz de covarianza del ruido de observacion R (diagonal).
    Ruido independiente por modo con magnitud igual al RMSE tipico de la CNN.
    Bajo alta turbulencia, la CNN pierde el rastro en Tip y Tilt, por lo que 
    se les asigna una menor confianza (mayor covarianza).
    """
    R = np.eye(N_MODES, dtype=np.float64) * (cnn_rmse ** 2)
    
    # Penalizacion heuristica para Tip (Z2) y Tilt (Z3) cuando d_r0 es alto
    if d_r0 > 3.0:
        penalty = (d_r0 - 3.0) * 0.8  # Error extra en radianes por turbulencia extrema
        R[0, 0] = (cnn_rmse + penalty) ** 2
        R[1, 1] = (cnn_rmse + penalty) ** 2
        
    return R


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

    def __init__(self, q_scale: float = 0.002, cnn_rmse: float = 0.05, delay: int = 1):
        self.x = np.zeros(N_MODES, dtype=np.float64)
        self.P = np.eye(N_MODES, dtype=np.float64) * 1.0
        self.q_scale = q_scale
        self.cnn_rmse = cnn_rmse
        self.delay = delay
        self._I = np.eye(N_MODES, dtype=np.float64)

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
        Q = build_process_noise_matrix(d_r0, wind_speed) * self.q_scale
        R = build_observation_noise_matrix(self.cnn_rmse, d_r0)

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

        # 4. PROYECCION LQG: x(t+k) = A^k @ x(t)
        x_predicted = self.x.copy()
        for _ in range(self.delay):
            x_predicted = A @ x_predicted

        return self.x.copy(), x_predicted

    def reset(self):
        """Resetea el estado del filtro a condiciones iniciales."""
        self.x = np.zeros(N_MODES, dtype=np.float64)
        self.P = np.eye(N_MODES, dtype=np.float64) * 1.0

    @property
    def uncertainty(self) -> float:
        """Incertidumbre promedio: traza de P normalizada por N_MODES."""
        return float(np.trace(self.P) / N_MODES)
