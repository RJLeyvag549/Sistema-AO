# Especificaciones de Modelos de Redes Neuronales de Óptica Adaptativa

Este documento describe la topología, hiperparámetros, activaciones y regresiones de los tres modelos de red neuronal del repositorio, así como su integración con el controlador predictivo.

---

## 1. Tabla Comparativa General

| Característica | Modelo A (Baseline/Phase Diversity) | ResNet-10 | ResNet-18 |
| :--- | :---: | :---: | :---: |
| **Canales de Entrada** | 2 (Enfocado / Desenfocado) | 2 (Enfocado / Desenfocado) | 2 (Enfocado / Desenfocado) |
| **Dimensión de Entrada** | 96 x 96 píxeles | 96 x 96 píxeles | 96 x 96 píxeles |
| **Capas Convolucionales** | 3 | 9 | 17 |
| **Bloques Residuales** | 0 | 4 (1 por etapa) | 8 (2 por etapa) |
| **Normalización** | Ninguna | BatchNorm2d en cada bloque | BatchNorm2d en cada bloque |
| **Activaciones** | LeakyReLU (pendiente = 0.1) | **ReLU** (use_leaky=False) | **LeakyReLU** (use_leaky=True) |
| **Reducción Espacial** | MaxPool2d (2 x 2, Stride 2) | Stride = 2 en convoluciones | Stride = 2 en convoluciones |
| **Agrupación Final** | Aplanado directo (12 x 12 x 128) | Adaptive Average Pooling (1 x 1) | Adaptive Average Pooling (1 x 1) |
| **Regularización** | Dropout (probabilidad = 0.2) | Ninguna | Ninguna |
| **Salida (Regresión)** | 11 Coeficientes Zernike (Z1-Z11) | 11 Coeficientes Zernike (Z1-Z11) | 11 Coeficientes Zernike (Z1-Z11) |
| **RMSE Típico (Filtro)** | **0.120 rad** | **0.080 rad** | **0.050 rad** |

---

## 2. Estructura Detallada: Modelo A (Phase Diversity)
*   **Entrada**: 2 canales x 96 x 96 píxeles.
*   **Capa Convolucional 1**: 32 filtros, kernel de 5 x 5, stride 1, padding 2.
    *   Activación: LeakyReLU (pendiente 0.1).
    *   Reducción: MaxPool2d (kernel 2 x 2, stride 2) -> Salida: 32 x 48 x 48.
*   **Capa Convolucional 2**: 64 filtros, kernel de 3 x 3, stride 1, padding 1.
    *   Activación: LeakyReLU (pendiente 0.1).
    *   Reducción: MaxPool2d (kernel 2 x 2, stride 2) -> Salida: 64 x 24 x 24.
*   **Capa Convolucional 3**: 128 filtros, kernel de 3 x 3, stride 1, padding 1.
    *   Activación: LeakyReLU (pendiente 0.1).
    *   Reducción: MaxPool2d (kernel 2 x 2, stride 2) -> Salida: 128 x 12 x 12.
*   **Capas Totalmente Conectadas (FC/Regresión)**:
    *   Aplanado del tensor a un vector de 18,432 elementos (12 x 12 x 128).
    *   Capa densa 1 (FC1): 18,432 entradas -> 256 salidas con LeakyReLU.
    *   Capa de Dropout: tasa de 0.2 (20% de probabilidad de descarte).
    *   Capa densa 2 (Salida): 256 entradas -> 11 coeficientes Zernike.

---

## 3. Estructura Detallada: ResNet-10
*   **Entrada**: 2 canales x 96 x 96 píxeles.
*   **Capa Convolucional Inicial**: 32 filtros, kernel de 7 x 7, stride 2, padding 3 -> BatchNorm2d -> ReLU -> MaxPool2d (kernel 3 x 3, stride 2, padding 1) -> Salida: 32 x 24 x 24.
*   **Bloques Residuales (BasicBlock)**:
    *   Cada bloque consta de: Conv 3x3 -> BatchNorm2d -> **ReLU** -> Conv 3x3 -> BatchNorm2d -> Suma con el atajo (shortcut) -> **ReLU**.
    *   El atajo (shortcut) iguala dimensiones con una Conv 1x1 y BatchNorm2d si cambian los canales o el tamaño espacial.
    *   **Nota de Compatibilidad**: Este modelo utiliza ReLU estándar. El uso de LeakyReLU en este modelo causaba distorsiones estáticas graves debido a la incompatibilidad con los pesos originales entrenados.
*   **Etapas Convolucionales**:
    *   **Layer 1**: 1 bloque de 32 canales, stride 1. Salida: 32 x 24 x 24.
    *   **Layer 2**: 1 bloque de 64 canales, stride 2. Salida: 64 x 12 x 12.
    *   **Layer 3**: 1 bloque de 128 canales, stride 2. Salida: 128 x 6 x 6.
    *   **Layer 4**: 1 bloque de 256 canales, stride 2. Salida: 256 x 3 x 3.
*   **Capa de Regresión**:
    *   Adaptive Average Pooling a tamaño 1 x 1 -> Salida: 256 x 1 x 1.
    *   Aplanado a vector de 256 elementos.
    *   Capa lineal final: 256 entradas -> 11 salidas.

---

## 4. Estructura Detallada: ResNet-18
*   **Entrada**: 2 canales x 96 x 96 píxeles.
*   **Capa Convolucional Inicial**: 64 filtros, kernel de 7 x 7, stride 2, padding 3 -> BatchNorm2d -> **LeakyReLU** -> MaxPool2d (kernel 3 x 3, stride 2, padding 1) -> Salida: 64 x 24 x 24.
*   **Bloques Residuales (BasicBlock)**:
    *   8 bloques en total (BasicBlock de doble Conv 3x3).
    *   **LeakyReLU** activado (`use_leaky=True`): Conv 3x3 -> BatchNorm2d -> LeakyReLU -> Conv 3x3 -> BatchNorm2d -> Suma con el atajo -> LeakyReLU.
*   **Etapas Convolucionales**:
    *   **Layer 1**: 2 bloques residuales de 64 canales, stride 1. Salida: 64 x 24 x 24.
    *   **Layer 2**: 2 bloques residuales de 128 canales, stride de reducción 2. Salida: 128 x 12 x 12.
    *   **Layer 3**: 2 bloques residuales de 256 canales, stride de reducción 2. Salida: 256 x 6 x 6.
    *   **Layer 4**: 2 bloques residuales de 512 canales, stride de reducción 2. Salida: 512 x 3 x 3.
*   **Capa de Regresión**:
    *   Adaptive Average Pooling a tamaño 1 x 1 -> Salida: 512 x 1 x 1.
    *   Aplanado a vector de 512 elementos.
    *   Capa lineal final: 512 entradas -> 11 salidas.

---

## 5. Acoplamiento Predictivo Temporal (Kalman / LQG)
El controlador predictivo ZernikeKalmanVectorial procesa los Zernikes de la red de forma dinámica y permite mitigar la latencia (delay) real del ciclo sensor-controlador:

1.  **Sincronización Dinámica de Ruido (R)**: El filtro se ajusta en tiempo real según el modelo seleccionado para modular la confianza en la predicción física vs la CNN.
2.  **Transición basada en Viento (A y Q)**: La matriz de transición $A$ modela la evolución temporal como $\alpha = 1 - 0.3 \times v$. La covarianza de ruido del proceso $Q$ se calcula de manera exacta como $Q = \text{var\_Noll} \times (D/r_0)^{5/3} \times (1 - \alpha^2) \times q_{scale}$.
3.  **Proyección Predictiva Futura**: El lazo realiza la proyección predictiva $x(t+k) = A^k @ x(t)$ para compensar la latencia acumulada.
