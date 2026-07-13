# Sistema AO - Control Unit (UBB Chile)

Este proyecto es una unidad de control de **Óptica Adaptativa (AO)** diseñada para orquestar la simulación y corrección de frentes de onda en tiempo real. El sistema está optimizado para trabajar con un modulador espacial de luz (SLM) **Holoeye Pluto (1550nm)**.

## Arquitectura del Sistema

El proyecto utiliza una arquitectura de microservicios orquestada con **Docker Compose**:

### Frontend
*   **Interfaz (`ao_interfaz`):** Dashboard premium estilo "Mission Control" construido con React + Vite. Permite la visualización de datos y el control del ciclo de calibración.

### Backend & Core
*   **Controlador (`ao_controlador`):** Orquestador central que coordina la comunicación entre la IA, el simulador y la base de datos.
*   **Simulador (`ao_simulador`):** Motor de física óptica optimizado para el SLM Holoeye Pluto
*   **Inferencia (`ao_inferencia`):** Servicio de IA que procesa imágenes mediante Redes Neuronales Convolucionales (CNN) para predecir coeficientes de Zernike.

### Infraestructura
*   **Base de Datos (`ao_database`):** InfluxDB 2.7 para la persistencia de telemetría y diagnóstico del sistema.

---

## Requisitos Previos

*   **Docker** (Versión 20.10+)
*   **Docker Compose** (Versión 2.0+)

---

## Instrucciones de Despliegue

Siga estos pasos para levantar el sistema completo:

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/RJLeyvag549/Sistema-AO.git
    cd Sistema-AO
    ```

2.  **Configurar variables de entorno:**
    Cree un archivo `.env` basado en el ejemplo (necesario para las credenciales de la DB):
    ```bash
    cp .env.example .env
    ```

3.  **Construir y levantar los contenedores:**
    ```bash
    docker-compose up -d --build
    ```

4.  **Acceder a las plataformas:**
    *   **Interfaz de Control:** [http://localhost:3000](http://localhost:3000)
    *   **Panel de InfluxDB (UI):** [http://localhost:8086](http://localhost:8086)

5.  **Verificar que todo esté funcionando (Logs):**
    Para ver la actividad de cada servicio en tiempo real, ejecute:
    ```bash
    docker-compose logs -f
    ```
    Esto es lo que verá por cada servicio:

    | Servicio | Cuándo aparece |
    |---|---|
    | `ao_simulador` | Cuando la interfaz carga o actualiza las imágenes de simulación. |
    | `ao_inferencia` | Cuando se inicia una calibración y se necesita una predicción de la IA. |
    | `ao_controlador` | Al arrancar y cada vez que se presiona "Calibrar Sistema" en la interfaz. |
    | `ao_database` | Al iniciar por primera vez y al apagar el sistema. En operación normal puede estar silencioso, pero los datos sí se están guardando. |

---

## Especificaciones de Hardware (Simulado)

El sistema está configurado con los parámetros reales del **Holoeye Pluto**:
*   **Longitud de onda:** 1550 nm (Rango de operación: 1400 nm a 1700 nm)
*   **Resolución:** 1920 x 1080 píxeles
*   **Pixel Pitch:** 8.0 µm
*   **Modulación:** Fase pura (0 - 2π)

---

## Requisitos de Hardware (Modo Cámara Real)

El sistema está diseñado para integrarse con una cámara física **Point Grey Chameleon CMLN-13S2M** vía USB. Debido a la necesidad de acceso directo a nivel de kernel al dispositivo USB, el demonio de la cámara no se ejecuta dentro de Docker, sino directamente en el entorno de Windows anfitrión.

Para ejecutar el sistema en modo cámara real, deberá instalar una sola vez:

1. **[FlyCapture2 SDK](https://www.flir.com/products/flycapture-sdk/)**: Descargue e instale el SDK de FLIR/Point Grey en su sistema Windows. Esto instalará los controladores USB necesarios y colocará la DLL base en `C:\Program Files\Point Grey Research\FlyCapture2\bin64\FlyCapture2_C_v100.dll`.
2. **Dependencias de Python en Windows**:
   Instale los paquetes necesarios en su entorno Python local de Windows:
   ```bash
   pip install opencv-python numpy requests
   ```

**Para Iniciar en Modo Cámara Real:**
No use `docker-compose up`. En su lugar, ejecute el script principal que levanta los contenedores y el demonio de la cámara en secuencia:
```cmd
iniciar_sistema_ao.bat
```

---

## Modelos CNN de Inferencia

El servicio `ao_inferencia` incluye tres modelos pre-entrenados listos para usar sin necesidad de configuración adicional.

### Modelos disponibles

| Modelo | Arquitectura | Parámetros | Uso |
|---|---|---|---|
| `phase_diversity` | CNN base (2 canales) | 4.8 M | Estimación de Zernike por diversidad de fase |
| `resnet10` | ResNet-10 (2 canales) | 1.2 M | Predicción rápida, bajo consumo de VRAM |
| `resnet18` | ResNet-18 (2 canales) | 11.2 M | Mayor precisión, requiere GPU |

### Carga de modelos — Prioridad automática

Al iniciar, el servicio busca los pesos en este orden:

```
1. /app/shared/   ← volumen Docker (salida de train.py, tiene precedencia)
2. /app/models/   ← modelos versionados en el repositorio (por defecto)
3. /app/          ← compatibilidad con instalaciones antiguas
```

**En un `git clone` + `docker-compose up` sin entrenamiento previo**, el sistema arranca directamente con los modelos pre-entrenados incluidos en `inferencia/models/`. No es necesario entrenar para que el sistema funcione.

Cuando se ejecuta `train.py`, los nuevos pesos se guardan en el volumen `shared_data` y toman precedencia automáticamente. Si el volumen se elimina (`docker-compose down -v`), el sistema vuelve a los modelos del repositorio sin intervención manual.

---

## Entrenamiento de Modelos (Opcional)

Para re-entrenar cualquier modelo con nuevas muestras simuladas al vuelo:

```bash
# Entrenar Phase Diversity (modelo base)
docker exec -it ao_inferencia python train.py --model phase_diversity --samples 100000 --epochs 10

# Entrenar ResNet-10 (más rápido)
docker exec -it ao_inferencia python train.py --model resnet10 --samples 100000 --epochs 10

# Entrenar ResNet-18 (mayor precisión, requiere GPU)
docker exec -it ao_inferencia python train.py --model resnet18 --samples 100000 --epochs 10
```

**Parámetros disponibles:**

| Parámetro | Descripción | Por defecto |
|---|---|---|
| `--model` | Modelo a entrenar: `phase_diversity`, `resnet10`, `resnet18` | `phase_diversity` |
| `--samples` | Número de muestras simuladas al vuelo | `100000` |
| `--val-samples` | Muestras de validación | `10000` |
| `--epochs` | Ciclos de entrenamiento | `10` |

Los pesos entrenados se guardan en el volumen compartido (`shared_data`) y se activan automáticamente en la siguiente inferencia sin necesidad de reiniciar los contenedores.

---

## Captura de Datos y Análisis (Desarrollo)

Para capturar datos de telemetría del sistema en operación y analizar el rendimiento del filtro Kalman+LQG:

```bash
# Captura automática (~13 min, cubre 40 combinaciones de viento y turbulencia)
python scripts/captura_datos_viento.py
```

El script genera un CSV en `scripts/resultados/` con los coeficientes Zernike reales, predicciones CNN y salida del controlador Kalman para cada frame. Los CSVs están excluidos del repositorio (`.gitignore`) por su tamaño.

---


