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

**Desarrollado para el Laboratorio de Óptica Adaptativa - Universidad del Bío-Bío.**
