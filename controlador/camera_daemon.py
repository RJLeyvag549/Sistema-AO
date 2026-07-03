"""
camera_daemon.py - Daemon de captura para Point Grey Chameleon CMLN-13S2M
Usa la C-API de FlyCapture2 (FlyCapture2_C_v100.dll) via ctypes.

Resolucion: 1280x960 Mono8 a 15 FPS (nativos del hardware).
Permite ajustar Shutter, Gain y Zoom (ROI size) en tiempo real
recibiendo comandos desde la respuesta del POST del backend.
"""

import ctypes
import ctypes.util
import os
import sys
import time
import threading
import requests
import numpy as np
import cv2

# ── Ruta a las DLLs de FlyCapture2 ──────────────────────────────────────────
FC2_BIN = r"C:\Program Files\Point Grey Research\FlyCapture2\bin64"
FC2_C_DLL = os.path.join(FC2_BIN, "FlyCapture2_C_v100.dll")

# ── Configuracion ─────────────────────────────────────────────────────────────
API_URL     = "http://localhost:5000/camera/frame"
WIDTH       = 1280
HEIGHT      = 960
FPS_TARGET  = 15
FRAME_TIME  = 1.0 / FPS_TARGET


# ── Constantes y Estructuras de la C-API de FlyCapture2 ──────────────────────
class fc2PropertyType(ctypes.c_uint):
    FC2_SHUTTER = 11
    FC2_GAIN    = 12

class fc2Property(ctypes.Structure):
    _fields_ = [
        ("type",           ctypes.c_uint),
        ("reserved",       ctypes.c_uint),
        ("present",        ctypes.c_bool),
        ("autoManualMode", ctypes.c_bool),
        ("onOff",          ctypes.c_bool),
        ("absControl",     ctypes.c_bool),
        ("onePush",        ctypes.c_bool),
        ("valueA",         ctypes.c_uint),
        ("valueB",         ctypes.c_uint),
        ("absValue",       ctypes.c_float),
        ("reserved1",      ctypes.c_uint * 8),
    ]

# Handles opacos de la API
fc2Context    = ctypes.c_void_p
fc2PGRGuid    = (ctypes.c_uint * 4)

class fc2Image(ctypes.Structure):
    _fields_ = [
        ("rows",           ctypes.c_uint),
        ("cols",           ctypes.c_uint),
        ("stride",         ctypes.c_uint),
        ("pData",          ctypes.c_void_p),
        ("dataSize",       ctypes.c_uint),
        ("receivedDataSize", ctypes.c_uint),
        ("format",         ctypes.c_uint),
        ("bayerFormat",    ctypes.c_uint),
        ("imageImpl",      ctypes.c_void_p),
    ]


def _load_fc2() -> ctypes.CDLL | None:
    if not os.path.exists(FC2_C_DLL):
        print(f"[CAMARA] DLL no encontrada: {FC2_C_DLL}")
        return None
    try:
        os.add_dll_directory(FC2_BIN)
        lib = ctypes.CDLL(FC2_C_DLL)
        print(f"[CAMARA] FlyCapture2_C_v100.dll cargada correctamente.")
        return lib
    except OSError as e:
        print(f"[CAMARA] Error cargando DLL: {e}")
        return None


# ── Clase principal del daemon ────────────────────────────────────────────────
class ChameleonCamera:
    def __init__(self):
        self.lib         = None
        self.context     = fc2Context(None)
        self.image       = fc2Image()
        self.is_running  = False
        self.is_connected = False
        self.thread      = None
        self.lock        = threading.Lock()
        self.resolution  = (WIDTH, HEIGHT)

        # Control de parpadeos y fallos consecutivos
        self.consecutive_failures = 0

        # Propiedades activas de la camara
        self.roi_size = 96
        self.current_shutter = -1.0
        self.current_gain = -1.0

        # Estado para simulacion de fallback
        self._mock_x  = WIDTH  / 2.0
        self._mock_y  = HEIGHT / 2.0
        self._mock_vx = 0.4
        self._mock_vy = 0.25
        self._frame_idx = 0

    def initialize(self) -> bool:
        self.lib = _load_fc2()
        if self.lib is None:
            print("[CAMARA] FlyCapture2 no disponible. Modo simulacion.")
            return True

        # 1. Crear contexto
        err = self.lib.fc2CreateContext(ctypes.byref(self.context))
        if err != 0:
            print(f"[CAMARA] fc2CreateContext fallo (err={err}). Modo simulacion.")
            return True

        # 2. Numero de camaras
        num_cams = ctypes.c_uint(0)
        self.lib.fc2GetNumOfCameras(self.context, ctypes.byref(num_cams))
        if num_cams.value == 0:
            print("[CAMARA] No se encontraron camaras en el bus USB. Modo simulacion.")
            return True

        # 3. Obtener GUID
        guid = fc2PGRGuid()
        err = self.lib.fc2GetCameraFromIndex(self.context, 0, ctypes.byref(guid))
        if err != 0:
            return True

        # 4. Conectar
        err = self.lib.fc2Connect(self.context, ctypes.byref(guid))
        if err != 0:
            return True

        # 5. Configurar modo de video nativo 1280x960 Mono8
        self.lib.fc2SetVideoModeAndFrameRate(
            self.context,
            ctypes.c_uint(17),  # FC2_VIDEOMODE_1280x960Y8
            ctypes.c_uint(3)    # FC2_FRAMERATE_15
        )

        # 6. Crear imagen
        self.lib.fc2CreateImage(ctypes.byref(self.image))

        # 7. Iniciar captura
        err = self.lib.fc2StartCapture(self.context)
        if err != 0:
            return True

        self.is_connected = True
        self.consecutive_failures = 0
        print(f"[CAMARA] Camara Chameleon fisica conectada y capturando a 1280x960.")
        return True

    def start(self) -> bool:
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.lib and self.is_connected:
            try:
                self.lib.fc2StopCapture(self.context)
                self.lib.fc2DestroyImage(ctypes.byref(self.image))
                self.lib.fc2Disconnect(self.context)
                self.lib.fc2DestroyContext(self.context)
            except Exception:
                pass
        print("[CAMARA] Daemon detenido.")

    def _capture_loop(self):
        while self.is_running:
            t_start = time.perf_counter()
            frame = None

            if self.is_connected and self.lib:
                try:
                    with self.lock:
                        err = self.lib.fc2RetrieveBuffer(self.context, ctypes.byref(self.image))
                    if err == 0 and self.image.pData:
                        rows = self.image.rows
                        cols = self.image.cols
                        buf = (ctypes.c_uint8 * (rows * cols)).from_address(self.image.pData)
                        frame = np.frombuffer(buf, dtype=np.uint8).reshape(rows, cols).copy()
                        self.resolution = (cols, rows)
                        self.consecutive_failures = 0
                    else:
                        # Error transitorio (ej: timeout err=20). No desconectar de inmediato
                        self.consecutive_failures += 1
                        if self.consecutive_failures > 15:
                            print(f"[CAMARA] Perdida persistente de conexion (err={err}). Desconectando...")
                            self.is_connected = False
                except Exception as e:
                    self.consecutive_failures += 1
                    if self.consecutive_failures > 15:
                        print(f"[CAMARA] Excepcion en captura: {e}. Desconectando...")
                        self.is_connected = False

            if frame is None:
                # Retorna frame simulado si no hay conexion fisica
                frame = self._generate_mock_frame()

            if frame is not None:
                self._process_and_send(frame)

            self._frame_idx += 1
            elapsed = time.perf_counter() - t_start
            remaining = FRAME_TIME - elapsed
            if remaining > 0.001:
                time.sleep(remaining)

    def _update_camera_properties(self, shutter, gain):
        if not self.lib or not self.is_connected or not self.context:
            return

        # Solo enviar comandos al hardware si el valor cambio en la UI
        if abs(self.current_shutter - shutter) > 0.05:
            self._set_property(fc2PropertyType.FC2_SHUTTER, shutter)
            self.current_shutter = shutter
            print(f"[CAMARA] Hardware: Shutter actualizado a {shutter:.2f} ms")

        if abs(self.current_gain - gain) > 0.05:
            self._set_property(fc2PropertyType.FC2_GAIN, gain)
            self.current_gain = gain
            print(f"[CAMARA] Hardware: Gain actualizado a {gain:.2f} dB")

    def _set_property(self, prop_type, val):
        try:
            prop = fc2Property()
            prop.type = prop_type
            # Leer estado actual para no alterar otros campos
            err = self.lib.fc2GetProperty(self.context, ctypes.byref(prop))
            if err == 0:
                prop.autoManualMode = False  # Modo manual obligatorio
                prop.absControl = True       # Usar unidades absolutas (ms / dB)
                prop.absValue = float(val)
                self.lib.fc2SetProperty(self.context, ctypes.byref(prop))
        except Exception as e:
            print(f"[CAMARA] Error configurando propiedad {prop_type}: {e}")

    def _generate_mock_frame(self) -> np.ndarray:
        W, H = self.resolution
        self._mock_x += self._mock_vx + np.random.normal(0, 0.7)
        self._mock_y += self._mock_vy + np.random.normal(0, 0.7)
        if self._mock_x < 150 or self._mock_x > W - 150:
            self._mock_vx *= -1
        if self._mock_y < 100 or self._mock_y > H - 100:
            self._mock_vy *= -1
        cx = int(np.clip(self._mock_x, self.roi_size, W - self.roi_size))
        cy = int(np.clip(self._mock_y, self.roi_size, H - self.roi_size))

        Y, X = np.ogrid[:H, :W]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        spot = 220 * np.exp(-dist**2 / 180.0) + 35 * np.exp(-dist**2 / 1200.0) * np.cos(dist / 2.5)**2
        img = np.clip(spot, 0, 255).astype(np.uint8)
        return cv2.add(img, np.random.poisson(3, img.shape).astype(np.uint8))

    def _process_and_send(self, frame: np.ndarray):
        W, H = self.resolution

        # 1. Centroide
        _, thresh = cv2.threshold(frame, 40, 255, cv2.THRESH_TOZERO)
        M = cv2.moments(thresh)
        cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else W // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else H // 2

        # 2. Recortar ROI dinamica basada en el Zoom (roi_size)
        r_size = self.roi_size
        x0 = int(np.clip(cx - r_size // 2, 0, W - r_size))
        y0 = int(np.clip(cy - r_size // 2, 0, H - r_size))
        roi_focused = frame[y0:y0+r_size, x0:x0+r_size].copy()
        
        # Redimensionar a 96x96 para la red neuronal de forma transparente
        if r_size != 96:
            roi_focused_nn = cv2.resize(roi_focused, (96, 96))
        else:
            roi_focused_nn = roi_focused

        # 3. Canal de Phase Diversity
        roi_defocused_nn = self._apply_numerical_defocus(roi_focused_nn)

        # 4. Preview
        preview = cv2.resize(frame, (320, 240))
        preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
        px = int(cx * 320 / W)
        py = int(cy * 240 / H)
        cv2.drawMarker(preview_bgr, (px, py), (0, 0, 220), cv2.MARKER_CROSS, 20, 2)

        # Dibujar rectangulo de la ROI/Zoom
        rx0 = int((cx - r_size//2) * 320 / W)
        ry0 = int((cy - r_size//2) * 240 / H)
        rx1 = int((cx + r_size//2) * 320 / W)
        ry1 = int((cy + r_size//2) * 240 / H)
        cv2.rectangle(preview_bgr, (rx0, ry0), (rx1, ry1), (0, 180, 0), 1)

        # 5. Enviar e interponer configuracion desde la respuesta
        payload = {
            "centroid":      [cx, cy],
            "roi_focused":   roi_focused_nn.tolist(),
            "roi_defocused": roi_defocused_nn.tolist(),
            "preview_bytes": preview_bgr[:, :, 2].tobytes().hex(),
            "timestamp":     time.time(),
        }
        try:
            resp = requests.post(API_URL, json=payload, timeout=0.25)
            if resp.status_code == 200:
                data = resp.json()
                shutter = data.get("camera_shutter", 33.3)
                gain = data.get("camera_gain", 0.0)
                roi_size_new = data.get("camera_roi_size", 96)
                
                # Sincronizar zoom
                if roi_size_new != self.roi_size:
                    self.roi_size = roi_size_new
                    print(f"[CAMARA] Zoom (ROI) actualizado a {roi_size_new}x{roi_size_new} px")
                
                # Sincronizar Shutter y Gain en la camara fisica
                self._update_camera_properties(shutter, gain)
        except Exception:
            pass

    def _apply_numerical_defocus(self, psf: np.ndarray, defocus_rad: float = 1.5) -> np.ndarray:
        img = psf.astype(np.float32) / 255.0
        amplitude = np.sqrt(np.clip(img, 0, None))
        pupil = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(amplitude)))
        Hp, Wp = pupil.shape
        y, x = np.mgrid[-1:1:Hp*1j, -1:1:Wp*1j]
        z4 = np.sqrt(3.0) * (2.0 * (x**2 + y**2) - 1.0)
        pupil_ab = pupil * np.exp(1j * defocus_rad * z4)
        psf_def = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(pupil_ab))))**2
        mx = np.max(psf_def)
        return (psf_def / mx * 255).astype(np.uint8) if mx > 0 else np.zeros_like(psf)


if __name__ == "__main__":
    daemon = ChameleonCamera()
    if not daemon.initialize():
        sys.exit(1)

    daemon.start()
    print(f"\n[CAMARA] Daemon corriendo a {FPS_TARGET} FPS. Ctrl+C para detener.")
    print(f"[CAMARA] Interfaz en: http://localhost:3000  (pestana 'Camara Real')\n")

    prev_frames = 0
    try:
        while True:
            time.sleep(2.0)
            frames = daemon._frame_idx
            real_fps = (frames - prev_frames) / 2.0
            prev_frames = frames
            mode = "FC2-Chameleon" if daemon.is_connected else "Simulacion"
            res = f"{daemon.resolution[0]}x{daemon.resolution[1]}"
            print(f"[CAMARA] {mode} | {res} | FPS: {real_fps:.1f} | Shutter: {daemon.current_shutter:.1f}ms | Gain: {daemon.current_gain:.1f}dB | Zoom: {daemon.roi_size}   ", end="\r")
    except KeyboardInterrupt:
        daemon.stop()
