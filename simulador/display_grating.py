import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from screeninfo import get_monitors

# ==========================================
# 1. GENERACIÓN DE LA IMAGEN DE REJILLA (GRATING)
# ==========================================
# Resolución del Holoeye Pluto 2.1
WIDTH = 1920
HEIGHT = 1080
PERIOD = 9.0  # Período de 9 píxeles

# Crear malla de coordenadas (X, Y)
x = np.arange(WIDTH)
y = np.arange(HEIGHT)
X, Y = np.meshgrid(x, y)

# Aplicar la fórmula: fase = X * 2 * pi / Periodo
# 'plain = ones(size(y))' implica que el perfil es constante a lo largo de Y
fase = (X * 2 * np.pi / PERIOD)

# Envolver la fase en el rango de 0 a 2*pi
fase_envuelta = np.mod(fase, 2 * np.pi)

# Mapear linealmente de [0, 2*pi) a [0, 255] para escala de grises de 8 bits
# Nota: El SLM mapea los niveles de gris (0-255) a desfases (0-2pi) con su tabla de calibración (LUT)
datos_gris = (fase_envuelta / (2 * np.pi) * 255).astype(np.uint8)

# --- APLICACIÓN DE MÁSCARA CIRCULAR (PUPILA) ---
centro_x = WIDTH // 2
centro_y = HEIGHT // 2
radio = HEIGHT // 4  # 270 píxeles (la mitad del tamaño anterior)

distancia = np.sqrt((X - centro_x)**2 + (Y - centro_y)**2)
mascara_circular = distancia <= radio

# Fuera del círculo el desfase es cero (color negro)
datos_gris[~mascara_circular] = 0

# Crear y guardar la imagen
ruta_imagen = "rejilla_fase.png"
imagen_grating = Image.fromarray(datos_gris)
imagen_grating.save(ruta_imagen)
print(f"[OK] Imagen de rejilla circular generada y guardada como '{ruta_imagen}'")

# ==========================================
# 2. PROYECCIÓN AUTOMÁTICA EN EL SLM (PANTALLA 2)
# ==========================================
def mostrar_en_slm():
    # Detectar pantallas
    try:
        monitors = get_monitors()
    except Exception as e:
        print(f"[WARN] No se pudo leer la lista de monitores de forma automática: {e}")
        monitors = []

    # Seleccionar la segunda pantalla (SLM) si está disponible, sino la primera (para pruebas)
    if len(monitors) > 1:
        slm = monitors[1]
        print(f"[INFO] SLM detectado automáticamente: {slm.name} ({slm.width}x{slm.height}) en posición X={slm.x}, Y={slm.y}")
    else:
        print("[WARN] No se detectó un segundo monitor por HDMI. Proyectando en la pantalla principal para pruebas.")
        if len(monitors) == 1:
            slm = monitors[0]
        else:
            # Fallback genérico si screeninfo no devuelve nada
            class FallbackMonitor:
                x = 1920  # Suponiendo que la pantalla principal mide 1920 de ancho
                y = 0
                width = 1920
                height = 1080
            slm = FallbackMonitor()

    # Configurar ventana de Tkinter
    root = tk.Tk()
    
    # Establecer geometría para que cubra exactamente el SLM
    root.geometry(f"{WIDTH}x{HEIGHT}+{slm.x}+{slm.y}")
    
    # Remover bordes, barra de título y decoraciones
    root.overrideredirect(True)
    root.configure(bg='black')

    # Cargar la imagen generada
    img = Image.open(ruta_imagen)
    photo = ImageTk.PhotoImage(img)

    # Label para contener la imagen sin bordes adicionales
    label = tk.Label(root, image=photo, bg='black', borderwidth=0, highlightthickness=0)
    label.pack(fill="both", expand=True)

    # Salida segura con la tecla ESC
    root.bind("<Escape>", lambda e: root.destroy())

    print("\n=========================================================")
    print("PROYECTANDO EN EL SLM...")
    print("-> Haz clic en la ventana del SLM y presiona la tecla 'ESC' para salir.")
    print("=========================================================")
    
    root.mainloop()

if __name__ == "__main__":
    mostrar_en_slm()
