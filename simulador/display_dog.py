import numpy as np
from PIL import Image, ImageTk, ImageOps
import tkinter as tk
from screeninfo import get_monitors
import sys

# Ruta de la imagen del perro generada
DOG_PATH = r"C:\Users\rjley\.gemini\antigravity-ide\brain\59d1bba7-6aca-4ac9-9461-09f91423827a\cute_dog_1781038631732.png"

# Resolución del Holoeye Pluto 2.1
WIDTH = 1920
HEIGHT = 1080
RADIO = HEIGHT // 4  # 270 píxeles de radio (diámetro 540)
CENTRO_X = WIDTH // 2
CENTRO_Y = HEIGHT // 2

# ==========================================
# 1. PROCESAR LA IMAGEN DEL PERRO
# ==========================================
try:
    # Cargar y convertir a escala de grises (adecuado para el modulador de fase)
    img_perro = Image.open(DOG_PATH).convert("L")
    
    # Redimensionar y recortar para que encaje perfectamente en el círculo (540x540)
    diametro = RADIO * 2
    img_perro_crop = ImageOps.fit(img_perro, (diametro, diametro), Image.Resampling.LANCZOS)
    
    # Crear un lienzo negro de 1920x1080
    lienzo = Image.new("L", (WIDTH, HEIGHT), 0)
    
    # Pegar la imagen del perro en el centro geométrico
    pos_x = CENTRO_X - RADIO
    pos_y = CENTRO_Y - RADIO
    lienzo.paste(img_perro_crop, (pos_x, pos_y))
    
    # Aplicar la máscara circular
    x = np.arange(WIDTH)
    y = np.arange(HEIGHT)
    X, Y = np.meshgrid(x, y)
    distancia = np.sqrt((X - CENTRO_X)**2 + (Y - CENTRO_Y)**2)
    mascara_circular = distancia <= RADIO
    
    datos_gris = np.array(lienzo)
    datos_gris[~mascara_circular] = 0  # Todo lo de afuera queda en negro
    
    # Guardar resultado
    ruta_salida = "perro_circular.png"
    imagen_final = Image.fromarray(datos_gris)
    imagen_final.save(ruta_salida)
    print(f"[OK] Imagen circular del perro guardada como '{ruta_salida}'")
    
except Exception as e:
    print(f"[ERROR] No se pudo procesar la imagen del perro: {e}")
    sys.exit(1)

# ==========================================
# 2. PROYECCIÓN EN EL SLM (PANTALLA 2)
# ==========================================
def mostrar_en_slm():
    try:
        monitors = get_monitors()
    except Exception as e:
        print(f"[WARN] No se pudo leer la lista de monitores de forma automática: {e}")
        monitors = []

    if len(monitors) > 1:
        slm = monitors[1]
        print(f"[INFO] SLM detectado automáticamente en X={slm.x}, Y={slm.y}")
    else:
        print("[WARN] No se detectó el SLM por HDMI. Proyectando en la pantalla principal para pruebas.")
        if len(monitors) == 1:
            slm = monitors[0]
        else:
            class FallbackMonitor:
                x = 1920
                y = 0
            slm = FallbackMonitor()

    root = tk.Tk()
    root.geometry(f"{WIDTH}x{HEIGHT}+{slm.x}+{slm.y}")
    root.overrideredirect(True)
    root.configure(bg='black')

    photo = ImageTk.PhotoImage(imagen_final)
    label = tk.Label(root, image=photo, bg='black', borderwidth=0, highlightthickness=0)
    label.pack(fill="both", expand=True)

    root.bind("<Escape>", lambda e: root.destroy())

    print("\n=========================================================")
    print("PROYECTANDO PERRO EN EL SLM...")
    print("-> Haz clic en la ventana del SLM y presiona 'ESC' para salir.")
    print("=========================================================")
    
    root.mainloop()

if __name__ == "__main__":
    mostrar_en_slm()
