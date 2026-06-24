#!/bin/bash
# =============================================================================
# Fix WSL2 + Docker + torch.compile/inductor:
# inductor busca 'libcuda.so' (sin version), pero WSL2 solo provee 'libcuda.so.1'
# en el directorio de drivers. Hay que crear el symlink ANTES de que Python
# arranque, para que el dynamic linker lo vea cuando inductor lo necesite.
# =============================================================================

LIBCUDA_SO1=$(find /usr/lib/wsl -name "libcuda.so.1" 2>/dev/null | head -1)

if [ -n "$LIBCUDA_SO1" ]; then
    LIBCUDA_DIR=$(dirname "$LIBCUDA_SO1")

    # Symlink en /usr/local/lib (directorio estandar del linker, escribible)
    if [ ! -e "/usr/local/lib/libcuda.so" ]; then
        ln -sf "$LIBCUDA_SO1" "/usr/local/lib/libcuda.so"
        echo "[ENTRYPOINT] Symlink creado: /usr/local/lib/libcuda.so -> $LIBCUDA_SO1"
    else
        echo "[ENTRYPOINT] Symlink ya existe: /usr/local/lib/libcuda.so"
    fi

    # Tambien crear symlink en el directorio original del driver (por si inductor usa ruta absoluta)
    if [ ! -e "$LIBCUDA_DIR/libcuda.so" ]; then
        ln -sf "$LIBCUDA_SO1" "$LIBCUDA_DIR/libcuda.so" 2>/dev/null && \
            echo "[ENTRYPOINT] Symlink adicional: $LIBCUDA_DIR/libcuda.so" || \
            echo "[ENTRYPOINT] Sin permisos para $LIBCUDA_DIR (solo /usr/local/lib)"
    fi

    # Refrescar cache del dynamic linker
    ldconfig 2>/dev/null && echo "[ENTRYPOINT] ldconfig completado." || true

    export LD_LIBRARY_PATH="/usr/local/lib:$LIBCUDA_DIR:${LD_LIBRARY_PATH:-}"
    echo "[ENTRYPOINT] LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
else
    echo "[ENTRYPOINT] libcuda.so.1 no encontrado (entorno no-WSL). Continuando sin fix."
fi

exec python main.py
