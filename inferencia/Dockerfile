FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Instalar gcc: requerido por torch.compile/inductor para compilar kernels optimizados
# La imagen 'runtime' no incluye compiladores (a diferencia de la imagen 'devel')
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    # Verificar que pip NO sobreescribio PyTorch CUDA con version CPU-only
    python -c "\
import torch; \
assert torch.version.cuda is not None, \
'[ERROR] pip reinstalo torch sin soporte CUDA. Revisar requirements.txt o dependencias de paquetes.'; \
print(f'[OK] PyTorch {torch.__version__} | CUDA compilado con: {torch.version.cuda}')"

COPY . .

# Convertir CRLF -> LF (necesario porque el archivo se edita en Windows)
# y otorgar permisos de ejecucion al entrypoint
RUN sed -i 's/\r//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

