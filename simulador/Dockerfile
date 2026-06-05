FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exponemos solo el puerto 5000 para la API de React
EXPOSE 5000

CMD ["python", "main.py"]