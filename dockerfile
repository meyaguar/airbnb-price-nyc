# Imagen base ligera de Python
FROM python:3.11-slim

# Configuración básica de Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# 1) Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copiar código de la API, fuente y modelo
COPY api/ api/
COPY src/ src/
COPY models/ models/

# Puerto donde exponemos la API dentro del contenedor
EXPOSE 8000

# 3) Comando para arrancar la API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
