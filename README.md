# NYC Airbnb Price Prediction – Arquitectura y Despliegue

> **Demo en producción:** https://airbnb-nyc.streamlit.app/  
> **API FastAPI desplegada en Docker sobre AWS EC2:** http://3.135.181.84:8000/docs 


**Demo Analítica con Python** cuyo objetivo es predecir el precio por noche de alojamientos tipo Airbnb en la ciudad de Nueva York, utilizando un pipeline completo de **ML + API + Dashboard**.

---

## 1. Visión General del Proyecto

- **Problema de negocio**: ayudar a anfitriones de Airbnb a fijar un precio competitivo para sus propiedades en NYC, considerando ubicación, tipo de habitación, actividad del anuncio y disponibilidad.
- **Enfoque técnico**:
  - Aprendizaje supervisado (regresión) sobre un dataset de más de 270k listados.
  - API de predicción de precios expuesta con **FastAPI**.
  - Dashboard interactivo tipo **“Calculadora de Precios Airbnb”** desarrollado en **Streamlit** y desplegado en **Streamlit Cloud**.
  - Backend de predicción desplegado como contenedor **Docker** en **AWS EC2**, con imagen almacenada en **AWS ECR**.

---

## 2. Arquitectura de Alto Nivel

La solución se organiza en tres componentes principales:

1. **Pipeline de Datos y Modelo (local / repo)**  
   - Limpieza y normalización del dataset NYC Airbnb.  
   - Ingeniería de características numéricas, categóricas y geoespaciales.  
   - Entrenamiento de un modelo de regresión (XGBoost / Random Forest) para predecir el precio.

2. **API de Predicción (backend)**  
   - Implementada en **FastAPI**, con definición clara de esquemas de entrada/salida.  
   - Empaquetada en un contenedor Docker y desplegada en una instancia **AWS EC2**.  

3. **Dashboard en Streamlit Cloud (frontend)**  
   - Interfaz web donde el usuario:
     - Ingresa las características de su propiedad.
     - Envía una solicitud a la API `/properties/predict_price`.
     - Visualiza el precio estimado, mapas y gráficos de apoyo.  
   - Uso de **Mapbox** para mapas y **pydeck / plotly** para visualizaciones.  

**Esquema simplificado:**

```text
[Dataset Airbnb NYC] 
        |
        v
  (EDA + Features + Modelo)
        |
        v
[models/model.joblib]  --->  [FastAPI @ EC2 (Docker)]  --->  /properties/predict_price
                                         ^
                                         |
                 [Streamlit Cloud Dashboard]  --->  Llama API_URL (Elastic IP)
```

---

## 3. Stack Tecnológico

- **Lenguaje**: Python 3.11 (tanto para API como para dashboard).
- **Librerías clave**:
  - Datos y ML: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `scipy`, `joblib`, `shap`.
  - Backend: `fastapi`, `uvicorn`, `pydantic`.
  - Frontend: `streamlit`, `pydeck`, `plotly`, `requests`.
- **Infraestructura**:
  - **Docker Desktop** en la máquina local para construir imágenes.
  - **AWS**: IAM, ECR, EC2, Elastic IP.
  - **Streamlit Cloud** para el deployment del dashboard.

---

## 4. Detalle del Backend de Predicción

### 4.1. Pipeline de datos y modelo

- Limpieza del CSV original:
  - Normalización de nombres de columnas y tipos de datos (`price`, `lat`, `long`, fechas, etc.).
  - Corrección de typos en `neighbourhood_group` (ej. `brookln` → `Brooklyn`).
  - Filtros de calidad (rangos razonables de precio, noches mínimas, etc.).
- Ingeniería de características:
  - Variables base: grupo de barrio, tipo de habitación, latitud, longitud, reseñas, disponibilidad, etc.
  - Variables geoespaciales: distancia a Times Square y Wall Street (fórmula de Haversine).
  - Variables logarítmicas (`log1p_…`) para controlar distribuciones sesgadas.
- Modelo:
  - Entrenamiento con **XGBRegressor** sobre `log1p(price)`.
  - Alternativa de evaluación con **RandomForestRegressor** más interpretable.
  - Métricas alcanzadas (ejemplo, modelo evaluado):  
    - RMSE ≈ 12.06  
    - MAE ≈ 1.70  
    - R² ≈ 0.9968  
    - MAPE ≈ 0.58 %

### 4.2. API FastAPI

- **Esquema de entrada** (`PropertyInput`):  
  - `neighbourhood_group` (Manhattan, Brooklyn, etc.)  
  - `neighbourhood` (opcional)  
  - `room_type` (tipo de alojamiento)  
  - `latitude`, `longitude`  
  - `minimum_nights`, `number_of_reviews`, `reviews_per_month`  
  - `calculated_host_listings_count`, `availability_365`  

- **Esquema de salida** (`PredictionResponse`):
  - `predicted_price`: precio estimado por noche (float).  
  - `currency`: `"USD"`.

- **Endpoints principales**:  
  - `GET /health` → verificación básica del servicio.  
  - `POST /properties/predict_price` → recibe `PropertyInput`, añade features derivadas y devuelve `PredictionResponse`.  

- El modelo se carga desde `models/model.joblib` con cache en memoria para evitar recargas en cada request.  

---

## 5. Infraestructura en AWS

### 5.1. Usuario IAM y ECR

- Usuario IAM: **`airbnb-api-user`**, con permisos para:
  - **AmazonEC2ContainerRegistryFullAccess** (ECR).
  - **AmazonEC2FullAccess** (para laboratorio).
- Repositorio privado en **Amazon ECR**:
  - Nombre: `airbnb-nyc-api`.  
  - URI: `245639922091.dkr.ecr.us-east-2.amazonaws.com/airbnb-nyc-api`.  

### 5.2. Instancia EC2 (Backend)

- Sistema operativo: **Amazon Linux 2023 (kernel 6.1)**.  
- Tipo de instancia: **t3.micro** (apto para free tier, suficiente para el laboratorio y demo).  
- Características clave:
  - Acceso por SSH con usuario `ec2-user` y key pair tipo `.pem`.  
  - **Docker** instalado con `dnf`, servicio habilitado y corriendo vía `systemctl`.  
  - **AWS CLI** configurado dentro de la EC2 con las credenciales de `airbnb-api-user`.  

- **Security Group**:
  - Regla SSH (TCP 22) restringida a **“My IP”** para administración segura.  
  - Regla Custom TCP (puerto **8000**) abierta a `0.0.0.0/0` para exponer la API.  

- **Elastic IP**:
  - IP elástica asignada: **`3.135.181.84`**, asociada a la instancia backend.  
  - Esto garantiza una dirección estable para `API_URL`, independientemente del ciclo de vida de la instancia.  

---

## 6. Flujo de Despliegue del Backend (Resumen)

1. **Entorno local**  
   - Instalación y verificación de Docker Desktop (`docker --version`, `docker info`).  
   - Instalación de AWS CLI y configuración con `aws configure` (usuario `airbnb-api-user`).  

2. **Construcción de la imagen Docker (local)**  
   - Dockerfile en la raíz del proyecto (`E:\REPOS\airbnb-price-nyc`).  
   - Comando:  
     ```bash
     docker build -t airbnb-nyc-api:latest .
     ```
   - Prueba local:
     ```bash
     docker run --rm -p 8000:8000 airbnb-nyc-api:latest
     # Navegar a http://localhost:8000/docs
     ```

3. **Subir la imagen a ECR**  
   - Login a ECR:
     ```bash
     aws ecr get-login-password --region us-east-2 |        docker login --username AWS --password-stdin 245639922091.dkr.ecr.us-east-2.amazonaws.com
     ```
   - Etiquetar y hacer push:
     ```bash
     docker tag airbnb-nyc-api:latest        245639922091.dkr.ecr.us-east-2.amazonaws.com/airbnb-nyc-api:latest

     docker push 245639922091.dkr.ecr.us-east-2.amazonaws.com/airbnb-nyc-api:latest
     ```

4. **Provisionar y preparar la EC2**  
   - Lanzar instancia **t3.micro / Amazon Linux 2023** con Security Group adecuado.  
   - Ajustar permisos del `.pem` en Windows y conectarse por SSH.  
   - Instalar Docker y AWS CLI en la instancia.  
   - Configurar `aws configure` dentro de la EC2 (mismas credenciales).  

5. **Desplegar el contenedor en la EC2**  
   - Login a ECR desde la EC2:
     ```bash
     aws ecr get-login-password --region us-east-2 |        sudo docker login --username AWS --password-stdin 245639922091.dkr.ecr.us-east-2.amazonaws.com
     ```
   - Descargar imagen:
     ```bash
     sudo docker pull 245639922091.dkr.ecr.us-east-2.amazonaws.com/airbnb-nyc-api:latest
     ```
   - Ejecutar contenedor:
     ```bash
     sudo docker run -d --name airbnb-api -p 8000:8000        245639922091.dkr.ecr.us-east-2.amazonaws.com/airbnb-nyc-api:latest
     ```
   - Verificación:
     - `sudo docker ps`  
     - `curl http://127.0.0.1:8000/docs` desde la EC2.  
     - `http://3.135.181.84:8000/docs` desde el navegador local.  

---

## 7. Despliegue del Dashboard en Streamlit Cloud

### 7.1. Repositorio en GitHub

- Estructura mínima del proyecto:

  ```text
  E:\REPOS\airbnb-price-nyc
  ├── api/           # FastAPI backend
  ├── dashboard/     # streamlit_app.py
  ├── data/raw/      # Airbnb_Open_Data.csv
  ├── models/        # model.joblib, feature_metadata, metrics, etc.
  ├── src/           # eda.py, features.py, train.py, utils_geo.py, ...
  ├── requirements.txt
  └── README.md
  ```

- El repositorio se publica en GitHub y es el origen único para el deploy del dashboard.  

### 7.2. Creación de la app en Streamlit Cloud

1. Ingresar a Streamlit Cloud y crear **New app**.  
2. Seleccionar:
   - Repository: `TU_USUARIO/airbnb-price-nyc`  
   - Branch: `main`  
   - Main file path: `dashboard/streamlit_app.py`  
3. Deploy inicial (Streamlit instala dependencias desde `requirements.txt`).  

### 7.3. Configuración de Python y Secrets

- En **Advanced settings** de la app:
  - Seleccionar **Python 3.11** para garantizar compatibilidad con librerías.  
- En la sección **Secrets** (actúa como `secrets.toml` remoto):  

  ```toml
  API_URL = "http://3.135.181.84:8000"
  MAPBOX_TOKEN = "pk.eyJ1IjoiZXlhZ3VhciIsImEiOiJjbWk5ejZiaGcwNXhlMm5wdjBkNnh3eWhiIn0.17XLpBqQ5V0CBvUiO59kjA"
  ```

- El código del dashboard lee estos valores con `st.secrets["API_URL"]` y `st.secrets["MAPBOX_TOKEN"]`, por lo que deben estar definidos en la raíz del archivo de secretos (sin secciones anidadas).  

### 7.4. Actualización continua

- Cada vez que se hace `git push` a `main`:
  - Streamlit Cloud detecta el nuevo commit y redepliega automáticamente.
  - No es necesario reajustar los Secrets salvo que cambie la IP de la API o el token de Mapbox.  

---

## 8. Cómo ejecutar el proyecto localmente (modo laboratorio)

### 8.1. Backend local (FastAPI)

1. Crear entorno virtual e instalar dependencias:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```
2. Definir variables en `.env` (ejemplo):
   ```env
   API_HOST=0.0.0.0
   API_PORT=8000
   DATASET_PATH=./data/raw/Airbnb_Open_Data.csv
   MAPBOX_TOKEN=pk.eyJ1IjoiZXlhZ3VhciIsImEiOiJjbWk5ejZiaGcwNXhlMm5wdjBkNnh3eWhiIn0.17XLpBqQ5V0CBvUiO59kjA
   ```
3. Ejecutar la API:
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```
4. Probar en: `http://localhost:8000/docs`.

### 8.2. Dashboard local (Streamlit)

1. Con el entorno activado:
   ```bash
   streamlit run dashboard/streamlit_app.py
   ```
2. Abrir: `http://localhost:8501`.

---

## 9. Puntos Clave

- **Integración extremo a extremo**:
  - Desde un dataset real de Airbnb NYC hasta un sistema funcional de predicción con dashboard web.
- **Buenas prácticas aplicadas**:
  - Separación clara en capas: EDA/Features, Modelo, API, Dashboard.
  - Uso de `pydantic` para validación de datos y contratos robustos en la API.
  - Manejo de secretos fuera del código (Streamlit Secrets, `.env`).
  - Despliegue reproducible con Docker y ECR, siguiendo un flujo estándar de CI/CD manual.  

- **Valor académico**:
  - Combina **analítica de datos**, **machine learning**, **servicios web** y **cloud computing** en un caso práctico alineado con la industria.
