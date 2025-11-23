PY=python
export PYTHONPATH := $(PWD)

.PHONY: preprocess train evaluate api dash all docker-build docker-train docker-up docker-logs

# Valor por defecto; puedes sobreescribir: make preprocess DATASET_PATH=...
DATASET_PATH ?= data/raw/Airbnb_Open_Data.csv

preprocess:
	$(PY) -m src.eda --input "$(DATASET_PATH)" --quick-clean
	$(PY) -m src.features

train:
	$(PY) -m src.train

evaluate:
	$(PY) -m src.evaluate

api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

dash:
	streamlit run dashboard/streamlit_app.py --server.port 8501 --server.address localhost

all: preprocess train evaluate

# Docker helpers
docker-build:
	docker compose build
docker-train:
	docker compose run --rm train
docker-up:
	docker compose up -d api
docker-logs:
	docker compose logs -f api
