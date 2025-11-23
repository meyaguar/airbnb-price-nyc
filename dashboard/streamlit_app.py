# dashboard/streamlit_app.py
from pathlib import Path
import os
import json
import requests
import pandas as pd
import streamlit as st
import pydeck as pdk

import plotly.express as px

# Configuración de página
st.set_page_config(page_title="Airbnb Price Calculator (NYC)", layout="wide")
st.title("🗽 Airbnb NYC - Calculadora de Precio Estimado")

# --- Configuración de pestañas ---
tab1, tab2 = st.tabs(["🤖 Machine Learning", "📊 Dashboard"])

with tab1:

    # --- Paths base del proyecto ---
    ROOT = Path(__file__).resolve().parents[1]
    DATA_INTERIM = ROOT / "data" / "interim"
    MODELS_DIR = ROOT / "models"

    # --- Helpers para configuración robusta ---
    def get_api_url() -> str:
        # 1) secretos (si existieran)
        try:
            if "API_URL" in st.secrets:
                return st.secrets["API_URL"]
        except Exception:
            pass
        # 2) variable de entorno
        env_url = os.getenv("API_URL")
        if env_url:
            return env_url
        # 3) default local
        return "http://localhost:8000"

    def get_mapbox_token() -> str | None:
        # 1) secretos (si existieran)
        try:
            if "MAPBOX_TOKEN" in st.secrets:
                return st.secrets["MAPBOX_TOKEN"]
        except Exception:
            pass
        # 2) variable de entorno (p.ej., MAPBOX_TOKEN o MAPBOX_API_KEY)
        return os.getenv("MAPBOX_TOKEN") or os.getenv("MAPBOX_API_KEY")

    API_URL_DEFAULT = get_api_url()
    MAPBOX_TOKEN = get_mapbox_token()



    # ---- Sidebar: config y inputs ----
    st.sidebar.header("Configuración")
    api_url = st.sidebar.text_input("API URL", value=API_URL_DEFAULT, help="URL de tu API FastAPI")
    st.caption(f"API activa: {api_url}")

    st.sidebar.header("Input de propiedad")
    neighbourhood_group = st.sidebar.selectbox(
        "Neighbourhood group", ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
    )
    room_type = st.sidebar.selectbox(
        "Room type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
    )
    latitude = st.sidebar.number_input("Latitude", value=40.7580, format="%.6f")
    longitude = st.sidebar.number_input("Longitude", value=-73.9855, format="%.6f")
    minimum_nights = st.sidebar.number_input("Minimum nights", value=3, min_value=1, max_value=365)
    number_of_reviews = st.sidebar.number_input("Number of reviews", value=25, min_value=0, max_value=5000)
    reviews_per_month = st.sidebar.number_input("Reviews per month", value=1.2, min_value=0.0, max_value=30.0, step=0.1)
    calculated_host_listings_count = st.sidebar.number_input("Host listings count", value=1, min_value=0, max_value=1000)
    availability_365 = st.sidebar.number_input("Availability 365", value=120, min_value=0, max_value=366)

    payload = {
        "neighbourhood_group": neighbourhood_group,
        "neighbourhood": None,  # no se usa en el modelo actual
        "room_type": room_type,
        "latitude": latitude,
        "longitude": longitude,
        "minimum_nights": int(minimum_nights),
        "number_of_reviews": int(number_of_reviews),
        "reviews_per_month": float(reviews_per_month),
        "calculated_host_listings_count": int(calculated_host_listings_count),
        "availability_365": int(availability_365),
    }

    col_left, col_right = st.columns([1, 1])

    # ---- Utilidad para llamar a la API con manejo de errores ----
    def call_predict(api_base: str, data: dict) -> dict:
        url = f"{api_base.rstrip('/')}/properties/predict_price"
        r = requests.post(url, json=data, timeout=15)
        r.raise_for_status()
        return r.json()

    # ---- Predicción ----
    with col_left:
        st.subheader("🔮 Precio estimado por noche (USD)")
        if st.button("Calcular precio"):
            try:
                pred = call_predict(api_url, payload)
                st.success(f"USD ${pred['predicted_price']:.2f}")
            except requests.exceptions.ConnectionError:
                st.error("No se pudo conectar con la API. ¿Está corriendo en esa URL?")
            except requests.exceptions.HTTPError as e:
                st.error(f"Error HTTP desde la API: {e} — {e.response.text}")
            except Exception as e:
                st.error(f"Error llamando a la API: {e}")

        # Métricas del modelo
        metrics_path = MODELS_DIR / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            st.markdown("**Métricas (test set)**")
            st.json(metrics)
        else:
            st.info("Entrena y evalúa el modelo para ver métricas (models/metrics.json).")

    # ---- Mapa / EDA rápido ----
    with col_right:
        st.subheader("🗺️ Mapa de densidad de precios (Airbnb NYC)")
        parquet_path = DATA_INTERIM / "listings_clean.parquet"
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path)
                # recorta para performance
                if len(df) > 20000:
                    df = df.sample(20000, random_state=42)

                needed = {"latitude", "longitude", "price"}
                if needed.issubset(df.columns):
                    st.caption(f"{len(df):,} puntos renderizados.")

                    # Si no hay token, usamos mapa sin estilo base (sigue mostrando el heatmap sobre fondo simple)
                    map_style = "mapbox://styles/mapbox/light-v9" if MAPBOX_TOKEN else None
                    if MAPBOX_TOKEN:
                        # pydeck usará este token para mapbox
                        pdk.settings.mapbox_api_key = MAPBOX_TOKEN


                    df = df.rename(columns={
                        "neighbourhood": "barrio",
                        "neighbourhood_group": "distrito",
                        "price": "precio"
                    })

                    fig = px.density_mapbox(
                        df,
                        lat="latitude",
                        lon="longitude",
                        z="precio",          # Peso que antes era get_weight
                        radius=10,          # Equivalente a radiusPixels
                        center=dict(lat=40.75, lon=-73.98),
                        zoom=9,
                        mapbox_style="carto-positron",  # Puedes cambiar por map_style si lo tienes
                        color_continuous_scale="inferno",
                        hover_data={          
                            "distrito": True,             # TOOLTIP PERSONALIZADO
                            "barrio": True,
                            "precio": ":.0f",               # formateo opcional
                            "latitude": False,             # ocultar si quieres
                            "longitude": False
                        },
                    )

                    st.plotly_chart(fig,use_container_width=True)
                    if not MAPBOX_TOKEN:
                        st.info("No se detectó MAPBOX_TOKEN. El mapa se muestra sin estilo base. "
                                "Puedes definir MAPBOX_TOKEN en secrets o variables de entorno para un mapa más bonito.")
                else:
                    st.warning(f"El parquet no tiene columnas requeridas {needed}.")
            except Exception as e:
                st.error(f"Error cargando parquet: {e}")
        else:
            st.info("Ejecuta el preprocess para generar data/interim/listings_clean.parquet")

    # ---- Resumen por tipo ----
    st.subheader("📊 Resumen por Room Type / Neighbourhood Group")
    try:
        parquet_path = DATA_INTERIM / "listings_clean.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            cols = [c for c in ["neighbourhood_group", "room_type", "price"] if c in df.columns]
            if set(["room_type", "price"]).issubset(cols):
                st.dataframe(
                    df[cols].groupby(["neighbourhood_group", "room_type"], dropna=True)["price"]
                    .median()
                    .sort_values(ascending=False)
                    .reset_index(name="median_price")
                )
    except Exception as e:
        st.error(f"Error en resumen: {e}")

    st.markdown("---")
    st.caption("API: " + api_url)

with tab2:
    st.title("📊 Dashboard Analítico - Airbnb NYC")
    
    # Cargar datos limpios
    parquet_path = DATA_INTERIM / "listings_clean.parquet"
    
    if not parquet_path.exists():
        st.warning("Primero ejecuta el preprocesamiento para generar los datos limpios.")
        st.stop()
    
    try:
        df = pd.read_parquet(parquet_path)
        # 🔧 Limpieza global inicial
        valid_groups = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
        df = df[df['neighbourhood_group'].isin(valid_groups)].copy()
        df = df.dropna(subset=['neighbourhood_group', 'room_type'])

        st.success(f"✅ Datos cargados: {len(df):,} registros")
        
        # --- KPIs Principales ---
        st.subheader("📈 KPIs Principales")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_price = df['price'].mean()
            st.metric("Precio Promedio", f"${avg_price:.2f}")
        
        with col2:
            total_listings = len(df)
            st.metric("Total de Listados", f"{total_listings:,}")
        
        with col3:
            avg_reviews = df['number_of_reviews'].mean()
            st.metric("Reviews Promedio", f"{avg_reviews:.1f}")
        
        with col4:
            occupancy_rate = (df['availability_365'].mean() / 365) * 100
            st.metric("Disponibilidad Promedio", f"{100 - occupancy_rate:.1f}%")
        
        # --- Filtros ---
        st.subheader("🔍 Filtros")
        
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            neighbourhood_groups = sorted(df['neighbourhood_group'].unique())
            selected_neighbourhoods = st.multiselect(
                "Distritos",
                neighbourhood_groups,
                default=neighbourhood_groups
            )
        
        with col_filter2:
            # 🔤 Traducción de tipos de habitación
            room_type_map = {
                "Entire home/apt": "Casa/Apartamento",
                "Private room": "Privada",
                "Shared room": "Compartida",
                "Hotel room": "Hotel"
            }

            # Lista original de tipos de habitación
            room_types_en = sorted(df['room_type'].unique())

            # Traducción visible al usuario
            room_types_es = [room_type_map.get(rt, rt) for rt in room_types_en]
            selected_room_types_es = st.multiselect(
                "Tipos de Habitación",
                room_types_es,
                default=room_types_es
            )

            # 🔁 Convertir la selección del usuario de español → inglés
            inv_room_type_map = {v: k for k, v in room_type_map.items()}
            selected_room_types = [inv_room_type_map[rt] for rt in selected_room_types_es]
                
        with col_filter3:
            price_range = st.slider(
                "Rango de Precio (USD)",
                min_value=int(df['price'].min()),
                max_value=int(df['price'].max()),
                value=(int(df['price'].min()), int(df['price'].max()))
            )
        
        # Aplicar filtros
        filtered_df = df[
            (df['neighbourhood_group'].isin(selected_neighbourhoods)) &
            (df['room_type'].isin(selected_room_types)) &
            (df['price'] >= price_range[0]) &
            (df['price'] <= price_range[1])
        ]
        
        st.caption(f"📊 Mostrando {len(filtered_df):,} registros después de filtrar")
        
        # --- Gráficas ---
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("💰 Distribución de Precios")
            
            # Histograma de precios
            fig_price = px.histogram(
                filtered_df, 
                x='price',
                nbins=50,
                title='Distribución de Precios',
                labels={'price': 'Precio (USD)'}
            )
            fig_price.update_layout(showlegend=False)
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Precio promedio por distrito y tipo de habitación
            st.subheader("🏙️ Precio por Distrito y Tipo")

            

            price_summary = filtered_df.groupby(['neighbourhood_group', 'room_type'])['price'].mean().reset_index()
              

            fig_bar = px.bar(
                price_summary,
                x='neighbourhood_group',
                y='price',
                color='room_type',
                barmode='group',
                title='Precio Promedio por Distrito y Tipo de Habitación',
                labels={'price': 'Precio Promedio (USD)', 'neighbourhood_group': 'Distrito'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col_chart2:
            st.subheader("🏘️ Distribución por Tipo de Habitación")
            
            # Pie chart de room types
            room_type_counts = filtered_df['room_type'].value_counts()
            fig_pie = px.pie(
                values=room_type_counts.values,
                names=room_type_counts.index,
                title='Distribución por Tipo de Habitación'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Mapa de calor de correlaciones
            st.subheader("🔗 Correlaciones")
            numeric_cols = filtered_df.select_dtypes(include=['float64', 'int64']).columns
            correlation_matrix = filtered_df[numeric_cols].corr()
            
            fig_corr = px.imshow(
                correlation_matrix,
                title='Matriz de Correlación',
                aspect='auto',
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # --- Análisis de Reviews ---
        st.subheader("⭐ Análisis de Reviews")
        
        col_review1, col_review2 = st.columns(2)
        
        with col_review1:
            # Reviews por distrito
            reviews_by_neighbourhood = filtered_df.groupby('neighbourhood_group')['number_of_reviews'].sum().sort_values(ascending=False)
            fig_reviews = px.bar(
                x=reviews_by_neighbourhood.index,
                y=reviews_by_neighbourhood.values,
                title='Total de Reviews por Distrito',
                labels={'x': 'Distrito', 'y': 'Número de Reviews'}
            )
            st.plotly_chart(fig_reviews, use_container_width=True)
        
        with col_review2:
            # Relación precio vs reviews
            fig_scatter = px.scatter(
                filtered_df.sample(min(1000, len(filtered_df))),  # Muestra para performance
                x='price',
                y='number_of_reviews',
                color='room_type',
                title='Relación: Precio vs Número de Reviews',
                labels={'price': 'Precio (USD)', 'number_of_reviews': 'Número de Reviews'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # --- Tabla resumen ---
        st.subheader("📋 Resumen de Datos Filtrados")
        
        # Mostrar estadísticas descriptivas
        st.write("**Estadísticas Descriptivas:**")
        st.dataframe(filtered_df[['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'availability_365']].describe())
        
        
    except Exception as e:
        st.error(f"Error cargando los datos: {e}")