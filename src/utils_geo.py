# src/utils_geo.py
from math import radians, sin, cos, asin, sqrt


def _haversine(lat1, lon1, lat2, lon2):
    """
    Distancia aproximada en km entre dos puntos de la Tierra usando fórmula de Haversine.
    """
    R = 6371.0  # radio de la Tierra en km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2.0) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


# Coordenadas de referencia en NYC
TIMES_SQ_LAT, TIMES_SQ_LON = 40.7580, -73.9855
WALL_ST_LAT, WALL_ST_LON = 40.7060, -74.0086


def distance_to_times_sq(lat, lon):
    return _haversine(lat, lon, TIMES_SQ_LAT, TIMES_SQ_LON)


def distance_to_wall_st(lat, lon):
    return _haversine(lat, lon, WALL_ST_LAT, WALL_ST_LON)
