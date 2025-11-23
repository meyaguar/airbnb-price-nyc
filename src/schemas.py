# src/schemas.py
from typing import Optional

from pydantic import BaseModel, Field


class PropertyInput(BaseModel):
    neighbourhood_group: str = Field(..., description="Group: Manhattan, Brooklyn, etc.")
    neighbourhood: Optional[str] = Field(
        None, description="Neighbourhood específico (si se conoce)"
    )
    room_type: str = Field(..., description="Tipo de habitación/listing")
    latitude: float
    longitude: float
    minimum_nights: int
    number_of_reviews: int
    reviews_per_month: float
    calculated_host_listings_count: int
    availability_365: int


class PredictionResponse(BaseModel):
    predicted_price: float
    currency: str = "USD"
