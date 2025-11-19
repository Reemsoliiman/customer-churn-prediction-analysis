# src/api/schemas.py
from pydantic import BaseModel, Field
from typing import Literal


class ChurnInput(BaseModel):
    account_length: int = Field(..., alias="Account length", ge=1, le=250)
    international_plan: Literal["Yes", "No"] = Field(..., alias="International plan")
    voice_mail_plan: Literal["Yes", "No"] = Field(..., alias="Voice mail plan")
    number_vmail_messages: int = Field(..., alias="Number vmail messages", ge=0, le=100)
    total_day_minutes: float = Field(..., alias="Total day minutes", ge=0, le=400)
    total_eve_minutes: float = Field(..., alias="Total eve minutes", ge=0, le=400)
    total_night_minutes: float = Field(..., alias="Total night minutes", ge=0, le=400)
    total_intl_minutes: float = Field(..., alias="Total intl minutes", ge=0, le=30)
    total_day_calls: int = Field(..., alias="Total day calls", ge=0, le=200)
    total_eve_calls: int = Field(..., alias="Total eve calls", ge=0, le=200)
    total_night_calls: int = Field(..., alias="Total night calls", ge=0, le=200)
    total_intl_calls: int = Field(..., alias="Total intl calls", ge=0, le=30)
    customer_service_calls: int = Field(..., alias="Customer service calls", ge=0, le=10)

    model_config = {
        "populate_by_name": True,   # This allows using snake_case field names
        "extra": "ignore"
    }