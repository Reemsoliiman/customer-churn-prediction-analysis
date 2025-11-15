from pydantic import BaseModel, Field
from typing import Literal


class ChurnInput(BaseModel):
    """
    Input schema for the /predict endpoint.
    JSON keys must match the original CSV column names (with spaces).
    """
    Account_length: int = Field(..., ge=1, le=250, alias="Account length")
    International_plan: Literal["Yes", "No"] = Field(..., alias="International plan")
    Voice_mail_plan: Literal["Yes", "No"] = Field(..., alias="Voice mail plan")
    Number_vmail_messages: int = Field(..., ge=0, le=100, alias="Number vmail messages")
    Total_day_minutes: float = Field(..., ge=0, le=400, alias="Total day minutes")
    Total_eve_minutes: float = Field(..., ge=0, le=400, alias="Total eve minutes")
    Total_night_minutes: float = Field(..., ge=0, le=400, alias="Total night minutes")
    Total_intl_minutes: float = Field(..., ge=0, le=30, alias="Total intl minutes")
    Total_day_calls: int = Field(..., ge=0, le=200, alias="Total day calls")
    Total_eve_calls: int = Field(..., ge=0, le=200, alias="Total eve calls")
    Total_night_calls: int = Field(..., ge=0, le=200, alias="Total night calls")
    Total_intl_calls: int = Field(..., ge=0, le=30, alias="Total intl calls")
    Customer_service_calls: int = Field(..., ge=0, le=10, alias="Customer service calls")

    class Config:
        populate_by_name = True