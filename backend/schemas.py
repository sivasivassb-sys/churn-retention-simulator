from typing import Literal, Optional
from pydantic import BaseModel, Field

YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]
MultipleLinesOption = Literal["Yes", "No", "No phone service"]
InternetOption = Literal["DSL", "Fiber optic", "No"]
ContractOption = Literal["Month-to-month", "One year", "Two year"]
PaymentOption = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


class CustomerFeatures(BaseModel):
    """Raw 19-feature Telco customer profile matching src/preprocessing.py."""

    gender: Literal["Male", "Female"] = "Female"
    SeniorCitizen: Literal[0, 1] = 0
    Partner: YesNo = "No"
    Dependents: YesNo = "No"
    tenure: int = Field(default=3, ge=0, le=120)
    PhoneService: YesNo = "Yes"
    MultipleLines: MultipleLinesOption = "No"
    InternetService: InternetOption = "Fiber optic"
    OnlineSecurity: InternetAddon = "No"
    OnlineBackup: InternetAddon = "No"
    DeviceProtection: InternetAddon = "No"
    TechSupport: InternetAddon = "No"
    StreamingTV: InternetAddon = "Yes"
    StreamingMovies: InternetAddon = "Yes"
    Contract: ContractOption = "Month-to-month"
    PaperlessBilling: YesNo = "Yes"
    PaymentMethod: PaymentOption = "Electronic check"
    MonthlyCharges: float = Field(default=89.5, ge=0.0, le=500.0)
    TotalCharges: float = Field(default=268.5, ge=0.0, le=20000.0)


class CustomerChanges(BaseModel):
    """Optional retention levers applied in the What-If simulator."""

    Partner: Optional[YesNo] = None
    Dependents: Optional[YesNo] = None
    tenure: Optional[int] = Field(default=None, ge=0, le=120)
    PhoneService: Optional[YesNo] = None
    MultipleLines: Optional[MultipleLinesOption] = None
    InternetService: Optional[InternetOption] = None
    OnlineSecurity: Optional[InternetAddon] = None
    OnlineBackup: Optional[InternetAddon] = None
    DeviceProtection: Optional[InternetAddon] = None
    TechSupport: Optional[InternetAddon] = None
    StreamingTV: Optional[InternetAddon] = None
    StreamingMovies: Optional[InternetAddon] = None
    Contract: Optional[ContractOption] = None
    PaperlessBilling: Optional[YesNo] = None
    PaymentMethod: Optional[PaymentOption] = None
    MonthlyCharges: Optional[float] = Field(default=None, ge=0.0, le=500.0)
    TotalCharges: Optional[float] = Field(default=None, ge=0.0, le=20000.0)


class SimulateRequest(BaseModel):
    """Payload for comparing baseline vs. modified retention scenario."""

    customer: CustomerFeatures
    changes: CustomerChanges
    intervention_cost_usd: float = Field(default=120.0, ge=0.0, le=5000.0)
    horizon_months: int = Field(default=12, ge=1, le=60)
    top_k: int = Field(default=8, ge=1, le=25)