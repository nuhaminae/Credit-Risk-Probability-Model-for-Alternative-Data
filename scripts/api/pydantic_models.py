from pydantic import BaseModel

class RiskRequest(BaseModel):
    Amount: float
    Value: float
    txn_hour: int


class RiskResponse(BaseModel):
    risk_probability: float
