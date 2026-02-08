from pydantic import BaseModel

class ChurnInput(BaseModel):
    Age: int
    Gender: str
    Balance: float
    Tenure: int
    CreditScore: int
    NumOfProducts: int
    IsActiveMember: int
