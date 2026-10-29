from typing import Literal

from pydantic import BaseModel, Field

class EmailRequest(BaseModel):
    sender: str = Field(min_lnegth=3, max_length=254)
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=10_000)
    
class EmailTriageResponse(BaseModel):
    subject: str
    category: Literal["work","personal"]
    priority: Literal["high","normal"]
    summary: str
            