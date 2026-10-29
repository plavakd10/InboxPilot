from fastapi import APIRouter, HTTPException
from openai import APIError, APITimeoutError

from app.schemas import EmailRequest, EmailTriageResponse
from app.services.llm_triage import triage_with_llm

router = APIRouter()

@router.post("/emails/triage",response_model=EmailTriageResponse)
def triage_email(email: EmailRequest) -> EmailTriageResponse:
    try:
        return triage_with_llm(email)
    except APITimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Email triage timed out. Please try again."
        )
        
    except APIError:
        raise HTTPException(
            status_code=502,
            detail="Email triage provider is temporary unavailable."
        )    
    
    except ValueError:
        raise HTTPException(
            status_code=502,
            detail="Email triage did not produce a usable result."
        )    