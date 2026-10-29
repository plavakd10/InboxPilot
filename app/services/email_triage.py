from app.schemas import EmailRequest, EmailTriageResponse

URGENT_KEYWORDS = [
        "urgent",
        "immediately",
        "outage",
        "production database is down",
        "cannot complete payments"
    ]
    
WORK_KEYWORDS = [
        "meeting",
        "project",
        "customer",
        "production"
    ]
    
NON_URGENT_PHASES = [
        "not urgent",
        "no longer urgent",
        "isn't urgent",
        "is not an emergency"
    ]

def classify_priority(text: str) -> str:
    for phrase in NON_URGENT_PHASES:
        if phrase in text:
            return "normal"
    
    if any(keyword in text for keyword in URGENT_KEYWORDS):
        return "high"
    
    return "normal"

def classify_category(text: str) -> str:
    if any(keyword in text for keyword in WORK_KEYWORDS):
        return "work"
    return "personal"
    


def classify_email(email: EmailRequest) -> EmailTriageResponse:
    text = f"{email.subject} {email.body}".lower()
    
    return EmailTriageResponse(
        subject=email.subject,
        category=classify_category(text),
        priority=classify_priority(text),
        summary=email.body[:200].strip()
    )
    
    
    