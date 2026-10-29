from fastapi.testclient import TestClient

from app.schemas import EmailTriageResponse

from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "app": "InboxPilot"
    }
    
def test_urgent_email_is_classified_correctky():
    payload = {
        "sender": "ops@example.com",
        "subject": "Production database is down",
        "body": (
            "Our production database is unavailable. "
            "Customers cannot complete payments. "
            "Please investigate immediately."
        ),
    } 
    
    response = client.post("/emails/triage",json=payload)
    
    assert response.status_code == 200
    
    data = response.json()
    
    assert data["subject"] == payload["subject"]
    assert data["category"] == "work"
    assert data["priority"] == "high"
    assert isinstance(data["summary"], str)   
    
def test_empty_subject_is_rejected():
    payload = {
        "sender": "ops@example.com",
        "subject": "",
        "body": "Please investigate.",
    }    
    
    response = client.post("/emails/triage",json=payload)
    
    assert response.status_code == 422
    
def test_missing_body_is_rejected():
    payload = {
        "sender": "ops@example.com",
        "subject": "Meeting tomorrow"
    }    
    
    response = client.post("/emails/triage",json=payload)
    
    assert response.status_code == 422
    
def test_response_has_expected_fields():
    payload = {
        "sender": "friend@example.com",
        "subject": "Weekend plans",
        "body": "Want to meet this weekend?",
    }    
    
    response = client.post("/emails/triage",json=payload)
    
    assert response.status_code == 200
    
    assert set(response.json().keys()) == {
        "subject",
        "category",
        "priority",
        "summary",
    }
    
def test_non_urgent_email_gets_normal_priority():
    payload = {
        "sender": "team@example.com",
        "subject": "Meeting notes",
        "body": "The client said the issue is not urgent anymore"
    }    
    
    response = client.post("/emails/triage",json=payload)
    
    assert response.status_code == 200
    assert response.json()["priority"] == "normal"
    
def test_urgent_outage_gets_high_priority():
    payload = {
        "sender": "ops@example.com",
        "subject": "Production database is down",
        "body": "Customers cannot complete payments"
    }    
    
    response = client.post("/emails/triage",json=payload)
    
    assert response.status_code == 200
    assert response.json()["priority"] == "high"
    
def test_personal_email_gets_personal_category():
    payload = {
        "sender": "friends@example.com",
        "subject": "Weekend plans",
        "body": "Want to watch a movie this weekend?"
    }    
    
    response = client.post("/emails/triage", json=payload)
    
    assert response.status_code == 200
    assert response.json()["category"] == "personal"
    
def test_empty_body_is_rejected():
    payload = {
        "sender": "ops@example.com",
        "subject": "Server update",
        "body": ""
    }    
    
    response = client.post("/emails/triage",json=payload)
    
    assert response.status_code == 422
    
def test_triage_endpoint_with_mocked_llm(monkeypatch):
    from app.api import routes
    
    def fake_triage(email):
        return EmailTriageResponse(
            subject = email.subject,
            category="work",
            priority="high",
            summary="A production issue needs attention."
        )    
        
    monkeypatch.setattr(
        routes,
        "triage_with_llm",
        fake_triage
    )
    
    payload = {
        "sender": "ops@example.com",
        "subject": "Production issue",
        "body": "Payments are failing.",
    }
    
    response = client.post("/emails/triage",json=payload)
    
    assert response.status_code == 200
    
    assert response.json()["category"] == "work"
    assert response.json()["priority"] == "high"
    assert response.json()["summary"] == (
        "A production issue needs attention."
    ) 