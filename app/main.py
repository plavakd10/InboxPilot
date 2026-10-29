from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title = "InboxPilot",
    description = "An AI-powered email triage API",
    version = "0.3.0",
)

app.include_router(router)

@app.get("/")
def health_check():
    return {"status": "healthy", "app": "InboxPilot"}
