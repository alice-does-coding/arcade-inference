from fastapi import FastAPI
from app.routers import generate, health

app = FastAPI(title="lurkr-inference", description="LLM inference service for Lurkr")

app.include_router(health.router)
app.include_router(generate.router)
