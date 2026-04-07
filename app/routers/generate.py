from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class GenerateRequest(BaseModel):
    prompt: str
    system: str | None = None
    max_tokens: int = 120
    temperature: float = 0.9
    model: str | None = None


class GenerateResponse(BaseModel):
    text: str
    provider: str
    model: str


@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    # TODO: provider routing (Mistral → HF fallback)
    raise NotImplementedError
