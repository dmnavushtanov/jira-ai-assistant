from fastapi import FastAPI
from pydantic import BaseModel
import logging

from src.configs import load_config, setup_logging
from src.agents.router_agent import RouterAgent

config = load_config()
setup_logging(config)
logger = logging.getLogger(__name__)

app = FastAPI(title="Jira AI Assistant")
router_agent = RouterAgent()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

@app.get("/health")
def health() -> dict[str, str]:
    """Simple healthcheck endpoint"""
    return {"status": "ok"}

@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest) -> AnswerResponse:
    """Return the assistant answer for a question."""
    logger.info("Received question via API: %s", req.question)
    answer = router_agent.ask(req.question)
    return AnswerResponse(answer=answer)

