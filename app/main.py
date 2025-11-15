"""FastAPI application entry point."""
from __future__ import annotations

from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status

from app.config import Settings, get_settings
from app.core.quiz_processor import QuizProcessor
from app.models import APIResponse, QuizPayload
from app.prompts import templates
from app.utils.logger import configure_logger, log_extra

logger = configure_logger(__name__)

app = FastAPI(title="LLM Analysis Quiz")


@app.on_event("startup")
async def startup_event() -> None:  # pragma: no cover - startup/shutdown side effects
    """Initialise shared resources when the application boots."""
    app.state.quiz_processor = QuizProcessor()


@app.on_event("shutdown")
async def shutdown_event() -> None:  # pragma: no cover - startup/shutdown side effects
    """Gracefully close shared resources on application shutdown."""
    processor: Optional[QuizProcessor] = getattr(app.state, "quiz_processor", None)
    if processor is not None:
        await processor.close()


def get_processor(request: Request) -> QuizProcessor:
    processor: Optional[QuizProcessor] = getattr(request.app.state, "quiz_processor", None)
    if processor is None:
        raise RuntimeError("Quiz processor is not initialised")
    return processor


def get_config() -> Settings:
    return get_settings()


@app.post("/solve", response_model=APIResponse)
async def solve_quiz(
    payload: QuizPayload,
    background_tasks: BackgroundTasks,
    processor: QuizProcessor = Depends(get_processor),
    settings: Settings = Depends(get_config),
) -> APIResponse:
    if payload.secret != settings.student_secret or payload.email.lower() != settings.student_email.lower():
        logger.warning(
            "Rejected request due to invalid credentials",
            extra=log_extra(email=payload.email),
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid credentials")

    async def _run() -> None:
        try:
            result = await processor.process(payload)
            logger.info("Quiz solved", extra=log_extra(correct=result.correct, next_url=result.next_url))
        except Exception as exc:
            logger.exception("Quiz processing failed", extra=log_extra(error=str(exc)))

    background_tasks.add_task(_run)

    return APIResponse(
        status="accepted",
        detail="Quiz accepted for processing",
        defensive_system_prompt=templates.system_prompt(),
        offensive_user_prompt=templates.user_prompt(),
    )


@app.get("/healthz")
async def healthcheck(processor: QuizProcessor = Depends(get_processor)) -> dict:
    return {"status": "ok", "providers": processor.provider_stats()}
