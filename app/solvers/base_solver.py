"""Base solver that orchestrates parsing, analysis and final reasoning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.core.llm_manager import LLMManager
from app.solvers.data_parser import DataParser
from app.solvers.analyzer import DataAnalyzer
from app.solvers.visualizer import Visualizer
from app.utils.logger import configure_logger, log_extra

logger = configure_logger(__name__)


@dataclass
class SolverContext:
    """Aggregated context produced while processing a single quiz step."""

    raw_page: Dict[str, Any]
    parsed_documents: Dict[str, Any]
    interim_results: Dict[str, Any]


class QuizSolver:
    """High level helper that uses a mixture of deterministic tools and LLMs."""

    def __init__(self, llm_manager: LLMManager) -> None:
        self._llm = llm_manager
        self._parser = DataParser()
        self._analyzer = DataAnalyzer()
        self._visualizer = Visualizer()

    async def plan(self, instructions: str) -> str:
        """Ask the LLM to propose a step-by-step plan for solving the quiz."""

        prompt = (
            "You are an expert data scientist participating in a timed quiz. "
            "Summarise the following task, list the datasets or URLs to download, "
            "the analyses required and the final answer format. Present your plan "
            "as bullet points.\n\n"
            f"Task:\n{instructions}\n"
        )
        result = await self._llm.generate(prompt, system_prompt=(
            "You operate inside an automated agent. Provide concise actionable steps "
            "and do not hallucinate unavailable resources."
        ))
        logger.info("Generated quiz plan", extra=log_extra(provider=result.provider, model=result.model))
        return result.content

    async def derive_answer(self, context: SolverContext, question: str) -> Dict[str, Any]:
        """Combine deterministic results with reasoning from the LLM."""

        support_material = []
        for key, value in context.parsed_documents.items():
            support_material.append(f"## {key}\n{value}\n")
        for key, value in context.interim_results.items():
            support_material.append(f"## Result: {key}\n{value}\n")

        prompt = (
            "You receive structured data extracted from a quiz webpage. "
            "Use the evidence to compute the final answer. Return JSON with keys "
            "answer, confidence (0-1) and rationale. If unsure say answer is null.\n"
            f"Question: {question}\n"
            f"Context:\n{''.join(support_material)}"
        )
        result = await self._llm.generate(prompt, max_tokens=800)
        logger.info(
            "Generated answer candidate",
            extra=log_extra(provider=result.provider, model=result.model, latency=result.latency),
        )
        return {
            "raw": result.content,
            "provider": result.provider,
            "model": result.model,
            "latency": result.latency,
        }

    @property
    def parser(self) -> DataParser:
        return self._parser

    @property
    def analyzer(self) -> DataAnalyzer:
        return self._analyzer

    @property
    def visualizer(self) -> Visualizer:
        return self._visualizer
