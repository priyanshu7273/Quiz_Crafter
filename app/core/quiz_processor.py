"""End-to-end quiz chain processor."""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Optional

import pandas as pd

import aiohttp

from app.config import get_settings
from app.core.browser_manager import BrowserManager
from app.core.llm_manager import LLMManager
from app.models import QuizPayload, QuizProcessingResult, QuizSubmission, SubmissionResponse
from app.solvers.base_solver import QuizSolver, SolverContext
from app.utils.helpers import run_with_retry, time_budget
from app.utils.logger import configure_logger, log_extra

logger = configure_logger(__name__)


@dataclass
class QuizState:
    url: str
    attempts: int = 0
    solved: bool = False


class QuizProcessor:
    """Coordinates the tooling required to solve chained quiz tasks."""

    def __init__(self, session: Optional[aiohttp.ClientSession] = None) -> None:
        self._settings = get_settings()
        self._session = session or aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._settings.request_timeout_seconds))
        self._llm = LLMManager()
        self._solver = QuizSolver(self._llm)

    async def close(self) -> None:
        await self._session.close()

    async def process(self, payload: QuizPayload) -> QuizProcessingResult:
        """Process the initial quiz URL until the chain finishes or time expires."""

        logger.info(
            "Starting quiz processing",
            extra=log_extra(url=str(payload.url), email=payload.email),
        )

        async with time_budget(self._settings.quiz_timeout_seconds) as timer:
            current = QuizState(url=str(payload.url))
            last_response: Optional[SubmissionResponse] = None
            last_submission: Optional[QuizSubmission] = None
            solution: Dict[str, Any] = {}
            async with BrowserManager() as browser:
                while current.url:
                    remaining = timer.remaining(timedelta(seconds=self._settings.quiz_timeout_seconds))
                    logger.info(
                        "Solving quiz step",
                        extra=log_extra(url=current.url, remaining=str(remaining)),
                    )
                    solution = await self._solve_single(browser, current.url)
                    submission = QuizSubmission(
                        email=payload.email,
                        secret=payload.secret,
                        url=current.url,
                        answer=solution["answer"],
                        metadata=solution.get("metadata", {}),
                    )
                    last_submission = submission
                    current.attempts += 1
                    last_response = await self._submit(submission)
                    if last_response.correct:
                        current.solved = True
                        if not last_response.url:
                            break
                        current = QuizState(url=str(last_response.url))
                    elif last_response.url:
                        logger.info("Received follow-up URL despite incorrect answer")
                        current = QuizState(url=str(last_response.url))
                    else:
                        logger.warning("Answer incorrect; stopping chain")
                        break

        if not last_response or not last_submission:
            raise RuntimeError("No response received from quiz server")

        return QuizProcessingResult(
            correct=bool(last_response.correct),
            answer=last_submission.answer,
            explanation=solution.get("explanation", ""),
            next_url=last_response.url,
            raw_response=last_response.dict(),
        )

    def provider_stats(self) -> Dict[str, Any]:
        return self._llm.stats()

    async def _solve_single(self, browser: BrowserManager, url: str) -> Dict[str, Any]:
        page_data = await browser.render(url)
        parsed_documents: Dict[str, Any] = {"page_text": page_data["text"]}
        interim_results: Dict[str, Any] = {}

        # Parse embedded base64 download hints
        downloads = self._extract_download_links(page_data["html"])
        for name, download_url in downloads.items():
            content = await self._download(download_url)
            parsed = await self._solver.parser.parse(content)
            parsed_documents[name] = self._stringify_document(parsed)
            dataframe = self._first_table(parsed)
            if dataframe is not None and not dataframe.empty:
                try:
                    aggregate = await self._solver.analyzer.aggregate(
                        dataframe,
                        operation="sum",
                        column=dataframe.columns[-1],
                    )
                    interim_results[f"sum_{name}"] = aggregate.value
                except Exception as exc:
                    logger.warning(
                        "Failed to aggregate table",
                        extra=log_extra(name=name, error=str(exc)),
                    )

        plan = await self._solver.plan(page_data["text"])
        answer_llm = await self._solver.derive_answer(
            SolverContext(raw_page=page_data, parsed_documents=parsed_documents, interim_results=interim_results),
            question=page_data["text"],
        )
        try:
            structured = json.loads(answer_llm["raw"])
        except json.JSONDecodeError:
            structured = {"answer": answer_llm["raw"], "confidence": 0.0, "rationale": "LLM response not JSON"}
        return {
            "answer": structured.get("answer"),
            "metadata": {
                "confidence": structured.get("confidence"),
                "rationale": structured.get("rationale"),
                "llm_provider": answer_llm["provider"],
                "llm_model": answer_llm["model"],
                "plan": plan,
                "downloads": list(downloads.keys()),
            },
            "explanation": structured.get("rationale", ""),
        }

    async def _submit(self, submission: QuizSubmission) -> SubmissionResponse:
        async def _post() -> SubmissionResponse:
            logger.info("Submitting answer", extra=log_extra(url=submission.url))
            async with self._session.post(
                str(submission.url),
                json=submission.dict(),
            ) as response:
                data = await response.json()
                return SubmissionResponse(**data)

        return await run_with_retry(
            _post,
            attempts=self._settings.max_retries,
            initial_delay=self._settings.retry_backoff_seconds,
        )

    async def _download(self, url: str) -> bytes:
        logger.info("Downloading artefact", extra=log_extra(url=url))
        async with self._session.get(url) as response:
            response.raise_for_status()
            return await response.read()

    def _extract_download_links(self, html: str) -> Dict[str, str]:
        pattern = re.compile(r"download data-url=\"(?P<url>[^\"]+)\" data-name=\"(?P<name>[^\"]+)\"")
        matches = pattern.finditer(html)
        downloads = {match.group("name"): match.group("url") for match in matches}
        if downloads:
            logger.info("Found downloadable artefacts", extra=log_extra(count=len(downloads)))
        return downloads

    def _first_table(self, parsed: Any) -> Optional[pd.DataFrame]:
        """Extract the first tabular structure from parsed artefacts if present."""

        if isinstance(parsed, pd.DataFrame):
            return parsed

        if isinstance(parsed, dict):
            tables = parsed.get("tables")
            if isinstance(tables, list) and tables:
                first = tables[0]
                dataframe = self._coerce_table(first)
                if dataframe is not None:
                    return dataframe
            if "pages" in parsed and isinstance(parsed["pages"], list):
                for page in parsed["pages"]:
                    if isinstance(page, dict):
                        tables = page.get("tables")
                        if isinstance(tables, list) and tables:
                            dataframe = self._coerce_table(tables[0])
                            if dataframe is not None:
                                return dataframe
        return None

    def _coerce_table(self, table: Any) -> Optional[pd.DataFrame]:
        """Attempt to convert a generic table structure into a DataFrame."""

        if isinstance(table, pd.DataFrame):
            return table
        if isinstance(table, list) and table:
            first_row = table[0]
            if isinstance(first_row, dict):
                return pd.DataFrame(table)
            if isinstance(first_row, (list, tuple)):
                return pd.DataFrame(table)
        if isinstance(table, dict):
            headers = table.get("headers")
            rows = table.get("rows")
            if headers and rows:
                return pd.DataFrame(rows, columns=headers)
            data = table.get("data")
            if isinstance(data, list) and data:
                return pd.DataFrame(data)
        return None

    def _stringify_document(self, parsed: Any) -> str:
        """Render parsed artefacts into a readable string for LLM context."""

        if isinstance(parsed, pd.DataFrame):
            return parsed.to_csv(index=False)
        if isinstance(parsed, dict):
            text = parsed.get("text")
            if isinstance(text, str) and text.strip():
                return text
            try:
                return json.dumps(parsed, indent=2, default=str)
            except TypeError:
                return str(parsed)
        if isinstance(parsed, (list, tuple)):
            try:
                return json.dumps(parsed, indent=2, default=str)
            except TypeError:
                return str(parsed)
        if isinstance(parsed, bytes):
            return parsed.decode("utf-8", errors="ignore")
        return str(parsed)
