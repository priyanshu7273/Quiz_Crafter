"""Collection of deterministic analysis helpers used by the solver."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from app.utils.logger import configure_logger, log_extra

logger = configure_logger(__name__)


@dataclass
class AnalysisResult:
    """A light-weight wrapper for deterministic computations."""

    description: str
    value: Any


class DataAnalyzer:
    """Expose common aggregations used by quiz tasks."""

    async def aggregate(self, df: pd.DataFrame, operation: str, column: Optional[str] = None) -> AnalysisResult:
        logger.info("Running aggregate", extra=log_extra(operation=operation, column=column))
        column = column or df.columns[0]
        series = df[column].apply(self._coerce_numeric)
        if operation == "sum":
            return AnalysisResult(f"sum({column})", float(series.sum()))
        if operation == "mean":
            return AnalysisResult(f"mean({column})", float(series.mean()))
        if operation == "median":
            return AnalysisResult(f"median({column})", float(series.median()))
        if operation == "max":
            return AnalysisResult(f"max({column})", float(series.max()))
        if operation == "min":
            return AnalysisResult(f"min({column})", float(series.min()))
        raise ValueError(f"Unsupported operation: {operation}")

    async def filter_rows(self, df: pd.DataFrame, **criteria: Any) -> AnalysisResult:
        frame = df
        for column, expected in criteria.items():
            if isinstance(expected, Iterable) and not isinstance(expected, (str, bytes)):
                frame = frame[frame[column].isin(list(expected))]
            else:
                frame = frame[frame[column] == expected]
        logger.info("Filter applied", extra=log_extra(rows=len(frame)))
        return AnalysisResult("filtered_rows", frame)

    async def describe(self, df: pd.DataFrame) -> AnalysisResult:
        summary = df.describe(include="all").fillna(0)
        return AnalysisResult("describe", summary)

    def _coerce_numeric(self, value: Any) -> float:
        if isinstance(value, (int, float, np.number)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.replace(",", "")
            match = np.nan
            try:
                match = float(cleaned)
            except ValueError:
                match = np.nan
            return match
        return np.nan
