"""Visualisation helpers producing embeddable artefacts."""
from __future__ import annotations

import base64
import io
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd

from app.utils.logger import configure_logger, log_extra

logger = configure_logger(__name__)


class Visualizer:
    """Generate lightweight static charts encoded as base64 PNG images."""

    async def bar_chart(self, df: pd.DataFrame, *, x: str, y: str, title: str) -> Dict[str, Any]:
        logger.info("Rendering bar chart", extra=log_extra(columns=[x, y]))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(df[x], df[y])
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        encoded = self._to_base64(fig)
        plt.close(fig)
        return {"type": "bar", "title": title, "image": encoded}

    async def line_chart(self, df: pd.DataFrame, *, x: str, y: str, title: str) -> Dict[str, Any]:
        logger.info("Rendering line chart", extra=log_extra(columns=[x, y]))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df[x], df[y], marker="o")
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        encoded = self._to_base64(fig)
        plt.close(fig)
        return {"type": "line", "title": title, "image": encoded}

    def _to_base64(self, fig) -> str:
        buffer = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buffer, format="png")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")
