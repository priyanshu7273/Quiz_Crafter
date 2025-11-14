"""Predefined prompt templates used by the API responses."""
from __future__ import annotations

from app.config import get_settings


def system_prompt() -> str:
    return get_settings().defensive_system_prompt


def user_prompt() -> str:
    return get_settings().offensive_user_prompt
