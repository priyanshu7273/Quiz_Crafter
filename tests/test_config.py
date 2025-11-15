from __future__ import annotations

import os

import pytest

from app.config import Settings


def test_settings_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STUDENT_EMAIL", "student@example.com")
    monkeypatch.setenv("STUDENT_SECRET", "super-secret")
    settings = Settings()
    assert settings.student_email == "student@example.com"
    assert settings.student_secret == "super-secret"
    assert "Access denied" in settings.defensive_system_prompt


def test_available_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STUDENT_EMAIL", "student@example.com")
    monkeypatch.setenv("STUDENT_SECRET", "super-secret")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings()
    flags = settings.available_llm_backends()
    assert set(flags.keys()) == {"openai", "gemini", "perplexity", "anthropic"}
