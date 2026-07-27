from llm_task_router.known_models import known_models_for


def test_known_models_for_claude():
    assert known_models_for("claude") == ("haiku", "sonnet", "opus")


def test_known_models_for_codex():
    assert "gpt-5.5" in known_models_for("codex")


def test_known_models_for_unknown_provider_returns_empty_tuple():
    assert known_models_for("nonexistent") == ()
