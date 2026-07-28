"""tier2_classifier.py must never hit a real Postgres, load a real embedding
model, or make a real claude_cli call - every test here patches
`llm_task_router.tier2_classifier.vector_store` and `.claude_cli` directly,
matching the mocking style already used in test_router.py for tier-3."""

from unittest.mock import patch

from llm_task_router.schema import ProviderResult
from llm_task_router.tier2_classifier import Tier2Resolution, resolve_high_stakes, resolve_task_type
from llm_task_router.vector_store import NeighborMatch

_EMBEDDING = [0.1, 0.2, 0.3]

# --- resolve_task_type: NN-confident path ---------------------------------


def test_resolve_task_type_uses_nn_majority_when_four_of_five_agree():
    matches = [
        NeighborMatch(label="code_gen", similarity=0.9),
        NeighborMatch(label="code_gen", similarity=0.85),
        NeighborMatch(label="code_gen", similarity=0.8),
        NeighborMatch(label="code_gen", similarity=0.75),
        NeighborMatch(label="refactor", similarity=0.7),
    ]
    with (
        patch("llm_task_router.tier2_classifier.vector_store.nearest_neighbors", return_value=matches) as mock_nn,
        patch("llm_task_router.tier2_classifier.claude_cli.invoke") as mock_invoke,
        patch("llm_task_router.tier2_classifier.vector_store.insert_example") as mock_insert,
    ):
        result = resolve_task_type("write a function that validates emails", _EMBEDDING)

    assert result == Tier2Resolution(task_type="code_gen", source="nn")
    mock_nn.assert_called_once_with(_EMBEDDING, "task_type", k=5)
    mock_invoke.assert_not_called()
    mock_insert.assert_not_called()  # NN-confident matches are never written back


def test_resolve_task_type_falls_through_to_llm_when_only_three_of_five_agree():
    matches = [
        NeighborMatch(label="code_gen", similarity=0.9),
        NeighborMatch(label="code_gen", similarity=0.85),
        NeighborMatch(label="code_gen", similarity=0.8),
        NeighborMatch(label="refactor", similarity=0.75),
        NeighborMatch(label="architecture", similarity=0.7),
    ]
    with (
        patch("llm_task_router.tier2_classifier.vector_store.nearest_neighbors", return_value=matches),
        patch(
            "llm_task_router.tier2_classifier.claude_cli.invoke",
            return_value=ProviderResult(text="CODE_GEN", cost_usd=0.003, duration_ms=400),
        ) as mock_invoke,
        patch("llm_task_router.tier2_classifier.vector_store.insert_example") as mock_insert,
    ):
        result = resolve_task_type("write a function that validates emails", _EMBEDDING)

    assert result == Tier2Resolution(task_type="code_gen", source="llm_fallback")
    args, kwargs = mock_invoke.call_args
    assert args[1] == "haiku"
    assert kwargs["disable_tools"] is True
    assert "session_id" not in kwargs
    assert "on_event" not in kwargs
    mock_insert.assert_called_once_with(
        "write a function that validates emails",
        _EMBEDDING,
        task_type="code_gen",
        is_high_stakes=None,
        source="llm_fallback",
    )


def test_resolve_task_type_falls_through_to_llm_when_fewer_than_five_neighbors_exist():
    """Even a unanimous 2/2 shouldn't count as confident - the store barely
    has labeled examples yet for this axis, so a same-label pair isn't a
    real signal (see _nn_majority's docstring)."""
    matches = [
        NeighborMatch(label="triage", similarity=0.9),
        NeighborMatch(label="triage", similarity=0.85),
    ]
    with (
        patch("llm_task_router.tier2_classifier.vector_store.nearest_neighbors", return_value=matches),
        patch(
            "llm_task_router.tier2_classifier.claude_cli.invoke",
            return_value=ProviderResult(text="TRIAGE", cost_usd=0.003, duration_ms=400),
        ) as mock_invoke,
        patch("llm_task_router.tier2_classifier.vector_store.insert_example"),
    ):
        result = resolve_task_type("debug the null pointer", _EMBEDDING)

    assert result == Tier2Resolution(task_type="triage", source="llm_fallback")
    mock_invoke.assert_called_once()


def test_resolve_task_type_falls_through_to_llm_when_no_neighbors_exist():
    with (
        patch("llm_task_router.tier2_classifier.vector_store.nearest_neighbors", return_value=[]),
        patch(
            "llm_task_router.tier2_classifier.claude_cli.invoke",
            return_value=ProviderResult(text="ARCHITECTURE", cost_usd=0.003, duration_ms=400),
        ) as mock_invoke,
        patch("llm_task_router.tier2_classifier.vector_store.insert_example"),
    ):
        result = resolve_task_type("design a new system", _EMBEDDING)

    assert result == Tier2Resolution(task_type="architecture", source="llm_fallback")
    mock_invoke.assert_called_once()


# --- resolve_task_type: LLM fallback failure paths ------------------------


def test_resolve_task_type_returns_none_when_llm_unavailable_and_nn_unconfident():
    with (
        patch("llm_task_router.tier2_classifier.vector_store.nearest_neighbors", return_value=[]),
        patch(
            "llm_task_router.tier2_classifier.claude_cli.invoke",
            return_value=ProviderResult(text="", cost_usd=0.0, duration_ms=0, error="mocked provider error"),
        ),
        patch("llm_task_router.tier2_classifier.vector_store.insert_example") as mock_insert,
    ):
        result = resolve_task_type("something ambiguous", _EMBEDDING)

    assert result is None
    mock_insert.assert_not_called()


def test_resolve_task_type_returns_none_on_unparseable_llm_text():
    with (
        patch("llm_task_router.tier2_classifier.vector_store.nearest_neighbors", return_value=[]),
        patch(
            "llm_task_router.tier2_classifier.claude_cli.invoke",
            return_value=ProviderResult(text="uh, not sure", cost_usd=0.003, duration_ms=400),
        ),
        patch("llm_task_router.tier2_classifier.vector_store.insert_example") as mock_insert,
    ):
        result = resolve_task_type("something ambiguous", _EMBEDDING)

    assert result is None
    mock_insert.assert_not_called()


def test_resolve_task_type_returns_none_on_raised_exception():
    with (
        patch("llm_task_router.tier2_classifier.vector_store.nearest_neighbors", return_value=[]),
        patch("llm_task_router.tier2_classifier.claude_cli.invoke", side_effect=FileNotFoundError("no claude binary")),
        patch("llm_task_router.tier2_classifier.vector_store.insert_example") as mock_insert,
    ):
        result = resolve_task_type("something ambiguous", _EMBEDDING)

    assert result is None
    mock_insert.assert_not_called()


# --- resolve_high_stakes ---------------------------------------------------


def test_resolve_high_stakes_uses_nn_majority_when_confident():
    matches = [NeighborMatch(label=True, similarity=0.9)] * 4 + [NeighborMatch(label=False, similarity=0.6)]
    with (
        patch("llm_task_router.tier2_classifier.vector_store.nearest_neighbors", return_value=matches) as mock_nn,
        patch("llm_task_router.tier2_classifier.claude_cli.invoke") as mock_invoke,
        patch("llm_task_router.tier2_classifier.vector_store.insert_example") as mock_insert,
    ):
        result = resolve_high_stakes("migrate the k8s cluster to a new region", _EMBEDDING)

    assert result is True
    mock_nn.assert_called_once_with(_EMBEDDING, "is_high_stakes", k=5)
    mock_invoke.assert_not_called()
    mock_insert.assert_not_called()


def test_resolve_high_stakes_llm_fallback_yes_writes_back_with_task_type():
    with (
        patch("llm_task_router.tier2_classifier.vector_store.nearest_neighbors", return_value=[]),
        patch(
            "llm_task_router.tier2_classifier.claude_cli.invoke",
            return_value=ProviderResult(text="YES", cost_usd=0.003, duration_ms=400),
        ) as mock_invoke,
        patch("llm_task_router.tier2_classifier.vector_store.insert_example") as mock_insert,
    ):
        result = resolve_high_stakes("migrate the k8s cluster to a new region", _EMBEDDING, task_type="multi_step")

    assert result is True
    args, kwargs = mock_invoke.call_args
    assert args[1] == "haiku"
    assert kwargs["disable_tools"] is True
    mock_insert.assert_called_once_with(
        "migrate the k8s cluster to a new region",
        _EMBEDDING,
        task_type="multi_step",
        is_high_stakes=True,
        source="llm_fallback",
    )


def test_resolve_high_stakes_llm_fallback_no_writes_back_false():
    with (
        patch("llm_task_router.tier2_classifier.vector_store.nearest_neighbors", return_value=[]),
        patch(
            "llm_task_router.tier2_classifier.claude_cli.invoke",
            return_value=ProviderResult(text="NO", cost_usd=0.003, duration_ms=400),
        ),
        patch("llm_task_router.tier2_classifier.vector_store.insert_example") as mock_insert,
    ):
        result = resolve_high_stakes("what's a good design for this button?", _EMBEDDING)

    assert result is False
    mock_insert.assert_called_once_with(
        "what's a good design for this button?",
        _EMBEDDING,
        task_type=None,
        is_high_stakes=False,
        source="llm_fallback",
    )


def test_resolve_high_stakes_returns_none_when_llm_unavailable():
    with (
        patch("llm_task_router.tier2_classifier.vector_store.nearest_neighbors", return_value=[]),
        patch(
            "llm_task_router.tier2_classifier.claude_cli.invoke",
            return_value=ProviderResult(text="", cost_usd=0.0, duration_ms=0, error="mocked"),
        ),
        patch("llm_task_router.tier2_classifier.vector_store.insert_example") as mock_insert,
    ):
        result = resolve_high_stakes("ambiguous description", _EMBEDDING)

    assert result is None
    mock_insert.assert_not_called()


def test_resolve_high_stakes_returns_none_on_raised_exception():
    with (
        patch("llm_task_router.tier2_classifier.vector_store.nearest_neighbors", return_value=[]),
        patch("llm_task_router.tier2_classifier.claude_cli.invoke", side_effect=FileNotFoundError("no claude binary")),
        patch("llm_task_router.tier2_classifier.vector_store.insert_example") as mock_insert,
    ):
        result = resolve_high_stakes("ambiguous description", _EMBEDDING)

    assert result is None
    mock_insert.assert_not_called()
