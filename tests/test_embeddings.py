"""embeddings.embed() must never load the real sentence-transformers model
(slow, downloads weights on first use) - every test here patches
`llm_task_router.embeddings.SentenceTransformer` directly, matching the
subprocess/CLI double pattern already used in test_claude_cli.py."""

from unittest.mock import MagicMock, patch

from llm_task_router import embeddings


class _FakeArray:
    """Stand-in for the numpy array SentenceTransformer.encode() returns -
    exercises the .tolist() call embed() depends on to keep numpy out of this
    module's return type, without importing numpy in the test."""

    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return self._values


def _fake_model(encode_return=(0.1, 0.2, 0.3)):
    model = MagicMock()
    model.encode.return_value = _FakeArray(encode_return)
    return model


def setup_function(_fn):
    # embeddings._model is process-lifetime singleton state - reset before
    # every test so singleton-loads-once assertions aren't polluted by test
    # execution order.
    embeddings._model = None


def teardown_function(_fn):
    embeddings._model = None


def test_embed_returns_plain_list_not_numpy_array():
    fake_model = _fake_model([0.1, 0.2, 0.3])
    with patch("llm_task_router.embeddings.SentenceTransformer", return_value=fake_model):
        result = embeddings.embed("classify this description")

    assert result == [0.1, 0.2, 0.3]
    assert isinstance(result, list)
    fake_model.encode.assert_called_once_with("classify this description")


def test_embed_loads_model_with_expected_name_and_dimension():
    fake_model = _fake_model()
    with patch("llm_task_router.embeddings.SentenceTransformer", return_value=fake_model) as mock_cls:
        embeddings.embed("hello")

    mock_cls.assert_called_once_with(embeddings.MODEL_NAME)
    assert embeddings.MODEL_NAME == "all-MiniLM-L6-v2"


def test_embed_loads_model_only_once_across_multiple_calls():
    """The singleton contract: repeated embed() calls in one process must
    construct SentenceTransformer exactly once, not once per call."""
    fake_model = _fake_model()
    with patch("llm_task_router.embeddings.SentenceTransformer", return_value=fake_model) as mock_cls:
        embeddings.embed("first")
        embeddings.embed("second")
        embeddings.embed("third")

    mock_cls.assert_called_once()
    assert fake_model.encode.call_count == 3


def test_embed_reuses_model_instance_set_before_first_call():
    """Regression guard for the lazy-load branch itself: if _model is
    already populated, SentenceTransformer must not be constructed at all."""
    fake_model = _fake_model([0.9])
    embeddings._model = fake_model
    with patch("llm_task_router.embeddings.SentenceTransformer") as mock_cls:
        result = embeddings.embed("already warm")

    mock_cls.assert_not_called()
    assert result == [0.9]
