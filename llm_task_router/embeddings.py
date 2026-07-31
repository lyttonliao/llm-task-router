"""Local embedding wrapper for the tier-2 continuous-learning classifier
(see /Users/lliao/.claude/plans/i-believe-for-now-smooth-cook.md for the full
design). Deliberately module-level, not wrapped in a class that needs
instantiation - `router.PROVIDERS` already learned the hard way (see
docs/auth-and-providers.md, "Adding a provider", point 2) that a
pre-grabbed function reference silently defeats `patch()` in tests; tests
here patch `llm_task_router.embeddings.SentenceTransformer` and
`llm_task_router.embeddings.embed` directly instead.

Runtime dependency reversal, scoped to this module (and vector_store.py):
this repo's "zero third-party dependencies" pillar applies to the CLI/
provider-adapter layer, not tier 2's classifier - `sentence-transformers`
(and the `torch` it pulls in) is a deliberate, permanent addition for the
local-embeddings tier, not a stray import.
"""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, small/fast - the standard
# lightweight default; see the plan doc's "Open judgment calls" for revisiting
# this only if embedding quality proves insufficient on real queries.

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-loads the singleton model on first call, reused for the process
    lifetime - avoids paying model-load cost (and, on first-ever run,
    a weights download) more than once per process."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(text: str) -> list[float]:
    """Returns the embedding as a plain list[float], not a numpy array -
    downstream callers (vector_store.py's SQL params, JSON-friendly test
    fixtures) treat this as a plain data structure; keeping numpy out of
    other modules' type signatures is deliberate."""
    model = _get_model()
    vector = model.encode(text)
    return vector.tolist()
