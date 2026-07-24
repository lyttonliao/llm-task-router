import pytest

from llm_task_router.classifier import TYPE_DOMAIN_GRID, classify, classify_description
from llm_task_router.schema import DOMAINS, TASK_TYPES


def test_grid_covers_every_type_and_domain_combination():
    for task_type in TASK_TYPES:
        assert task_type in TYPE_DOMAIN_GRID
        for domain in DOMAINS:
            assert domain in TYPE_DOMAIN_GRID[task_type]


def test_summarization_is_uniformly_low():
    assert all(classify("summarization", domain) == "L" for domain in DOMAINS)


def test_architecture_is_uniformly_high():
    assert all(classify("architecture", domain) == "H" for domain in DOMAINS)


def test_infra_code_gen_escalates_high():
    assert classify("code_gen", "infra") == "H"


def test_unknown_task_type_raises():
    with pytest.raises(ValueError):
        classify("not_a_real_type", "backend")


def test_unknown_domain_raises():
    with pytest.raises(ValueError):
        classify("triage", "not_a_real_domain")


def test_classify_description_infers_task_type_and_domain():
    classification = classify_description("Investigate an HTTP 500 error in the authentication API")

    assert classification.task_type == "triage"
    assert classification.domain == "backend"
    assert classification.task_type_source == "inferred"
    assert classification.domain_source == "inferred"


def test_classify_description_escalates_unknown_task_shape():
    classification = classify_description("Help with this request")

    assert classification.task_type == "architecture"
    assert classification.domain == "other"
    assert classification.task_type_source == "fallback"
    assert classification.domain_source == "fallback"


def test_classify_description_honors_caller_overrides():
    classification = classify_description("Summarize the frontend release notes", task_type="code_review", domain="data")

    assert classification.task_type == "code_review"
    assert classification.domain == "data"
    assert classification.task_type_source == "provided"
    assert classification.domain_source == "provided"
