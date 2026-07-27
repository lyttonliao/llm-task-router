import pytest

from llm_task_router.classifier import TYPE_DOMAIN_GRID, classify, classify_description, has_high_stakes_signal
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


def test_code_gen_is_uniformly_low():
    assert all(classify("code_gen", domain) == "L" for domain in DOMAINS)


def test_refactor_is_uniformly_low():
    assert all(classify("refactor", domain) == "L" for domain in DOMAINS)


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


def test_has_high_stakes_signal_true_for_compliance_and_scale_vocabulary():
    assert has_high_stakes_signal("this touches customer data and has a strict compliance requirement")
    assert has_high_stakes_signal("we need multi-region disaster recovery for this")
    assert has_high_stakes_signal("a production outage would mean real downtime for customers")


def test_has_high_stakes_signal_false_for_generic_shape_words_alone():
    """The whole point of this gate: "design"/"scalable"/"fault-tolerant"
    alone are shape words, not stakes words - a trivial question can use
    them just as readily as a real one."""
    assert not has_high_stakes_signal("design a scalable, fault-tolerant system for this workload")
    assert not has_high_stakes_signal("what's a good design for this button component?")


def test_classify_description_honors_caller_overrides():
    classification = classify_description("Summarize the frontend release notes", task_type="code_review", domain="data")

    assert classification.task_type == "code_review"
    assert classification.domain == "data"
    assert classification.task_type_source == "provided"
    assert classification.domain_source == "provided"
