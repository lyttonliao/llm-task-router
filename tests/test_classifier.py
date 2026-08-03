import pytest

from llm_task_router.classifier import (
    TYPE_DOMAIN_GRID,
    classify,
    classify_description,
    extract_instruction,
    has_high_stakes_signal,
    has_non_engineering_signal,
)
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


def test_extract_instruction_splits_on_blank_line():
    description = "review this before I ship it\n\nkubernetes production ETL pipeline bullet points"
    assert extract_instruction(description) == "review this before I ship it"


def test_extract_instruction_splits_on_single_newline_without_blank_line():
    description = "review this\nkubernetes production ETL pipeline bullet points"
    assert extract_instruction(description) == "review this"


def test_extract_instruction_passthrough_for_single_line():
    description = "okay one last review of my resume"
    assert extract_instruction(description) == description


def test_has_non_engineering_signal_true_for_resume_and_proofread_wording():
    assert has_non_engineering_signal("okay one last review of my resume")
    assert has_non_engineering_signal("can you proofread my cover letter")
    assert has_non_engineering_signal("check the wording and grammar in this")


def test_has_non_engineering_signal_false_for_genuine_code_review():
    assert not has_non_engineering_signal("review this pull request for the auth service")


def test_pasted_content_does_not_leak_domain_or_high_stakes():
    """Real-incident regression (routing_decisions ids 55/63): pasting resume
    bullets containing infra/impact vocabulary below a one-line ask must not
    push domain into infra/data or trip the high-stakes gate - only the
    instruction line should be scanned for keyword signal."""
    description = (
        "can you review this before I send it\n\n"
        "Engineered a kubernetes deployment pipeline for production workloads at scale"
    )
    classification = classify_description(description)

    assert classification.domain == "other"
    assert not has_high_stakes_signal(description)


def test_legitimate_single_line_domain_and_impact_signal_still_fires():
    """The fix scopes matching to the instruction, not to a shorter string -
    a single-line message with real domain/impact vocabulary must classify
    and escalate exactly as before."""
    description = "design the kubernetes rollout for production"
    classification = classify_description(description)

    assert classification.domain == "infra"
    assert has_high_stakes_signal(description)
