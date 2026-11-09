import logging

from mmar_carl.logging_utils import log_step_complete


def test_failed_step_log_includes_identity_and_reason(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="mmar_carl"):
        log_step_complete(
            2,
            False,
            0.004,
            step_title="Review answer",
            step_type="llm",
            error_message="Missing required input key 'draft'\nfrom step 1",
        )

    assert "Step 2 'Review answer' (llm) failed in 0.00s" in caplog.text
    assert "Missing required input key 'draft' from step 1" in caplog.text


def test_failed_step_log_redacts_bearer_tokens(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="mmar_carl"):
        log_step_complete(
            1,
            False,
            1.0,
            error_message="Authorization: Bearer secret-token-value",
        )

    assert "secret-token-value" not in caplog.text
    assert "[REDACTED]" in caplog.text
