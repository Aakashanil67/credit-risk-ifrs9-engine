import pytest

from src.decision_policy import DecisionPolicy


def test_expected_value_threshold_follows_the_margin_cost_and_lgd_assumptions():
    """At the break-even PD, approving earns zero expected value before any behavioural response."""
    policy = DecisionPolicy(
        margin_rate=0.12, operating_cost_rate=0.02, capital_cost_rate=0.02, lgd=0.45
    )

    assert policy.threshold == pytest.approx(0.08 / 0.57)
    assert policy.expected_value(policy.threshold, ead=100_000) == pytest.approx(0.0)


def test_expected_value_policy_declines_at_or_above_its_break_even_pd():
    """A positive NPV applicant approves; the break-even applicant does not."""
    policy = DecisionPolicy(
        margin_rate=0.12, operating_cost_rate=0.02, capital_cost_rate=0.02, lgd=0.45
    )

    assert policy.decision(policy.threshold - 0.001) == "approve"
    assert policy.decision(policy.threshold) == "decline"


def test_policy_rejects_rates_outside_zero_to_one():
    """Negative costs or an LGD above 100% are input errors, not policy settings."""
    with pytest.raises(ValueError, match="lgd"):
        DecisionPolicy(margin_rate=0.12, operating_cost_rate=0.02, capital_cost_rate=0.02, lgd=1.01)
