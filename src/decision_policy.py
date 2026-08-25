"""Illustrative approval policy derived from per-loan expected value."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionPolicy:
    margin_rate: float
    operating_cost_rate: float
    capital_cost_rate: float
    lgd: float

    def __post_init__(self) -> None:
        for name, rate in {
            "margin_rate": self.margin_rate,
            "operating_cost_rate": self.operating_cost_rate,
            "capital_cost_rate": self.capital_cost_rate,
            "lgd": self.lgd,
        }.items():
            if not 0 <= rate <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.margin_rate <= self.operating_cost_rate + self.capital_cost_rate:
            raise ValueError("margin_rate must exceed operating and capital costs")

    @property
    def threshold(self) -> float:
        return (self.margin_rate - self.operating_cost_rate - self.capital_cost_rate) / (
            self.margin_rate + self.lgd
        )

    def expected_value(self, pd_score: float, ead: float) -> float:
        if not 0 <= pd_score <= 1:
            raise ValueError("pd_score must be between 0 and 1")
        if ead <= 0:
            raise ValueError("ead must be positive")
        return ead * (
            (1 - pd_score) * self.margin_rate
            - pd_score * self.lgd
            - self.operating_cost_rate
            - self.capital_cost_rate
        )

    def decision(self, pd_score: float) -> str:
        return "decline" if pd_score >= self.threshold else "approve"
