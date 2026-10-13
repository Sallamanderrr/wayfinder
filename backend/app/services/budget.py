"""Budget tables computed in code, with exact decimal arithmetic.

Live runs showed the model doing per-line arithmetic in its head (off by a few cents,
adjusted so the rows matched a converted total) even when told to use the calculator.
So the planner hands over the line items and this module does all the maths: per-person
price x travellers, conversion, totals and the budget check.

Totals are sums of the rounded lines, so the table always adds up.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")


@dataclass(frozen=True)
class BudgetLine:
    name: str
    price_per_person: float


def budget_table(
    lines: list[BudgetLine],
    travellers: int,
    currency: str,
    *,
    target_currency: str | None = None,
    rate: float = 1.0,
    rate_date: str = "",
    budget: float | None = None,
) -> str:
    """Render the budget as a markdown table plus a budget verdict."""
    if travellers < 1:
        raise ValueError("travellers must be at least 1.")
    if not lines:
        raise ValueError("Give at least one item.")
    converting = target_currency is not None and target_currency != currency
    shown_currency = target_currency if converting else currency
    exact_rate = Decimal(str(rate))

    header = f"| Item | Per person ({currency}) | For {travellers} ({currency}) |"
    divider = "|---|---|---|"
    if converting:
        header += f" For {travellers} ({target_currency}) |"
        divider += "---|"
    rows = [header, divider]

    total = Decimal(0)
    total_converted = Decimal(0)
    for line in lines:
        per_person = _money(line.price_per_person)
        group = _round(per_person * travellers)
        total += group
        row = f"| {line.name} | {_fmt(per_person)} | {_fmt(group)} |"
        if converting:
            converted = _round(group * exact_rate)
            total_converted += converted
            row += f" {_fmt(converted)} |"
        rows.append(row)

    total_row = f"| **Total** | | **{_fmt(total)}** |"
    if converting:
        total_row += f" **{_fmt(total_converted)}** |"
    rows.append(total_row)

    title = f"Budget for {travellers} traveller{'s' if travellers > 1 else ''}"
    if converting:
        title += f" (rate 1 {currency} = {exact_rate} {target_currency}"
        title += f", reference rate of {rate_date})" if rate_date else ")"
    parts = [title + ":", "", *rows]

    if budget is not None:
        budget_amount = _money(budget)
        spent = total_converted if converting else total
        difference = budget_amount - spent
        verdict = (
            f"Within budget: {_fmt(difference)} {shown_currency} left of {_fmt(budget_amount)} {shown_currency}."
            if difference >= 0
            else f"Over budget by {_fmt(-difference)} {shown_currency} (budget {_fmt(budget_amount)} {shown_currency})."
        )
        parts += ["", verdict]
    return "\n".join(parts)


def _money(value: float) -> Decimal:
    return _round(Decimal(str(value)))


def _round(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def _fmt(value: Decimal) -> str:
    return f"{value:,.2f}"
