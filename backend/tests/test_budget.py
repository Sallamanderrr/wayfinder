from __future__ import annotations

import pytest

from app.checks import unsupported_amounts
from app.services.budget import BudgetLine, budget_table

LISBON = [
    BudgetLine("National Tile Museum", 8),
    BudgetLine("Tram 28", 3.1),
    BudgetLine("Jerónimos Monastery", 18),
    BudgetLine("Lisbon Oceanarium", 25),
]


# --- budget_table ------------------------------------------------------------------------


def test_budget_table_converts_each_line_and_adds_up():
    # The exact plan from a live run where the model got three rows wrong by hand.
    table = budget_table(LISBON, 2, "EUR", target_currency="SEK", rate=11.2006, rate_date="2026-10-13", budget=1500)
    assert "| National Tile Museum | 8.00 | 16.00 | 179.21 |" in table  # model wrote 179.20
    assert "| Jerónimos Monastery | 18.00 | 36.00 | 403.22 |" in table  # model wrote 403.21
    assert "| Lisbon Oceanarium | 25.00 | 50.00 | 560.03 |" in table  # model wrote 560.00
    assert "| **Total** | | **108.20** | **1,211.90** |" in table  # sum of the rounded rows
    assert "Within budget: 288.10 SEK left of 1,500.00 SEK." in table
    assert "rate 1 EUR = 11.2006 SEK, reference rate of 2026-10-13" in table


def test_budget_table_without_conversion_and_over_budget():
    table = budget_table([BudgetLine("Louvre Museum", 22)], 3, "EUR", budget=50)
    assert "| Item | Per person (EUR) | For 3 (EUR) |" in table
    assert "| **Total** | | **66.00** |" in table
    assert "Over budget by 16.00 EUR (budget 50.00 EUR)." in table


def test_budget_table_uses_exact_decimals():
    # 0.1 + 0.2 style float errors must not leak into money.
    table = budget_table([BudgetLine("A", 0.1), BudgetLine("B", 0.2)], 1, "EUR")
    assert "| **Total** | | **0.30** |" in table


@pytest.mark.parametrize(("lines", "travellers"), [([], 2), (LISBON, 0)])
def test_budget_table_rejects_bad_input(lines, travellers):
    with pytest.raises(ValueError):
        budget_table(lines, travellers, "EUR")


def test_budget_tool_fetches_the_rate(planning_tools):
    tool = next(t for t in planning_tools if t.name == "budget_breakdown")
    result = tool.invoke(
        {
            "items": [{"name": "Vasa Museum", "price_per_person": 230}],
            "travellers": 2,
            "currency_code": "SEK",
            "budget": 50,
            "budget_currency": "EUR",
        }
    )
    assert "| Vasa Museum | 230.00 | 460.00 | 41.08 |" in result  # 460 x 0.0893 = 41.078
    assert "Within budget: 8.92 EUR left" in result
    assert tool.invoke({"items": [], "travellers": 2, "currency_code": "EUR"}).startswith("Error:")


# --- unsupported_amounts -----------------------------------------------------------------


def test_flags_amounts_no_tool_produced():
    draft = "| Tile | 179.20 |\n| Tram | 69.44 |\n| Total | 1,211.90 |\nRate 11.2006, 1.5 h, on 2026-10-17."
    sources = ["108.2 EUR = 1,211.90 SEK (rate 11.2006)", "16 * 11.2006 = 179.2096", "6.2 * 11.2006 = 69.44372"]
    # 179.2096 rounds to 179.21, so 179.20 was made up; 69.44 and 1,211.90 are backed by tools.
    assert unsupported_amounts(draft, sources) == ["179.20"]


def test_ignores_numbers_that_are_not_two_decimal_amounts():
    assert unsupported_amounts("Opens 9:00, 18 EUR, 1.5 h, rate 11.2006, 85% rain", []) == []
