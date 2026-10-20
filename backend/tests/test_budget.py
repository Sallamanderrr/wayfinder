from __future__ import annotations

import pytest

from app.checks import money_amounts, unsupported_amounts, unsupported_budget_inputs
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
    sources = ["108.2 EUR = 1,211.90 SEK (rate 11.2006)", "179.2096", "69.44372"]
    # 179.2096 rounds to 179.21, so 179.20 was made up; 69.44 and 1,211.90 are backed by tools.
    assert unsupported_amounts(draft, sources) == ["179.20"]


def test_money_amounts_are_two_decimal_numbers_or_numbers_next_to_a_currency():
    text = "€8 each, 16 EUR for two, SEK 330, 1,500 SEK, 69.44, meals €15–20, 2 people, 1.5 h, 85%, on 2026-10-17"
    assert money_amounts(text) == ["8", "16", "330", "1,500", "69.44", "15"]


def test_accepts_a_correct_answer_from_a_live_run():
    # The Paris museums answer, checked against the conversion results it was built from.
    draft = "Musée de l'Orangerie: 140.01 SEK (12.5 EUR)\nLouvre Museum: 246.41 SEK (22 EUR)\nThe exchange rate used is 1 EUR = 11.2006 SEK."
    sources = [
        "12.50 EUR = 140.01 SEK (rate 1 EUR = 11.2006 SEK, reference rate of 2026-10-13)",
        "22.00 EUR = 246.41 SEK (rate 1 EUR = 11.2006 SEK, reference rate of 2026-10-13)",
    ]
    assert unsupported_amounts(draft, sources) == []


def test_whole_amounts_are_checked_at_whole_unit_precision():
    # From a live run: "~78 SEK" for a 24-hour pass the tool converted to 77.85 SEK is honest
    # rounding. A two-decimal amount must still match to the cent.
    sources = ["7.00 EUR = 77.85 SEK (rate 1 EUR = 11.1221 SEK)"]
    assert unsupported_amounts("a 24-hour pass costs about €7 (~78 SEK)", sources) == []
    assert unsupported_amounts("a 24-hour pass costs 77.90 SEK", sources) == ["77.90"]
    assert unsupported_amounts("a 24-hour pass costs about 80 SEK", sources) == ["80"]


def test_flags_hand_computed_whole_amounts():
    # From a live run: the budget was converted by hand and the remainder made up.
    draft = "a budget of 1,500 SEK (~136 EUR) ... Within budget: 63.80 EUR left of 136.00 EUR (approx. 703 SEK left)"
    sources = ["Plan two days in Lisbon ... Our budget for sights is 1,500 SEK", "Within budget: 63.80 EUR left of 136.00 EUR."]
    assert unsupported_amounts(draft, sources) == ["703"]  # 136 slips through: the tool echoed it ...


def test_free_sights_can_go_into_the_budget(research_tools):
    # From a live run: free sights passed at price 0 were flagged, because the data said
    # "free entry" in words. The attractions data now spells out the 0.
    found = next(t for t in research_tools if t.name == "find_attractions").invoke({"city": "Lisbon", "max_price": 0})
    assert "LX Factory (market, Alcântara) | free entry (0 EUR)" in found
    args = {"items": [{"name": "LX Factory", "price_per_person": 0}], "budget": 1500}
    assert unsupported_budget_inputs(args, ["Our budget for sights is 1,500 SEK", found]) == []


def test_flags_budget_inputs_that_the_user_never_gave():
    # ... which is why the tool's inputs are checked too.
    args = {
        "items": [{"name": "Tile Museum", "price_per_person": 8}, {"name": "Tram 28", "price_per_person": 3.1}],
        "travellers": 2,
        "currency_code": "EUR",
        "budget": 136,
        "budget_currency": "EUR",
    }
    sources = ["Our budget for sights is 1,500 SEK", "National Tile Museum | 8 EUR per adult", "Tram 28 | 3.1 EUR per adult"]
    assert unsupported_budget_inputs(args, sources) == ["budget 136"]
    assert unsupported_budget_inputs({**args, "budget": 1500, "budget_currency": "SEK"}, sources) == []
