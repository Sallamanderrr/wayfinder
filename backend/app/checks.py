"""Deterministic checks that back up the LLM verifier.

LLMs are poor at checking arithmetic, but number provenance is easy to check in code:

- `unsupported_amounts`: money amounts in a draft (two-decimal numbers like 1,211.90, and
  numbers next to a currency like "703 SEK") that match no number in the sources, i.e.
  figures the model computed or converted by hand.
- `unsupported_budget_inputs`: values the planner passed to `budget_breakdown` (the budget
  and the prices) that come from neither the user nor the research. Without this, a budget
  converted by hand would be echoed back by the tool and look backed up.

Amounts are compared at the precision they are written: "179.21" must match a source to the
cent (179.2096 does), while a whole number like "~78 SEK" only has to match to the whole unit
(77.85 does). So honest rounding passes, but "703 SEK" for a true 691.32 is still caught.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

# Any number: 8, 3.10, 179.2096, 1,211.90.
_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_TWO_DECIMALS = re.compile(r"[\d,]+\.\d{2}")
_CURRENCY_AFTER = re.compile(r"\s?(?:EUR|SEK|USD|GBP|NOK|DKK|CHF|kr\b|€|\$|£)")
_CURRENCY_BEFORE = re.compile(r"(?:EUR|SEK|USD|GBP|NOK|DKK|CHF|€|\$|£)\s?$")


def unsupported_amounts(draft: str, sources: Iterable[str]) -> list[str]:
    """Money amounts in `draft` that match no number in `sources`, at the amount's own precision."""
    known = _known_cents(sources)
    known_whole = {(cents + 50) // 100 for cents in known}  # each source rounded to a whole unit
    missing: list[str] = []
    for amount in money_amounts(draft):
        cents = _cents(amount)
        whole_number = "." not in amount
        supported = (cents // 100 in known_whole) if whole_number else (cents in known)
        if not supported and amount not in missing:
            missing.append(amount)
    return missing


def unsupported_budget_inputs(args: dict[str, Any], sources: Iterable[str]) -> list[str]:
    """Descriptions of `budget_breakdown` inputs that appear in none of the sources."""
    known = _known_cents(sources)
    values = [("budget", args.get("budget"))]
    for item in args.get("items") or []:
        if isinstance(item, dict):
            values.append((f"price of {item.get('name', 'an item')}", item.get("price_per_person")))
    problems = []
    for label, value in values:
        if value is None:
            continue
        cents = _cents(str(value))
        if cents is not None and cents not in known:
            problems.append(f"{label} {value}")
    return problems


def money_amounts(text: str) -> list[str]:
    """Numbers that look like money: exactly two decimals, or written next to a currency."""
    amounts = []
    for match in _NUMBER.finditer(text):
        start, end = match.span()
        if start > 0 and (text[start - 1].isdigit() or text[start - 1] in ".,"):
            continue  # inside a longer token
        number = match.group()
        if (
            _TWO_DECIMALS.fullmatch(number)
            or _CURRENCY_AFTER.match(text, end)
            or _CURRENCY_BEFORE.search(text[max(0, start - 5) : start])
        ):
            amounts.append(number)
    return amounts


def _known_cents(sources: Iterable[str]) -> set[int]:
    known = set()
    for text in sources:
        for number in _NUMBER.findall(text):
            cents = _cents(number)
            if cents is not None:
                known.add(cents)
    return known


def _cents(number: str) -> int | None:
    try:
        value = Decimal(number.replace(",", ""))
    except InvalidOperation:
        return None
    return int((value * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))
