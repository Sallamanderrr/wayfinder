"""Deterministic checks that back up the LLM verifier.

`unsupported_amounts` finds money amounts in a draft (numbers with exactly two decimals,
like 1,211.90) that don't appear in any tool result, i.e. figures the model computed by
hand. LLMs are poor at checking arithmetic, but string-level provenance is easy in code.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from decimal import ROUND_HALF_UP, Decimal

# An amount with exactly two decimals, optionally with thousands separators: 69.44, 1,211.90.
_AMOUNT = re.compile(r"(?<![\d.,])(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}(?![\d])")
# Any number in a source: 8, 3.10, 179.2096, 1,211.90.
_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def unsupported_amounts(draft: str, sources: Iterable[str]) -> list[str]:
    """Amounts in `draft` that match no number in `sources`, after rounding to cents."""
    known = {_cents(number) for text in sources for number in _NUMBER.findall(text)}
    missing: list[str] = []
    for amount in _AMOUNT.findall(draft):
        if _cents(amount) not in known and amount not in missing:
            missing.append(amount)
    return missing


def _cents(number: str) -> int:
    return int((Decimal(number.replace(",", "")) * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))
