"""Structured attraction data (CSV) with typed filtering.

The LLM never writes queries itself: it calls a tool with typed filter arguments, and
this module does the filtering. That keeps the data access predictable and safe.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


@dataclass(frozen=True)
class Attraction:
    city: str
    name: str
    category: str
    neighbourhood: str
    price: float  # adult ticket, in `currency`
    currency: str
    duration_hours: float
    indoor: bool
    closed_on: str  # weekday name, or "" if open daily
    book_ahead: bool
    notes: str

    def describe(self) -> str:
        # "0 EUR" spelled out, so a free sight's price is a number the budget checks can find.
        price = f"free entry (0 {self.currency})" if self.price == 0 else f"{self.price:g} {self.currency} per adult"
        parts = [
            f"{self.name} ({self.category}, {self.neighbourhood})",
            price,
            f"~{self.duration_hours:g} h",
            "indoor" if self.indoor else "outdoor",
            f"closed on {self.closed_on}s" if self.closed_on else "open daily",
            "book ahead" if self.book_ahead else "",
            self.notes,
        ]
        return " | ".join(part for part in parts if part)


class AttractionsDB:
    def __init__(self, attractions: list[Attraction]):
        self._rows = attractions

    @classmethod
    def from_csv(cls, path: Path) -> AttractionsDB:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = [
                Attraction(
                    city=row["city"],
                    name=row["name"],
                    category=row["category"],
                    neighbourhood=row["neighbourhood"],
                    price=float(row["price"]),
                    currency=row["currency"],
                    duration_hours=float(row["duration_hours"]),
                    indoor=row["indoor"].strip().lower() == "yes",
                    closed_on=row["closed_on"].strip(),
                    book_ahead=row["book_ahead"].strip().lower() == "yes",
                    notes=row["notes"].strip(),
                )
                for row in csv.DictReader(handle)
            ]
        return cls(rows)

    @property
    def cities(self) -> list[str]:
        return sorted({row.city for row in self._rows})

    @property
    def categories(self) -> list[str]:
        return sorted({row.category for row in self._rows})

    def find(
        self,
        city: str,
        *,
        category: str | None = None,
        indoor: bool | None = None,
        max_price: float | None = None,
        open_on: str | None = None,
    ) -> list[Attraction]:
        """Return matching attractions, cheapest first.

        `open_on` accepts a weekday name ("Monday") or an ISO date ("2026-10-19") and
        excludes attractions closed on that day.
        """
        closed_day = _weekday(open_on) if open_on else None
        matches = [
            row
            for row in self._rows
            if row.city.casefold() == city.strip().casefold()
            and (category is None or row.category.casefold() == category.strip().casefold())
            and (indoor is None or row.indoor == indoor)
            and (max_price is None or row.price <= max_price)
            and (closed_day is None or row.closed_on.casefold() != closed_day)
        ]
        return sorted(matches, key=lambda row: (row.price, row.name))


def _weekday(value: str) -> str:
    text = value.strip().casefold()
    if text in WEEKDAYS:
        return text
    try:
        return WEEKDAYS[date.fromisoformat(text).weekday()]
    except ValueError as exc:
        raise ValueError(f"'{value}' is neither a weekday name nor a YYYY-MM-DD date.") from exc
