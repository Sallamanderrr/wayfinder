"""Live exchange rates from Frankfurter (central-bank reference rates, no API key).

Rates change once per working day, so they are cached in memory for an hour: a plan
that converts ten prices makes one HTTP call, not ten.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date

import httpx

RATE_URL = "https://api.frankfurter.dev/v2/rate/{base}/{quote}"
_CURRENCY_CODE = re.compile(r"^[A-Z]{3}$")


class CurrencyError(Exception):
    """A problem the agent should be told about (unknown code, API failure)."""


@dataclass(frozen=True)
class Rate:
    base: str
    quote: str
    rate: float
    as_of: str  # date the reference rate was published


@dataclass(frozen=True)
class Conversion:
    amount: float
    converted: float
    rate: Rate

    def describe(self) -> str:
        return (
            f"{self.amount:,.2f} {self.rate.base} = {self.converted:,.2f} {self.rate.quote} "
            f"(rate 1 {self.rate.base} = {self.rate.rate:.4f} {self.rate.quote}, reference rate of {self.rate.as_of})"
        )


class CurrencyService:
    def __init__(self, client: httpx.Client, cache_ttl_seconds: float = 3600):
        self._client = client
        self._ttl = cache_ttl_seconds
        self._cache: dict[tuple[str, str], tuple[float, Rate]] = {}  # (base, quote) -> (fetched at, rate)

    def convert(self, amount: float, from_currency: str, to_currency: str) -> Conversion:
        rate = self.rate(from_currency, to_currency)
        return Conversion(amount=amount, converted=round(amount * rate.rate, 2), rate=rate)

    def rate(self, base: str, quote: str) -> Rate:
        base, quote = _normalise(base), _normalise(quote)
        if base == quote:
            return Rate(base, quote, 1.0, date.today().isoformat())
        cached = self._cache.get((base, quote))
        if cached and time.monotonic() - cached[0] < self._ttl:
            return cached[1]
        try:
            response = self._client.get(RATE_URL.format(base=base.lower(), quote=quote.lower()))
        except httpx.HTTPError as exc:
            raise CurrencyError(f"The exchange-rate service is unavailable ({exc}).") from exc
        if response.status_code in (404, 422):
            raise CurrencyError(f"Unsupported currency pair {base}/{quote}.")
        try:
            response.raise_for_status()
            payload = response.json()
            rate = Rate(base, quote, float(payload["rate"]), str(payload.get("date", "")))
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise CurrencyError(f"Unexpected response from the exchange-rate service ({exc}).") from exc
        self._cache[(base, quote)] = (time.monotonic(), rate)
        return rate


def _normalise(code: str) -> str:
    code = code.strip().upper()
    if not _CURRENCY_CODE.match(code):
        raise CurrencyError(f"'{code}' is not a three-letter ISO currency code (e.g. EUR, SEK, USD).")
    return code
