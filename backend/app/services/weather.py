"""Live daily forecasts from Open-Meteo (free, no API key).

Two calls: geocode the city name to coordinates, then fetch the daily forecast.
Open-Meteo forecasts cover today plus the next 15 days; requests outside that window
are rejected with a message that points the agent to the climate notes in the guides.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta

import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_HORIZON_DAYS = 16  # today + 15 days

# WMO weather interpretation codes used by Open-Meteo.
WMO_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "light freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "violent rain showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}


class WeatherError(Exception):
    """A problem the agent should be told about (bad dates, unknown city, API failure)."""


@dataclass(frozen=True)
class DailyForecast:
    day: date
    conditions: str
    temp_min_c: float | None
    temp_max_c: float | None
    precipitation_probability: int | None
    precipitation_mm: float | None

    def describe(self) -> str:
        line = f"{self.day.isoformat()} ({self.day.strftime('%a')}): {self.conditions}"
        if self.temp_min_c is not None and self.temp_max_c is not None:
            line += f", {self.temp_min_c:.0f}–{self.temp_max_c:.0f} °C"
        if self.precipitation_probability is not None:
            line += f", {self.precipitation_probability}% chance of precipitation"
        if self.precipitation_mm:
            line += f" ({self.precipitation_mm:.1f} mm)"
        return line


@dataclass(frozen=True)
class Forecast:
    place: str
    days: list[DailyForecast]
    note: str | None = None


class WeatherService:
    def __init__(self, client: httpx.Client, today: Callable[[], date] = date.today):
        self._client = client
        self._today = today

    def forecast(self, city: str, start: date, end: date) -> Forecast:
        start, end, note = self._clamp_to_horizon(start, end)
        place = self._geocode(city)
        try:
            response = self._client.get(
                FORECAST_URL,
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max,precipitation_sum",
                    "timezone": "auto",
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise WeatherError(f"The weather service is unavailable ({exc}).") from exc
        daily = response.json()["daily"]
        days = [
            DailyForecast(
                day=date.fromisoformat(day),
                conditions=WMO_CODES.get(daily["weather_code"][i], "unknown conditions"),
                temp_min_c=daily["temperature_2m_min"][i],
                temp_max_c=daily["temperature_2m_max"][i],
                precipitation_probability=daily["precipitation_probability_max"][i],
                precipitation_mm=daily["precipitation_sum"][i],
            )
            for i, day in enumerate(daily["time"])
        ]
        return Forecast(place=f"{place['name']}, {place.get('country', '')}".strip(", "), days=days, note=note)

    def _clamp_to_horizon(self, start: date, end: date) -> tuple[date, date, str | None]:
        today = self._today()
        last = today + timedelta(days=FORECAST_HORIZON_DAYS - 1)
        if end < start:
            raise WeatherError("end_date is before start_date.")
        if end < today:
            raise WeatherError("Those dates are in the past; only forecasts are available.")
        if start > last:
            raise WeatherError(
                f"Forecasts only reach {last.isoformat()} ({FORECAST_HORIZON_DAYS} days ahead). "
                "For later dates, use the seasonal climate notes in the travel guides."
            )
        notes = []
        if start < today:
            start = today
            notes.append(f"Past days were skipped; the forecast starts today ({today.isoformat()}).")
        if end > last:
            end = last
            notes.append(f"Days after {last.isoformat()} are beyond the forecast horizon.")
        return start, end, " ".join(notes) or None

    def _geocode(self, city: str) -> dict:
        # The geocoder matches place names only, so "Lisbon, Portugal" must be sent as "Lisbon".
        name = city.split(",")[0].strip()
        try:
            response = self._client.get(
                GEOCODING_URL, params={"name": name, "count": 1, "language": "en", "format": "json"}
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise WeatherError(f"The geocoding service is unavailable ({exc}).") from exc
        results = response.json().get("results") or []
        if not results:
            raise WeatherError(f"Could not find a place called '{city}'.")
        return results[0]
