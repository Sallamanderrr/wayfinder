from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pytest
from fakes import HashingEmbeddings

from app.rag import KnowledgeBase
from app.services.attractions import AttractionsDB
from app.services.currency import CurrencyService
from app.services.weather import WeatherService
from app.tools import build_planning_tools, build_research_tools

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TODAY = date(2026, 10, 12)  # a Monday


def fake_api(request: httpx.Request) -> httpx.Response:
    """Offline stand-ins for Open-Meteo and Frankfurter."""
    host, path = request.url.host, request.url.path
    if host == "geocoding-api.open-meteo.com":
        name = request.url.params["name"]
        if name.lower() == "atlantis":
            return httpx.Response(200, json={"generationtime_ms": 0.1})
        return httpx.Response(
            200,
            json={"results": [{"name": name.title(), "country": "Portugal", "latitude": 38.72, "longitude": -9.14}]},
        )
    if host == "api.open-meteo.com":
        return httpx.Response(
            200,
            json={
                "daily": {
                    "time": ["2026-10-17", "2026-10-18"],
                    "weather_code": [63, 2],
                    "temperature_2m_max": [19.4, 21.0],
                    "temperature_2m_min": [14.1, 13.2],
                    "precipitation_probability_max": [85, 10],
                    "precipitation_sum": [6.2, 0.0],
                }
            },
        )
    if host == "api.frankfurter.dev" and path.startswith("/v2/rate/"):
        base, quote = path.removeprefix("/v2/rate/").split("/")
        rates = {("sek", "eur"): 0.0893, ("eur", "sek"): 11.1948}
        if (base, quote) not in rates:
            return httpx.Response(422, json={"status": 422, "message": f"invalid currency: {quote.upper()}"})
        return httpx.Response(
            200, json={"date": "2026-10-12", "base": base.upper(), "quote": quote.upper(), "rate": rates[(base, quote)]}
        )
    return httpx.Response(404)


@pytest.fixture
def http_client() -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(fake_api))


@pytest.fixture(scope="session")
def knowledge_base() -> KnowledgeBase:
    return KnowledgeBase.from_directory(DATA_DIR / "guides", HashingEmbeddings())


@pytest.fixture(scope="session")
def attractions() -> AttractionsDB:
    return AttractionsDB.from_csv(DATA_DIR / "attractions.csv")


@pytest.fixture
def weather(http_client) -> WeatherService:
    return WeatherService(http_client, today=lambda: TODAY)


@pytest.fixture
def currency(http_client) -> CurrencyService:
    return CurrencyService(http_client)


@pytest.fixture
def research_tools(knowledge_base, attractions, weather):
    return build_research_tools(knowledge_base, attractions, weather)


@pytest.fixture
def planning_tools(currency):
    return build_planning_tools(currency)
