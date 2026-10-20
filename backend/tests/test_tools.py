from __future__ import annotations

from datetime import date

import pytest
from conftest import TODAY

from app.services.currency import CurrencyError
from app.services.weather import WeatherError


def by_name(tools):
    return {tool.name: tool for tool in tools}


# --- attractions (structured data) -------------------------------------------------------


def test_find_attractions_filters_and_sorts(attractions):
    results = attractions.find("lisbon", indoor=True, max_price=15, open_on="Monday")
    names = [row.name for row in results]
    assert "National Tile Museum" not in names  # closed on Mondays
    assert "Calouste Gulbenkian Museum" in names
    assert all(row.indoor and row.price <= 15 for row in results)
    assert [row.price for row in results] == sorted(row.price for row in results)


def test_find_attractions_accepts_dates_for_closing_days(attractions):
    # 2026-10-20 is a Tuesday: the Louvre is closed, the Musée d'Orsay is open.
    names = [row.name for row in attractions.find("Paris", category="museum", open_on="2026-10-20")]
    assert "Louvre Museum" not in names and "Musée d'Orsay" in names


def test_find_attractions_tool_handles_unknown_input(research_tools):
    find = by_name(research_tools)["find_attractions"]
    assert "No attraction data for 'Berlin'" in find.invoke({"city": "Berlin"})
    assert find.invoke({"city": "Rome", "open_on": "someday"}).startswith("Error:")
    assert "Valid categories" in find.invoke({"city": "Rome", "category": "casino"})


# --- weather (public API) ----------------------------------------------------------------


def test_weather_tool_formats_daily_forecast(research_tools):
    forecast = by_name(research_tools)["get_weather_forecast"].invoke(
        {"city": "Lisbon", "start_date": "2026-10-17", "end_date": "2026-10-18"}
    )
    assert "Lisbon, Portugal" in forecast
    assert "2026-10-17 (Sat): rain, 14–19 °C, 85% chance of precipitation (6.2 mm)" in forecast


def test_weather_geocodes_the_city_part_of_a_place_name(weather, http_client):
    requested = []
    original_send = http_client._transport.handle_request

    def recording_send(request):
        requested.append(request.url.params.get("name"))
        return original_send(request)

    http_client._transport.handle_request = recording_send
    weather.forecast("Lisbon, Portugal", date(2026, 10, 17), date(2026, 10, 18))
    assert requested[0] == "Lisbon"


def test_weather_rejects_dates_beyond_the_forecast_horizon(weather):
    with pytest.raises(WeatherError, match="travel guides"):
        weather.forecast("Lisbon", date(2026, 11, 20), date(2026, 11, 22))


def test_weather_clamps_ranges_that_run_past_the_horizon(weather):
    start, end, note = weather._clamp_to_horizon(TODAY, date(2026, 11, 30))
    assert (start, end) == (TODAY, date(2026, 10, 27))
    assert "beyond the forecast horizon" in note


def test_weather_tool_explains_unknown_places_and_bad_dates(research_tools):
    tool = by_name(research_tools)["get_weather_forecast"]
    assert "Could not find a place called 'Atlantis'" in tool.invoke(
        {"city": "Atlantis", "start_date": "2026-10-13", "end_date": "2026-10-14"}
    )
    assert "YYYY-MM-DD" in tool.invoke({"city": "Lisbon", "start_date": "next friday", "end_date": "2026-10-14"})


# --- currency (public API) ---------------------------------------------------------------


def test_currency_conversion_uses_live_rate_and_caches_it(currency, http_client):
    calls = []
    original_send = http_client._transport.handle_request

    def counting_send(request):
        calls.append(request.url.path)
        return original_send(request)

    http_client._transport.handle_request = counting_send
    first = currency.convert(250, "eur", "SEK")
    currency.convert(100, "EUR", "SEK")
    assert first.converted == pytest.approx(2798.70)
    assert calls == ["/v2/rate/eur/sek"]  # second conversion served from cache
    assert "250.00 EUR = 2,798.70 SEK (rate 1 EUR = 11.1948 SEK" in first.describe()


def test_unexpected_tool_failures_are_reported_to_the_agent(knowledge_base, attractions):
    from app.tools import build_research_tools

    class BrokenWeather:
        def forecast(self, *args):
            raise KeyError("daily")  # e.g. the API changed its response format

    tool = by_name(build_research_tools(knowledge_base, attractions, BrokenWeather()))["get_weather_forecast"]
    result = tool.invoke({"city": "Lisbon", "start_date": "2026-10-17", "end_date": "2026-10-18"})
    assert result.startswith("Error: get_weather_forecast failed unexpectedly (KeyError)")


def test_tool_schemas_survive_the_error_wrapper(research_tools):
    schema = by_name(research_tools)["find_attractions"].tool_call_schema.model_json_schema()
    assert set(schema["properties"]) == {"city", "category", "indoor", "max_price", "open_on"}
    assert schema["required"] == ["city"]
    assert "weekday name" in schema["properties"]["open_on"]["description"]


def test_currency_errors(currency):
    with pytest.raises(CurrencyError, match="three-letter"):
        currency.convert(10, "euro", "SEK")
    with pytest.raises(CurrencyError, match="Unsupported"):
        currency.convert(10, "EUR", "XYZ")
    assert currency.convert(10, "SEK", "SEK").converted == 10
