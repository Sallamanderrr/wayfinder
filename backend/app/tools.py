"""LangChain tool adapters around the domain services.

Each agent gets only the tools that match its role:
- researcher: search_travel_guides (RAG), find_attractions (structured data), get_weather_forecast (API)
- planner:    budget_breakdown (computation + API), calculator, convert_currency (API)

Tools always return text. Expected failures (bad dates, unknown city, API down) are
returned as "Error: ..." strings so the agent can recover or explain; `_never_raises`
does the same for unexpected exceptions, which would otherwise end the whole turn.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from datetime import date

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from .rag import KnowledgeBase
from .services.attractions import AttractionsDB
from .services.budget import BudgetLine, budget_table
from .services.calculator import CalculatorError, evaluate, format_number
from .services.currency import CurrencyError, CurrencyService
from .services.weather import WeatherError, WeatherService

logger = logging.getLogger(__name__)


class BudgetItem(BaseModel):
    """One paid item in a trip budget."""

    name: str = Field(description="What the cost is for, e.g. 'Jerónimos Monastery'.")
    price_per_person: float = Field(description="Price for one person, in the prices' currency.")


def _never_raises(func: Callable[..., str]) -> Callable[..., str]:
    """Report unexpected exceptions to the agent as text instead of ending the turn."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> str:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            logger.exception("Tool %s failed", func.__name__)
            return f"Error: {func.__name__} failed unexpectedly ({type(exc).__name__}). Try another approach."

    return wrapper


def build_research_tools(
    kb: KnowledgeBase, attractions: AttractionsDB, weather: WeatherService
) -> list[BaseTool]:
    @tool(parse_docstring=True)
    @_never_raises
    def search_travel_guides(query: str, city: str | None = None) -> str:
        """Semantic search over the curated travel guides (the RAG knowledge base).

        The guides cover Lisbon, Barcelona, Rome, Paris and Stockholm (climate by season,
        neighbourhoods, getting around, typical costs, food, practical tips, day trips),
        plus general trip-planning advice (daily budgets, tipping, planning around weather).

        Args:
            query: What to look for, as a question or keywords.
            city: Optional city name; restricts the search to that city's guide.
        """
        if city and not kb.has_city(city):
            return f"No guide covers '{city}'. Guides are available for: {', '.join(kb.cities)}."
        passages = kb.search(query, city=city)
        if not passages:
            return "No matching passages found."
        return "\n\n".join(f"[{i}] {p.describe()}" for i, p in enumerate(passages, start=1))

    @tool(parse_docstring=True)
    @_never_raises
    def find_attractions(
        city: str,
        category: str | None = None,
        indoor: bool | None = None,
        max_price: float | None = None,
        open_on: str | None = None,
    ) -> str:
        """Query the attractions database (structured data) for one city, cheapest first.

        Each result has the adult ticket price in local currency, category, typical visit
        length, indoor/outdoor, closing day, whether to book ahead, and notes.

        Args:
            city: One of Lisbon, Barcelona, Rome, Paris, Stockholm.
            category: Optional filter: landmark, museum, church, park, viewpoint, food, market, neighbourhood, experience, beach or day trip.
            indoor: True for indoor only (good on rainy days), False for outdoor only, omit for both.
            max_price: Optional maximum adult price, in the attraction's local currency.
            open_on: Optional weekday name or YYYY-MM-DD date; excludes places closed that day.
        """
        if city.strip().casefold() not in {c.casefold() for c in attractions.cities}:
            return f"No attraction data for '{city}'. Available cities: {', '.join(attractions.cities)}."
        try:
            results = attractions.find(
                city, category=category, indoor=indoor, max_price=max_price, open_on=open_on
            )
        except ValueError as exc:
            return f"Error: {exc}"
        if not results:
            return (
                "No attractions match those filters. "
                f"Valid categories: {', '.join(attractions.categories)}."
            )
        lines = [f"- {row.describe()}" for row in results]
        return f"{len(results)} attraction(s) in {results[0].city} (attractions database):\n" + "\n".join(lines)

    @tool(parse_docstring=True)
    @_never_raises
    def get_weather_forecast(city: str, start_date: str, end_date: str) -> str:
        """Live daily weather forecast (Open-Meteo) for any city in the world.

        Only covers today and the next 15 days. For dates further ahead, use the seasonal
        climate notes in the travel guides instead.

        Args:
            city: City name, e.g. "Lisbon".
            start_date: First day, YYYY-MM-DD.
            end_date: Last day, YYYY-MM-DD (inclusive).
        """
        try:
            start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
        except ValueError:
            return "Error: start_date and end_date must be dates in YYYY-MM-DD format."
        try:
            forecast = weather.forecast(city, start, end)
        except WeatherError as exc:
            return f"Error: {exc}"
        lines = [f"Daily forecast for {forecast.place} (Open-Meteo):"]
        lines += [f"- {day.describe()}" for day in forecast.days]
        if forecast.note:
            lines.append(f"Note: {forecast.note}")
        return "\n".join(lines)

    return [search_travel_guides, find_attractions, get_weather_forecast]


def build_planning_tools(currency: CurrencyService) -> list[BaseTool]:
    @tool(parse_docstring=True)
    @_never_raises
    def budget_breakdown(
        items: list[BudgetItem],
        travellers: int,
        currency_code: str,
        budget: float | None = None,
        budget_currency: str | None = None,
    ) -> str:
        """Compute a trip budget as a ready-made table. Use it for every budget.

        Multiplies each per-person price by the number of travellers, converts to the
        budget's currency at today's rate, totals everything and checks the budget, all
        with exact decimal arithmetic. Copy the table into your answer unchanged.

        Args:
            items: The paid items, each with a name and a price per person.
            travellers: Number of travellers.
            currency_code: ISO code of the prices, e.g. "EUR".
            budget: The user's budget exactly as they stated it, in their own currency (e.g. 1500). Never convert it yourself.
            budget_currency: ISO code of the user's budget, e.g. "SEK". The tool converts the prices into it. Defaults to currency_code.
        """
        lines = [
            BudgetLine(name=item.name, price_per_person=item.price_per_person)
            if isinstance(item, BudgetItem)
            else BudgetLine(name=item["name"], price_per_person=item["price_per_person"])
            for item in items
        ]
        target = (budget_currency or currency_code).strip().upper()
        source = currency_code.strip().upper()
        try:
            rate = currency.rate(source, target)
            return budget_table(
                lines,
                travellers,
                source,
                target_currency=target,
                rate=rate.rate,
                rate_date=rate.as_of,
                budget=budget,
            )
        except (CurrencyError, ValueError) as exc:
            return f"Error: {exc}"

    @tool(parse_docstring=True)
    @_never_raises
    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression exactly. Use it for every total, split and budget check.

        Supports + - * / // % **, parentheses, and round(), min(), max(), abs().
        Example: "2 * (18 + 10) + 2 * 3.10".

        Args:
            expression: The arithmetic expression to evaluate.
        """
        try:
            return f"{expression} = {format_number(evaluate(expression))}"
        except CalculatorError as exc:
            return f"Error: {exc}"

    @tool(parse_docstring=True)
    @_never_raises
    def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
        """Convert money between currencies using today's central-bank reference rates (Frankfurter).

        Args:
            amount: The amount to convert.
            from_currency: Three-letter ISO code of the amount, e.g. "EUR".
            to_currency: Three-letter ISO code to convert into, e.g. "SEK".
        """
        try:
            return currency.convert(amount, from_currency, to_currency).describe()
        except CurrencyError as exc:
            return f"Error: {exc}"

    return [budget_breakdown, calculator, convert_currency]
