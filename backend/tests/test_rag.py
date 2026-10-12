from __future__ import annotations

from conftest import DATA_DIR

from app.rag import chunk_markdown, load_guides

GUIDE = """---
city: Testville
currency: EUR
---
# Testville

## Getting around
Take the tram.

## Typical costs
Coffee costs €2.
"""


def test_chunking_creates_one_document_per_section_with_metadata():
    docs = chunk_markdown(GUIDE, source="testville.md")
    assert [d.metadata["section"] for d in docs] == ["Getting around", "Typical costs"]
    assert docs[0].metadata == {"source": "testville.md", "section": "Getting around", "city": "Testville"}
    assert docs[0].page_content == "Testville — Getting around\nTake the tram."


def test_every_guide_has_sections():
    docs = load_guides(DATA_DIR / "guides")
    sources = {d.metadata["source"] for d in docs}
    assert len(sources) == 6
    assert all(d.page_content.strip() for d in docs)


def test_city_filter_restricts_results(knowledge_base):
    # The test embedder is lexical, so the query uses words from the target passage.
    hits = knowledge_base.search("Fiumicino airport Leonardo Express", city="rome")
    assert hits and all(hit.source == "rome.md" for hit in hits)
    assert hits[0].section == "Getting around"


def test_unfiltered_search_finds_relevant_section(knowledge_base):
    top = knowledge_base.search("tipping customs in Portugal Spain Italy France Sweden", k=1)[0]
    assert (top.source, top.section) == ("trip-planning-basics.md", "Tipping at a glance")


def test_search_tool_reports_missing_cities(research_tools):
    search = next(t for t in research_tools if t.name == "search_travel_guides")
    assert search.invoke({"query": "museums", "city": "Oslo"}).startswith("No guide covers 'Oslo'")
    assert "(source: stockholm.md ›" in search.invoke({"query": "fika coffee", "city": "Stockholm"})
