"""A small RAG knowledge base over the Markdown travel guides in data/guides/.

Pipeline: parse front matter -> one chunk per "## " section -> embed -> in-memory
vector index -> cosine-similarity search with an optional city filter.

Section-level chunks suit these documents: each section answers one kind of question
("Getting around", "Typical costs"), so a hit is a self-contained passage the agent can
cite. Every chunk starts with "<City> — <Section>" so the embedding carries that context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_TITLE = re.compile(r"^# (.+)$", re.MULTILINE)
_SECTION = re.compile(r"^## (.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Passage:
    source: str
    section: str
    text: str
    score: float

    def describe(self) -> str:
        return f"(source: {self.source} › {self.section}, relevance {self.score:.2f})\n{self.text}"


def chunk_markdown(text: str, source: str) -> list[Document]:
    """Split one guide into section-level documents with metadata."""
    metadata: dict[str, str] = {}
    front_matter = _FRONT_MATTER.match(text)
    if front_matter:
        for line in front_matter.group(1).splitlines():
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip()
        text = text[front_matter.end():]

    title_match = _TITLE.search(text)
    title = title_match.group(1).strip() if title_match else Path(source).stem
    headings = list(_SECTION.finditer(text))
    documents = []
    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[heading.end():end].strip()
        section = heading.group(1).strip()
        documents.append(
            Document(
                page_content=f"{title} — {section}\n{body}",
                metadata={"source": source, "section": section, "city": metadata.get("city", "")},
            )
        )
    return documents


def load_guides(directory: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(directory.glob("*.md")):
        documents.extend(chunk_markdown(path.read_text(encoding="utf-8"), source=path.name))
    return documents


class KnowledgeBase:
    def __init__(self, documents: list[Document], embeddings: Embeddings):
        self.documents = documents
        self._store = InMemoryVectorStore(embeddings)
        self._store.add_documents(documents)  # one batched embeddings call at startup

    @classmethod
    def from_directory(cls, directory: Path, embeddings: Embeddings) -> KnowledgeBase:
        return cls(load_guides(directory), embeddings)

    @property
    def cities(self) -> list[str]:
        return sorted({doc.metadata["city"] for doc in self.documents if doc.metadata["city"]})

    def has_city(self, city: str) -> bool:
        return city.strip().casefold() in {c.casefold() for c in self.cities}

    def search(self, query: str, city: str | None = None, k: int = 4) -> list[Passage]:
        city_filter = None
        if city:
            wanted = city.strip().casefold()

            def city_filter(doc: Document) -> bool:
                return doc.metadata["city"].casefold() == wanted

        hits = self._store.similarity_search_with_score(query, k=k, filter=city_filter)
        return [
            Passage(
                source=doc.metadata["source"],
                section=doc.metadata["section"],
                text=doc.page_content,
                score=score,
            )
            for doc, score in hits
        ]
