# tests/test_chunker.py
from src.copilot.ingestion.chunker import chunk_legal_text, LegalChunk


SAMPLE = """
Art. 1º Ficam instituídos o IBS e a CBS.

Art. 2º O IBS e a CBS são informados pelo princípio da neutralidade.

Art. 3º Para fins desta Lei Complementar, consideram-se:
§ 1º Operação: qualquer fornecimento oneroso.
§ 2º Fornecedor: pessoa física ou jurídica.
Parágrafo único. Aplica-se o disposto no caput.
"""


def test_yields_one_chunk_per_short_article():
    chunks = list(chunk_legal_text(SAMPLE, source="LC 214/2025"))
    articles = [c.article for c in chunks]
    assert "Art. 1º" in articles
    assert "Art. 2º" in articles


def test_metadata_is_populated():
    chunks = list(chunk_legal_text(SAMPLE, source="LC 214/2025"))
    first = chunks[0]
    assert isinstance(first, LegalChunk)
    assert first.source == "LC 214/2025"
    assert first.article == "Art. 1º"
    assert "IBS" in first.text


def test_long_article_is_split_by_paragraph():
    # Force subchunking by making Art. 3 huge
    long_body = "Regra base. " * 300  # ~3600 chars
    text = f"Art. 10 {long_body}\n§ 1º Exceção A.\n§ 2º Exceção B."
    chunks = list(chunk_legal_text(text, source="LC 214/2025"))

    paragraphs = [c.paragraph for c in chunks if c.paragraph]
    assert any("§ 1" in p for p in paragraphs)
    assert any("§ 2" in p for p in paragraphs)


def test_empty_text_yields_nothing():
    assert list(chunk_legal_text("", source="X")) == []


def test_text_without_articles_yields_nothing():
    assert list(chunk_legal_text("Random prose.", source="X")) == []