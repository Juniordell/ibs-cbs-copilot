"""Dump only chunks matching golden-set article numbers."""

import sys, os, json, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg
from dotenv import load_dotenv

load_dotenv()

GOLDEN = Path("evals/golden/golden_v1.jsonl")
OUT = Path("evals/fixtures/chunks_seed.sql")

# Collect ONLY article numbers (e.g. "Art. 15") — ignore source labels
ARTICLE_RE = re.compile(r"Art\.?\s*\d+", re.IGNORECASE)
wanted_articles = set()
for line in GOLDEN.read_text().splitlines():
    if not line.strip():
        continue
    item = json.loads(line)
    for src in item["expected_sources"]:
        m = ARTICLE_RE.search(src)
        if m:
            wanted_articles.add(m.group(0))

print(f"Wanted articles: {len(wanted_articles)}")

with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    with conn.cursor() as cur:
        # DISTINCT ON: one row per (article, source), preferring the caput chunk
        cur.execute("""
            SELECT DISTINCT ON (article, source)
                text, article, paragraph, item, source, embedding, metadata
            FROM chunks
            WHERE article = ANY(%s)
            ORDER BY article, source, paragraph NULLS FIRST
        """, (list(wanted_articles),))
        rows = cur.fetchall()

print(f"Got {len(rows)} rows")

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w") as f:
    for text, article, paragraph, item, source, embedding, metadata in rows:
        emb_str = "[" + ",".join(str(x) for x in embedding) + "]"
        text_safe = text.replace("'", "''")
        article_safe = article.replace("'", "''")
        source_safe = source.replace("'", "''")
        p_val = f"'{paragraph.replace(chr(39), chr(39)+chr(39))}'" if paragraph else "NULL"
        i_val = f"'{item.replace(chr(39), chr(39)+chr(39))}'" if item else "NULL"
        meta_json = json.dumps(metadata).replace("'", "''")
        f.write(
            f"INSERT INTO chunks (text, article, paragraph, item, source, embedding, metadata) "
            f"VALUES ('{text_safe}', '{article_safe}', {p_val}, {i_val}, '{source_safe}', "
            f"'{emb_str}'::vector, '{meta_json}'::jsonb) ON CONFLICT DO NOTHING;\n"
        )

print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB)")