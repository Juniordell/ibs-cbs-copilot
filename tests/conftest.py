from __future__ import annotations

import os

import psycopg
import pytest


@pytest.fixture(scope="session")
def db_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql://copilot:copilot_test@localhost:5432/copilot_test",
    )


@pytest.fixture(autouse=True)
def reset_chunks(db_url):
    """Wipe chunks between tests. Skips if table doesn't exist (unit tests)."""
    try:
        with psycopg.connect(db_url, connect_timeout=2) as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE chunks RESTART IDENTITY;")
    except psycopg.OperationalError:
        pass  # DB not available for pure unit tests
    yield


@pytest.fixture
def seed_chunks(db_url):
    """Insert 3 fake chunks so integration tests have data to retrieve."""
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (text, article, paragraph, item, source, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        "Art. 15. A alíquota do IBS incidente sobre cada operação corresponderá à soma da alíquota do Estado e do Município de destino.",
                        "Art. 15", None, None, "LC 214/2025",
                        [0.1] * 1536, "{}",
                    ),
                    (
                        "Art. 19. São contribuintes da CBS: I - fornecedor que realize operações; II - o adquirente.",
                        "Art. 19", None, None, "Decreto 12.955/2026",
                        [0.2] * 1536, "{}",
                    ),
                    (
                        "Art. 12. A base de cálculo do IBS e da CBS é o valor da operação.",
                        "Art. 12", None, None, "LC 214/2025",
                        [0.3] * 1536, "{}",
                    ),
                ],
            )
        conn.commit()
    yield