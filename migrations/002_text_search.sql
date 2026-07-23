ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS text_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('portuguese', text)) STORED;

CREATE INDEX IF NOT EXISTS chunks_text_tsv_idx
    ON chunks USING GIN (text_tsv);