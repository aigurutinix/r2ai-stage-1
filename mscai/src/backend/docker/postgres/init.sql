CREATE TABLE IF NOT EXISTS legal_knowledge_records (
    id BIGINT PRIMARY KEY,
    law_id TEXT NOT NULL,
    law_name TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    chapter TEXT,
    article TEXT NOT NULL,
    article_title TEXT NOT NULL,
    content TEXT NOT NULL,
    author TEXT NOT NULL,
    extra JSONB NOT NULL DEFAULT '[]'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_legal_knowledge_reference
    ON legal_knowledge_records (law_id, article);
