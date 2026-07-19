"""All sqlite access. Nothing outside this module opens a cursor."""

import sqlite3
from pathlib import Path

import numpy as np

VECTOR_DTYPE = np.float32

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id       TEXT PRIMARY KEY,
    arxiv_id       TEXT,
    title          TEXT,
    abstract       TEXT,
    published      TEXT,
    categories     TEXT,
    citation_count INTEGER,
    expanded       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_papers_arxiv ON papers(arxiv_id);

CREATE TABLE IF NOT EXISTS citations (
    citing_id TEXT NOT NULL,
    cited_id  TEXT NOT NULL,
    PRIMARY KEY (citing_id, cited_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_citations_cited ON citations(cited_id);

CREATE TABLE IF NOT EXISTS vectors (
    paper_id TEXT PRIMARY KEY,
    vec      BLOB NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    paper_id UNINDEXED,
    title,
    abstract
);
"""

PAPER_COLUMNS = (
    'paper_id', 'arxiv_id', 'title', 'abstract',
    'published', 'categories', 'citation_count',
)


def connect(db_path, check_same_thread, read_only):
    """read_only=True opens the file with sqlite's own mode=ro, so a stray write raises
    instead of silently succeeding. The server uses it; the scrape worker owns the only
    read-write connection. WAL lets those coexist without any application-level lock.
    """
    if read_only:
        conn = sqlite3.connect(
            f'file:{Path(db_path).as_posix()}?mode=ro',
            uri=True, check_same_thread=check_same_thread,
        )
    else:
        conn = sqlite3.connect(db_path, check_same_thread=check_same_thread)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')

    conn.row_factory = sqlite3.Row
    # Wait out a WAL checkpoint rather than raising "database is locked".
    conn.execute('PRAGMA busy_timeout=10000')
    return conn


def init_schema(conn):
    conn.executescript(SCHEMA)
    conn.commit()


def upsert_papers(conn, papers):
    """Insert or refresh metadata. Never touches `expanded` - that is owned by the scraper.

    COALESCE(excluded.x, papers.x) so a sparse re-fetch cannot null out data we already have.
    """
    # A paper can appear in both the refs and citations of one expansion (mutual citation);
    # dedup so the FTS resync below cannot insert it twice.
    papers = list({p['paper_id']: p for p in papers}.values())
    if not papers:
        return

    rows = [tuple(p[col] for col in PAPER_COLUMNS) for p in papers]
    conn.executemany(
        """
        INSERT INTO papers (paper_id, arxiv_id, title, abstract,
                            published, categories, citation_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(paper_id) DO UPDATE SET
            arxiv_id       = COALESCE(excluded.arxiv_id,       papers.arxiv_id),
            title          = COALESCE(excluded.title,          papers.title),
            abstract       = COALESCE(excluded.abstract,       papers.abstract),
            published      = COALESCE(excluded.published,      papers.published),
            categories     = COALESCE(excluded.categories,     papers.categories),
            citation_count = COALESCE(excluded.citation_count, papers.citation_count)
        """,
        rows,
    )

    # Standalone FTS table, so it needs explicit resync on every write.
    ids = [(p['paper_id'],) for p in papers]
    conn.executemany('DELETE FROM papers_fts WHERE paper_id = ?', ids)
    conn.executemany(
        'INSERT INTO papers_fts (paper_id, title, abstract) VALUES (?, ?, ?)',
        [(p['paper_id'], p['title'], p['abstract']) for p in papers],
    )
    conn.commit()


def insert_edges(conn, edges):
    """edges: iterable of (citing_id, cited_id)."""
    conn.executemany(
        'INSERT OR IGNORE INTO citations (citing_id, cited_id) VALUES (?, ?)',
        edges,
    )
    conn.commit()


def mark_expanded(conn, paper_id):
    conn.execute('UPDATE papers SET expanded = 1 WHERE paper_id = ?', (paper_id,))
    conn.commit()


def expanded_ids(conn):
    rows = conn.execute('SELECT paper_id FROM papers WHERE expanded = 1').fetchall()
    return {r['paper_id'] for r in rows}


def upsert_vectors(conn, pairs):
    """pairs: iterable of (paper_id, vector). One transaction for the whole batch -
    committing per row turns a 900-vector write into 900 fsyncs."""
    rows = [
        (paper_id, vec.astype(VECTOR_DTYPE).tobytes())
        for paper_id, vec in pairs
    ]
    conn.executemany(
        'INSERT INTO vectors (paper_id, vec) VALUES (?, ?) '
        'ON CONFLICT(paper_id) DO UPDATE SET vec = excluded.vec',
        rows,
    )
    conn.commit()


def load_vectors(conn):
    """Returns (ids, matrix) with matrix.shape == (len(ids), dim)."""
    rows = conn.execute('SELECT paper_id, vec FROM vectors ORDER BY paper_id').fetchall()
    if not rows:
        return [], np.zeros((0, 0), dtype=VECTOR_DTYPE)
    ids = [r['paper_id'] for r in rows]
    matrix = np.stack([np.frombuffer(r['vec'], dtype=VECTOR_DTYPE) for r in rows])
    return ids, matrix


def papers_needing_vectors(conn):
    """Papers with an abstract but no vector yet. Papers whose abstract S2 withheld are
    never embeddable and are correctly absent from similarity search."""
    rows = conn.execute(
        """
        SELECT p.paper_id, p.abstract
        FROM papers p
        LEFT JOIN vectors v ON v.paper_id = p.paper_id
        WHERE v.paper_id IS NULL AND p.abstract IS NOT NULL AND p.abstract != ''
        """
    ).fetchall()
    return [(r['paper_id'], r['abstract']) for r in rows]


def existing_vector_ids(conn, paper_ids):
    """Which of `paper_ids` already have a vector. Lets the scraper embed only the papers
    a fetch newly introduced, without a full papers/vectors anti-join each time."""
    ids = list(paper_ids)
    if not ids:
        return set()
    placeholders = ','.join('?' * len(ids))
    rows = conn.execute(
        f'SELECT paper_id FROM vectors WHERE paper_id IN ({placeholders})', ids
    ).fetchall()
    return {r['paper_id'] for r in rows}


def get_paper(conn, paper_id):
    row = conn.execute('SELECT * FROM papers WHERE paper_id = ?', (paper_id,)).fetchone()
    return dict(row) if row is not None else None


def get_papers(conn, paper_ids):
    """Returns {paper_id: row_dict} for the ids that exist."""
    if not paper_ids:
        return {}
    ids = list(paper_ids)
    placeholders = ','.join('?' * len(ids))
    rows = conn.execute(
        f'SELECT * FROM papers WHERE paper_id IN ({placeholders})', ids
    ).fetchall()
    return {r['paper_id']: dict(r) for r in rows}


def resolve_arxiv(conn, arxiv_id):
    row = conn.execute(
        'SELECT paper_id FROM papers WHERE arxiv_id = ?', (arxiv_id,)
    ).fetchone()
    return row['paper_id'] if row is not None else None


def refs_of(conn, paper_id):
    """Papers that `paper_id` cites."""
    rows = conn.execute(
        'SELECT cited_id FROM citations WHERE citing_id = ?', (paper_id,)
    ).fetchall()
    return {r['cited_id'] for r in rows}


def citers_of(conn, paper_id):
    """Papers that cite `paper_id`. This is what the index on cited_id is for."""
    rows = conn.execute(
        'SELECT citing_id FROM citations WHERE cited_id = ?', (paper_id,)
    ).fetchall()
    return {r['citing_id'] for r in rows}


def refs_of_many(conn, paper_ids):
    """Returns {paper_id: set(cited_ids)} - one query for the whole candidate set."""
    if not paper_ids:
        return {}
    ids = list(paper_ids)
    placeholders = ','.join('?' * len(ids))
    rows = conn.execute(
        f'SELECT citing_id, cited_id FROM citations WHERE citing_id IN ({placeholders})',
        ids,
    ).fetchall()
    out = {pid: set() for pid in ids}
    for r in rows:
        out[r['citing_id']].add(r['cited_id'])
    return out


def neighborhood(conn, paper_id, depth):
    """BFS `depth` hops over citation edges in both directions. Excludes the seed."""
    seen = {paper_id}
    frontier = {paper_id}
    for _ in range(depth):
        nxt = set()
        for pid in frontier:
            nxt |= refs_of(conn, pid)
            nxt |= citers_of(conn, pid)
        frontier = nxt - seen
        seen |= frontier
        if not frontier:
            break
    return seen - {paper_id}


def search_fts(conn, query, limit):
    rows = conn.execute(
        """
        SELECT p.*, papers_fts.rank AS rank
        FROM papers_fts
        JOIN papers p ON p.paper_id = papers_fts.paper_id
        WHERE papers_fts MATCH ?
        ORDER BY papers_fts.rank
        LIMIT ?
        """,
        (query, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def counts(conn):
    return {
        'papers': conn.execute('SELECT COUNT(*) c FROM papers').fetchone()['c'],
        'expanded': conn.execute(
            'SELECT COUNT(*) c FROM papers WHERE expanded = 1').fetchone()['c'],
        'citations': conn.execute('SELECT COUNT(*) c FROM citations').fetchone()['c'],
        'vectors': conn.execute('SELECT COUNT(*) c FROM vectors').fetchone()['c'],
    }
