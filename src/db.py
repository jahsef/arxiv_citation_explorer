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


# SQLite caps bound parameters per statement (SQLITE_MAX_VARIABLE_NUMBER: 999 on old
# builds). Any IN (?,?,...) over an unbounded id set must be chunked or a deep crawl's
# frontier blows past it ("too many SQL variables").
_MAX_VARS = 900


def _chunk(seq, n=_MAX_VARS):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def existing_vector_ids(conn, paper_ids):
    """Which of `paper_ids` already have a vector. Lets the scraper embed only the papers
    a fetch newly introduced, without a full papers/vectors anti-join each time."""
    ids = list(paper_ids)
    if not ids:
        return set()
    found = set()
    for chunk in _chunk(ids):
        placeholders = ','.join('?' * len(chunk))
        rows = conn.execute(
            f'SELECT paper_id FROM vectors WHERE paper_id IN ({placeholders})', chunk
        ).fetchall()
        found.update(r['paper_id'] for r in rows)
    return found


def get_paper(conn, paper_id):
    row = conn.execute('SELECT * FROM papers WHERE paper_id = ?', (paper_id,)).fetchone()
    return dict(row) if row is not None else None


def get_papers(conn, paper_ids):
    """Returns {paper_id: row_dict} for the ids that exist."""
    ids = list(paper_ids)
    if not ids:
        return {}
    out = {}
    for chunk in _chunk(ids):
        placeholders = ','.join('?' * len(chunk))
        rows = conn.execute(
            f'SELECT * FROM papers WHERE paper_id IN ({placeholders})', chunk
        ).fetchall()
        for r in rows:
            out[r['paper_id']] = dict(r)
    return out


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


def frontier_candidates(conn, candidate_ids, min_citations):
    """paper_ids at or above min_citations, most-cited first.

    The citation floor is applied in SQL, so on a deep-crawl frontier (tens of thousands of
    candidates, of which a handful clear a high threshold) only the eligible rows cross
    into Python. Chunked for the bound-parameter cap; the small eligible set is merged and
    sorted once at the end.
    """
    ids = list(candidate_ids)
    if not ids:
        return []
    eligible = []
    for chunk in _chunk(ids):
        placeholders = ','.join('?' * len(chunk))
        rows = conn.execute(
            f'SELECT paper_id, citation_count FROM papers '
            f'WHERE paper_id IN ({placeholders}) AND COALESCE(citation_count, 0) >= ?',
            (*chunk, min_citations),
        ).fetchall()
        eligible.extend(rows)
    eligible.sort(key=lambda r: -(r['citation_count'] or 0))
    return [r['paper_id'] for r in eligible]


def refs_of_many(conn, paper_ids):
    """Returns {paper_id: set(cited_ids)} for the whole candidate set, chunked to stay
    under SQLite's bound-parameter cap."""
    ids = list(paper_ids)
    if not ids:
        return {}
    out = {pid: set() for pid in ids}
    for chunk in _chunk(ids):
        placeholders = ','.join('?' * len(chunk))
        rows = conn.execute(
            f'SELECT citing_id, cited_id FROM citations '
            f'WHERE citing_id IN ({placeholders})',
            chunk,
        ).fetchall()
        for r in rows:
            out[r['citing_id']].add(r['cited_id'])
    return out


def citers_of_many(conn, paper_ids):
    """Returns {paper_id: set(citer_ids)} - the papers citing each. Forward-edge dual of
    refs_of_many (hits the cited_id index), for co-citation coupling."""
    ids = list(paper_ids)
    if not ids:
        return {}
    out = {pid: set() for pid in ids}
    for chunk in _chunk(ids):
        placeholders = ','.join('?' * len(chunk))
        rows = conn.execute(
            f'SELECT citing_id, cited_id FROM citations '
            f'WHERE cited_id IN ({placeholders})',
            chunk,
        ).fetchall()
        for r in rows:
            out[r['cited_id']].add(r['citing_id'])
    return out


# Year filter as SQL, matching metrics._year_ok: an unknown/blank date always passes;
# otherwise the leading 4 chars of `published` (YYYY or YYYY-MM-DD) must fall in range.
_YEAR_SQL = (
    "(published IS NULL OR published = '' "
    "OR CAST(substr(published, 1, 4) AS INTEGER) BETWEEN ? AND ?)"
)


def neighborhood(conn, paper_id, depth):
    """All papers within `depth` citation hops in either direction, excluding the seed.

    One recursive CTE instead of a Python BFS that issued two queries per node. UNION
    dedups, so cycles terminate; the depth guard bounds the walk.
    """
    rows = conn.execute(
        """
        WITH RECURSIVE nb(id, d) AS (
            SELECT ?, 0
            UNION
            SELECT c.cited_id, nb.d + 1 FROM citations c JOIN nb ON c.citing_id = nb.id
                WHERE nb.d < ?
            UNION
            SELECT c.citing_id, nb.d + 1 FROM citations c JOIN nb ON c.cited_id = nb.id
                WHERE nb.d < ?
        )
        SELECT DISTINCT id FROM nb WHERE id != ?
        """,
        (paper_id, depth, depth, paper_id),
    ).fetchall()
    return {r['id'] for r in rows}


def top_papers(conn, candidate_ids, min_citations, max_citations, min_year, max_year,
               limit):
    """Top `limit` candidates by citation count, filtered by the citation band and year in
    SQL. Full row dicts. For the graph walk's per-level pruning - only the survivors cross
    into Python."""
    ids = list(candidate_ids)
    if not ids:
        return []
    collected = []
    for chunk in _chunk(ids):
        placeholders = ','.join('?' * len(chunk))
        rows = conn.execute(
            f'SELECT * FROM papers WHERE paper_id IN ({placeholders}) '
            f'AND COALESCE(citation_count, 0) BETWEEN ? AND ? AND {_YEAR_SQL}',
            (*chunk, min_citations, max_citations, min_year, max_year),
        ).fetchall()
        collected.extend(dict(r) for r in rows)
    collected.sort(key=lambda r: -(r['citation_count'] or 0))
    return collected[:limit]


def scatter_candidates(conn, candidate_ids, min_citations, max_citations, min_year,
                       max_year):
    """Expanded candidates within the citation band and year window, as {paper_id: row},
    in SQL. Filtering the neighbourhood here means coupling/cosine only run on survivors."""
    ids = list(candidate_ids)
    if not ids:
        return {}
    out = {}
    for chunk in _chunk(ids):
        placeholders = ','.join('?' * len(chunk))
        rows = conn.execute(
            f'SELECT * FROM papers WHERE paper_id IN ({placeholders}) '
            f'AND expanded = 1 AND COALESCE(citation_count, 0) BETWEEN ? AND ? '
            f'AND {_YEAR_SQL}',
            (*chunk, min_citations, max_citations, min_year, max_year),
        ).fetchall()
        for r in rows:
            out[r['paper_id']] = dict(r)
    return out


def _fts_query(text):
    # Quote each token so FTS5 reads punctuation literally instead of as operators.
    return ' '.join('"' + token.replace('"', '""') + '"' for token in text.split())


def search_fts(conn, query, limit):
    match = _fts_query(query)
    if not match:
        return []
    rows = conn.execute(
        """
        SELECT p.*, papers_fts.rank AS rank
        FROM papers_fts
        JOIN papers p ON p.paper_id = papers_fts.paper_id
        WHERE papers_fts MATCH ?
        ORDER BY papers_fts.rank
        LIMIT ?
        """,
        (match, limit),
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
