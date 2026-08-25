"""FastAPI app. Serves the API and the static frontend from one process.

The connection here is read-only. All writes happen on the scrape worker thread, which
owns the only read-write connection (see src/jobs.py). That split is what lets a
multi-minute scrape run without blocking a single read, and it is enforced by sqlite
rather than by convention.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import db, embed, logging_setup, metrics
from src.env import load_env
from src.jobs import JobQueue
from src.vectors import VectorStore

load_env()  # populate os.environ from .env before CONFIG reads S2_API_KEY

STATIC_DIR = Path(__file__).parent / 'static'

CONFIG = {
    'db_path': 'data/papers.db',
    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
    # CPU: embedding now runs per-fetch in small batches, where GPU launch/transfer
    # overhead outweighs its throughput, and it keeps the GPU free for other work.
    'device': 'cpu',
    'embed_batch_size': 64,
    # LOG_LEVEL=DEBUG adds a one-line-per-request timing log; INFO (default) omits it.
    'log_level': getattr(logging, os.environ.get('LOG_LEVEL', 'INFO').upper(),
                         logging.INFO),
    # None = unauthenticated shared pool. `or None` so a blank .env line (S2_API_KEY=)
    # falls back to that instead of sending an empty, rejected key header.
    's2_api_key': os.environ.get('S2_API_KEY') or None,
    's2_min_interval': 1.5,
    's2_max_retries': 8,
    's2_timeout': 30.0,
    's2_page_limit': 1000,
}

STATE = {}

log = logging.getLogger('server')


@asynccontextmanager
async def lifespan(app):
    logging_setup.setup(CONFIG['log_level'])

    # Create the file and schema once with a writable connection, then drop it - from
    # here on the server never holds write access.
    bootstrap = db.connect(CONFIG['db_path'], check_same_thread=True, read_only=False)
    db.init_schema(bootstrap)
    bootstrap.close()

    conn = db.connect(CONFIG['db_path'], check_same_thread=False, read_only=True)
    STATE['conn'] = conn
    STATE['store'] = VectorStore.from_db(conn)
    STATE['model'] = embed.load_model(CONFIG['model_name'], CONFIG['device'])
    STATE['jobs'] = JobQueue(
        CONFIG, on_store_reload=lambda store: STATE.__setitem__('store', store))
    STATE['jobs'].start()

    log.info(f'serving {CONFIG["db_path"]} read-only, '
             f'{len(STATE["store"])} vectors loaded')
    yield

    STATE['jobs'].stop()
    conn.close()


app = FastAPI(title='arxiv research tool', lifespan=lifespan)


@app.middleware('http')
async def timing(request, call_next):
    started = time.monotonic()
    response = await call_next(request)
    log.debug(f'{request.method} {request.url.path} {response.status_code} '
              f'{(time.monotonic() - started) * 1000:.0f}ms')
    return response


class ScrapeRequest(BaseModel):
    arxiv_id: str
    depth: int
    max_citers: int
    citer_min_citations: int
    ref_min_citations: int
    add_recommendations: bool
    rec_limit: int
    expand_recommendations: bool


@app.get('/api/stats')
def stats():
    return db.counts(STATE['conn'])


@app.get('/api/paper/{paper_id}')
def paper(paper_id: str):
    row = db.get_paper(STATE['conn'], paper_id)
    if row is None:
        raise HTTPException(404, f'unknown paper: {paper_id}')
    return row


@app.get('/api/resolve/{arxiv_id}')
def resolve(arxiv_id: str):
    """arXiv id -> S2 paper id, for entering the app from an arXiv link."""
    paper_id = db.resolve_arxiv(STATE['conn'], arxiv_id)
    if paper_id is None:
        raise HTTPException(404, f'{arxiv_id} not scraped yet')
    return {'paper_id': paper_id, 'arxiv_id': arxiv_id}


# Effectively "no upper bound" - no paper has this many citations.
MAX_CITATIONS = 1_000_000_000


@app.get('/api/graph/{paper_id}')
def graph(paper_id: str, back_depth: int = 1, fwd_depth: int = 1, top_k: int = 10,
          min_citations: int = 0, max_citations: int = MAX_CITATIONS,
          min_year: int = 0, max_year: int = 9999):
    payload = metrics.build_graph(
        STATE['conn'], paper_id, back_depth, fwd_depth, top_k, min_citations,
        max_citations, min_year, max_year)
    if payload is None:
        raise HTTPException(404, f'unknown paper: {paper_id}')
    return payload


@app.get('/api/scatter/{paper_id}')
def scatter(paper_id: str, depth: int = 1, min_x: float = 0.0,
            min_y: float = 0.0, min_citations: int = 0,
            max_citations: int = MAX_CITATIONS, min_year: int = 0, max_year: int = 9999,
            y_metric: str = 'sim'):
    if y_metric not in ('sim', 'cocitation'):
        raise HTTPException(400, "y_metric must be 'sim' or 'cocitation'")
    payload = metrics.build_scatter(
        STATE['conn'], STATE['store'], paper_id, depth, min_x, min_y,
        min_citations, max_citations, min_year, max_year, y_metric)
    if payload is None:
        raise HTTPException(404, f'unknown paper: {paper_id}')
    return payload


@app.get('/api/search')
def search(q: str, mode: str = 'sim', limit: int = 20):
    """sim = dense retrieval, fts = exact token match. Complementary: embeddings miss
    rare literal tokens (method and dataset names), fts nails them."""
    if mode not in ('sim', 'fts'):
        raise HTTPException(400, "mode must be 'sim' or 'fts'")

    if mode == 'fts':
        return {'mode': mode, 'results': db.search_fts(STATE['conn'], q, limit)}

    query_vector = embed.embed_texts(
        STATE['model'], [q], CONFIG['embed_batch_size'])[0]
    # Bound once: a concurrent store swap must not be read twice in one request.
    store = STATE['store']
    hits = store.search(query_vector, limit)
    rows = db.get_papers(STATE['conn'], [paper_id for paper_id, _ in hits])

    results = []
    for paper_id, score in hits:
        if paper_id not in rows:
            continue
        row = dict(rows[paper_id])
        row['score'] = score
        results.append(row)
    return {'mode': mode, 'results': results}


@app.get('/api/jobs')
def jobs():
    return {'jobs': STATE['jobs'].listing()}


@app.post('/api/scrape')
def run_scrape(request: ScrapeRequest):
    """Enqueues and returns immediately. Poll /api/jobs for progress."""
    job = STATE['jobs'].submit(
        arxiv_id=request.arxiv_id,
        depth=request.depth,
        max_citers=request.max_citers,
        citer_min_citations=request.citer_min_citations,
        ref_min_citations=request.ref_min_citations,
        add_recommendations=request.add_recommendations,
        rec_limit=request.rec_limit,
        expand_recommendations=request.expand_recommendations,
    )
    return {'job_id': job['id']}


# Mounted last: this catches every path the API did not claim.
app.mount('/', StaticFiles(directory=STATIC_DIR, html=True), name='static')
