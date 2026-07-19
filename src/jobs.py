"""Single-threaded scrape queue.

The worker thread owns the only read-write sqlite connection in the process; the server
holds a read-only one. WAL lets those coexist, so there is no application-level lock
anywhere - which is why a multi-minute scrape no longer blocks reads.

Job state is published by rebinding whole values (`job['progress'] = {...}`), never by
mutating a shared structure in place. Under the GIL a reader therefore sees either the
previous snapshot or the next one, never a torn one.
"""

import itertools
import logging
import queue
import threading
import time
from collections import deque

from src import db, embed, s2, scrape
from src.vectors import VectorStore

log = logging.getLogger('scraping')

MAX_TRACKED_JOBS = 50


class JobQueue:
    def __init__(self, config, on_store_reload):
        self.config = config
        self.on_store_reload = on_store_reload
        self._pending = queue.Queue()
        self._jobs = deque(maxlen=MAX_TRACKED_JOBS)
        self._ids = itertools.count(1)
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(
            target=self._run, name='scrape-worker', daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def submit(self, arxiv_id, depth, max_citers, citer_min_citations,
               ref_min_citations, add_recommendations, rec_limit, expand_recommendations):
        job = {
            'id': next(self._ids),
            'arxiv_id': arxiv_id,
            'depth': depth,
            'max_citers': max_citers,
            'citer_min_citations': citer_min_citations,
            'ref_min_citations': ref_min_citations,
            'add_recommendations': add_recommendations,
            'rec_limit': rec_limit,
            'expand_recommendations': expand_recommendations,
            'state': 'queued',
            'queued_at': time.time(),
            'started_at': None,
            'finished_at': None,
            'progress': None,
            'result': None,
            'paper_id': None,
            'error': None,
        }
        self._jobs.append(job)
        self._pending.put(job)
        log.info(f'job {job["id"]} queued: {arxiv_id} '
                 f'(position {self._pending.qsize()})')
        return job

    def listing(self):
        """Newest first. Plain dicts, safe to serialize while the worker runs."""
        return list(self._jobs)[::-1]

    def _run(self):
        # Created inside the worker thread and never shared, so check_same_thread stays on.
        conn = db.connect(
            self.config['db_path'], check_same_thread=True, read_only=False)
        db.init_schema(conn)

        log.info(f'worker: loading {self.config["model_name"]} '
                 f'on {self.config["device"]}')
        model = embed.load_model(
            self.config['model_name'], self.config['device'])
        client = s2.S2Client(
            api_key=self.config['s2_api_key'],
            min_interval=self.config['s2_min_interval'],
            max_retries=self.config['s2_max_retries'],
            timeout=self.config['s2_timeout'],
            page_limit=self.config['s2_page_limit'],
        )
        log.info('worker: ready')

        while not self._stop.is_set():
            try:
                job = self._pending.get(timeout=0.5)
            except queue.Empty:
                continue
            self._execute(job, conn, client, model)

        conn.close()

    def _execute(self, job, conn, client, model):
        job['state'] = 'running'
        job['started_at'] = time.time()
        log.info(f'job {job["id"]} start: {job["arxiv_id"]} depth={job["depth"]} '
                 f'max_citers={job["max_citers"]} '
                 f'min_citations={job["citer_min_citations"]}')

        try:
            job['result'] = scrape.scrape(
                conn=conn,
                client=client,
                model=model,
                seed_arxiv_id=job['arxiv_id'],
                depth=job['depth'],
                max_citers=job['max_citers'],
                citer_min_citations=job['citer_min_citations'],
                ref_min_citations=job['ref_min_citations'],
                add_recommendations=job['add_recommendations'],
                rec_limit=job['rec_limit'],
                expand_recommendations=job['expand_recommendations'],
                embed_batch_size=self.config['embed_batch_size'],
                on_progress=lambda snapshot: job.__setitem__('progress', snapshot),
            )
            job['paper_id'] = db.resolve_arxiv(conn, job['arxiv_id'])
            job['state'] = 'done'

            # Rebuild and hand over a whole new store. One atomic rebind on the server
            # side, so searches mid-swap read a complete matrix either way.
            store = VectorStore.from_db(conn)
            log.info(f'job {job["id"]} reloading vector store ({len(store)} vectors)')
            self.on_store_reload(store)

        except Exception as exc:
            # A failed job must not take the worker down - the queue keeps going.
            job['state'] = 'failed'
            job['error'] = str(exc)
            log.error(f'job {job["id"]} failed: {exc}')

        finally:
            job['finished_at'] = time.time()
