"""Scrape a citation neighborhood into sqlite.

    python scrape_cli.py --arxiv-id 1706.03762 --depth 1

Depth counts hops of *neighbors* that get expanded: depth 0 expands only the seed,
depth 1 expands the seed plus its neighbors. Expanding is what gives a paper edges, so
the scatter needs depth >= 1 — at depth 0 every candidate couples to zero. Depth 0 is
enough for the citation graph.

Note this writes to the same database the server reads. The server holds a read-only
connection, so running both at once is safe.
"""

import argparse
import logging
import os

from src import db, embed, logging_setup, s2, scrape
from src.env import load_env

load_env()  # populate os.environ from .env before CONFIG reads S2_API_KEY

CONFIG = {
    'db_path': 'data/papers.db',
    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
    # CPU: per-fetch embedding is small-batch, where GPU overhead outweighs its throughput.
    'device': 'cpu',
    'embed_batch_size': 64,
    'log_level': logging.INFO,
    # None = unauthenticated shared pool. `or None` so a blank .env line (S2_API_KEY=)
    # falls back to that instead of sending an empty, rejected key header.
    's2_api_key': os.environ.get('S2_API_KEY') or None,
    's2_min_interval': 2,
    's2_max_retries': 8,
    's2_timeout': 30.0,
    's2_page_limit': 1000,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arxiv-id', required=True, help='seed paper, e.g. 1706.03762')
    parser.add_argument('--depth', type=int, required=True,
                        help='hops of neighbors to expand (0 = seed only)')
    parser.add_argument('--max-citers', type=int, default=200,
                        help='cap on citations fetched per paper (references are '
                             'always fetched in full)')
    parser.add_argument('--min-citations', type=int, default=10,
                        help='drop citers below this citation count (forward)')
    parser.add_argument('--ref-min-citations', type=int, default=0,
                        help='drop references below this citation count (backward); '
                             'raise for deep crawls to converge on foundational papers')
    parser.add_argument('--recommendations', action='store_true',
                        help='also add S2-recommended papers for the seed to the corpus')
    parser.add_argument('--rec-limit', type=int, default=500,
                        help='max recommended papers to add (S2 caps at 500)')
    parser.add_argument('--expand-recommendations', action='store_true',
                        help='also fetch references of recommended papers (connects them '
                             'to the graph and makes them scatter candidates)')
    args = parser.parse_args()

    logging_setup.setup(CONFIG['log_level'])

    conn = db.connect(CONFIG['db_path'], check_same_thread=True, read_only=False)
    db.init_schema(conn)

    client = s2.S2Client(
        api_key=CONFIG['s2_api_key'],
        min_interval=CONFIG['s2_min_interval'],
        max_retries=CONFIG['s2_max_retries'],
        timeout=CONFIG['s2_timeout'],
        page_limit=CONFIG['s2_page_limit'],
    )
    model = embed.load_model(CONFIG['model_name'], CONFIG['device'])

    scrape.scrape(
        conn=conn,
        client=client,
        model=model,
        seed_arxiv_id=args.arxiv_id,
        depth=args.depth,
        max_citers=args.max_citers,
        citer_min_citations=args.min_citations,
        ref_min_citations=args.ref_min_citations,
        add_recommendations=args.recommendations,
        rec_limit=args.rec_limit,
        expand_recommendations=args.expand_recommendations,
        embed_batch_size=CONFIG['embed_batch_size'],
        on_progress=lambda snapshot: None,
    )
    conn.close()


if __name__ == '__main__':
    main()
