"""Citation-neighborhood scraper.

`depth` is how many hops of *neighbors* get expanded: depth 0 expands only the seed,
depth 1 expands the seed plus its neighbors. Expanding a paper means fetching its own
references, which is what gives it edges. A paper we merely discovered has no edges of
its own and therefore couples 0.0 against everything - so the scatter needs depth >= 1.

Every log line is emitted *before* the slow thing it describes, so a stall is always
attributable to the phase named on the last line.
"""

import logging
import time

from src import db, embed, s2

log = logging.getLogger('scraping')


def scrape(conn, client, model, seed_arxiv_id, depth, max_citers,
           citer_min_citations, ref_min_citations, add_recommendations, rec_limit,
           expand_recommendations, embed_batch_size, on_progress):
    """BFS out from `seed_arxiv_id`.

    `on_progress` is called with a fresh dict as each paper completes; the caller may
    publish it. It is never mutated in place, so a reader always sees a whole snapshot.
    """
    started = time.monotonic()
    log.info(f'resolving seed {seed_arxiv_id}')
    seed = s2.to_row(client.get_paper(f'ARXIV:{seed_arxiv_id}'))
    db.upsert_papers(conn, [seed])
    seed_id = seed['paper_id']
    log.info(f'seed -> {seed_id[:8]} {seed["title"]}')

    # The seed is expanded FIRST and unconditionally - references (backward) and forward
    # citations. It is the anchor every metric is measured against and the user chose it
    # explicitly, so it is never skipped on a stale `expanded` flag from a prior run - that
    # is exactly what left seeds with zero references before this fix.
    on_progress({'phase': 'seed', 'expanded': 1, 'of_frontier': 1, 'current': seed_id})

    # Drain any embedding backlog from prior interrupted crawls up front, so existing
    # papers become plottable immediately (this is what left DINOv3's neighbourhood with
    # abstracts but no vectors). From here every fetch is embedded as it lands.
    embed_missing(conn, model, embed_batch_size)

    seed_refs = _expand_references(conn, client, seed_id, '  seed')
    seed_cites = _fetch_seed_citations(
        conn, client, seed_id, '  seed', max_citers, citer_min_citations)
    _embed_rows(conn, model, embed_batch_size, seed_refs + seed_cites + [seed])

    if add_recommendations:
        # Non-recursive, the seed's recommendations only - seeds the corpus with topically
        # related papers. A recommendation carries no citation edge (it need not cite the
        # seed).
        log.info(f'fetching recommendations for seed (limit {rec_limit})')
        recs = [s2.to_row(p) for p in client.get_recommendations(seed_id, rec_limit)]
        log.info(f'{len(recs)} recommendations added to corpus')
        db.upsert_papers(conn, recs)
        _embed_rows(conn, model, embed_batch_size, recs)

        if expand_recommendations:
            # Fetch each recommendation's own references, connecting them into the graph
            # and making them scatter candidates with real coupling instead of isolated
            # nodes. The seed is already expanded above, so drop it if it self-recommends.
            to_expand = [r for r in recs if r['paper_id'] != seed_id]
            log.info(f'expanding references for {len(to_expand)} recommendations')
            for j, rec in enumerate(to_expand, 1):
                rec_id = rec['paper_id']
                on_progress({
                    'phase': 'recs', 'expanded': j, 'of_frontier': len(to_expand),
                    'current': rec_id,
                })
                rec_refs = _expand_references(
                    conn, client, rec_id, f'  rec [{j}/{len(to_expand)}] {rec_id[:8]}')
                _embed_rows(conn, model, embed_batch_size, rec_refs)

    # Recurse outward from the seed's neighbours. The seed itself was hop 0 (done above);
    # `depth` counts how many further hops to expand, so depth 0 = seed only.
    expanded = db.expanded_ids(conn)
    seen = {seed_id}
    level = (db.refs_of(conn, seed_id) | db.citers_of(conn, seed_id)) - seen
    seen |= level

    for hop in range(1, depth + 1):
        # Expand the whole frontier - no cap. Crawl size is bounded by the caller's knobs
        # (depth, ref_min_citations). Most-cited first, so an interrupted crawl keeps the
        # important papers; already-expanded papers are skipped (resumability).
        pending = _order_frontier(conn, level - expanded, ref_min_citations)
        log.info(f'hop {hop}/{depth}: expanding {len(pending)} at this level')

        for i, paper_id in enumerate(pending, 1):
            tag = f'  [{i}/{len(pending)}] {paper_id[:8]}'
            on_progress({
                'hop': hop, 'of': depth,
                'expanded': i, 'of_frontier': len(pending),
                'current': paper_id,
            })
            # References only for sub-papers - a sub-paper is expanded purely so
            # coupling(seed, sub-paper) can be computed, which needs its references, not
            # its citations. Stored in full; the citation floor already gated the frontier.
            refs = _expand_references(conn, client, paper_id, tag)
            _embed_rows(conn, model, embed_batch_size, refs)
            expanded.add(paper_id)

        # Next level from the DB, not just this run's fetches, so a resumed crawl advances.
        next_level = set()
        for paper_id in level:
            next_level |= db.refs_of(conn, paper_id)
            next_level |= db.citers_of(conn, paper_id)
        next_level -= seen
        seen |= next_level
        level = next_level

        if not level:
            log.info('frontier exhausted')
            break

    embed_missing(conn, model, embed_batch_size)
    counts = db.counts(conn)
    log.info(f'done in {time.monotonic() - started:.1f}s — '
             f'papers={counts["papers"]} expanded={counts["expanded"]} '
             f'citations={counts["citations"]} vectors={counts["vectors"]}')
    return counts


def _expand_references(conn, client, paper_id, tag):
    """Fetch a paper's references, store them and the reference edges, mark it expanded.

    References are stored IN FULL - never filtered by citation count. Coupling is
    |target_refs INTERSECT candidate_refs| and needs complete reference lists on both sides;
    dropping references here zeroes out the coupling axis (a 2024 paper cites mostly
    low-cite recent work, so a high filter leaves it with no references at all). Crawl
    breadth is bounded separately, by the frontier filter in _order_frontier.
    """
    log.info(f'{tag} fetching references...')
    references = [s2.to_row(p) for p in client.get_references(paper_id)]
    log.info(f'{tag} {len(references)} references')
    db.upsert_papers(conn, references)
    db.insert_edges(conn, [(paper_id, r['paper_id']) for r in references])
    db.mark_expanded(conn, paper_id)
    return references


def _fetch_seed_citations(conn, client, paper_id, tag, max_citers, min_citations):
    """Forward direction, seed only. Citers below min_citations are dropped - newest-first
    ordering means an uncapped fetch is mostly recent near-zero-citation papers."""
    log.info(f'{tag} fetching citations (cap {max_citers}, min {min_citations} cites)...')
    fetched = [s2.to_row(p) for p in client.get_citations(paper_id, max_citers)]
    citations = [
        row for row in fetched if (row['citation_count'] or 0) >= min_citations
    ]
    log.info(f'{tag} {len(fetched)} fetched -> {len(citations)} kept after filter')
    db.upsert_papers(conn, citations)
    db.insert_edges(conn, [(c['paper_id'], paper_id) for c in citations])
    return citations


def _order_frontier(conn, candidate_ids, min_citations):
    """Which papers to expand next: those at or above min_citations, most-cited first.

    The citation floor gates the *crawl breadth*, not just what a fetch stores. Without it
    the frontier is built from whatever edges already exist in the DB - including papers
    dumped in by earlier runs at a lower (or zero) threshold - so a high min_citations
    would still expand thousands of stale low-cite papers. Filtering here makes the current
    setting actually bound the search space.
    """
    if not candidate_ids:
        return []
    rows = db.get_papers(conn, candidate_ids)
    eligible = [r for r in rows.values() if (r['citation_count'] or 0) >= min_citations]
    eligible.sort(key=lambda r: -(r['citation_count'] or 0))
    return [r['paper_id'] for r in eligible]


def embed_missing(conn, model, batch_size):
    """Embed every paper that has an abstract but no vector. Used once up front to drain a
    backlog left by prior interrupted crawls; per-fetch embedding keeps up after that."""
    pending = db.papers_needing_vectors(conn)
    if not pending:
        return

    log.info(f'embedding backlog of {len(pending)} abstract(s)...')
    started = time.monotonic()
    paper_ids = [paper_id for paper_id, _ in pending]
    abstracts = [abstract for _, abstract in pending]
    vectors = embed.embed_texts(model, abstracts, batch_size)

    log.info(f'embedded in {time.monotonic() - started:.1f}s')
    db.upsert_vectors(conn, zip(paper_ids, vectors))


def _embed_rows(conn, model, batch_size, rows):
    """Embed just-fetched papers that have an abstract and no vector yet - runs right after
    each fetch, so a long or interrupted crawl still produces usable vectors instead of
    embedding everything only at the very end. Only the papers this fetch newly introduced
    are embedded (one indexed lookup, no full-table scan)."""
    with_abstract = [r for r in rows if r['abstract']]
    if not with_abstract:
        return
    existing = db.existing_vector_ids(conn, [r['paper_id'] for r in with_abstract])
    fresh = {}
    for r in with_abstract:
        if r['paper_id'] not in existing:
            fresh[r['paper_id']] = r['abstract']  # dedup within the batch
    if not fresh:
        return
    vectors = embed.embed_texts(model, list(fresh.values()), batch_size)
    db.upsert_vectors(conn, zip(fresh.keys(), vectors))
