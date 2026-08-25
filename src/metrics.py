"""Bibliographic coupling and the two view payloads."""

import math

from src import db


def coupling(refs_a, refs_b):
    """Cosine-normalized bibliographic coupling (Kessler 1963).

    |A n B| / sqrt(|A| * |B|), not |A n B| / |A| - otherwise a 300-reference survey
    shares refs with everything and pins to the top of every plot.
    """
    if not refs_a or not refs_b:
        return 0.0
    return len(refs_a & refs_b) / math.sqrt(len(refs_a) * len(refs_b))


def _direction(candidate_id, target_cites, target_cited_by, target_published,
               candidate_published):
    """Sign of the builds-on axis.

    A direct citation edge settles it. Depth-2 siblings have no such edge, so fall back
    to chronology. Negative means the candidate influenced the target.
    """
    if candidate_id in target_cites:
        return -1.0, 'target_cites'
    if candidate_id in target_cited_by:
        return 1.0, 'cites_target'
    if target_published is None or candidate_published is None:
        return 0.0, 'unknown'
    if candidate_published < target_published:
        return -1.0, 'older'
    return 1.0, 'newer'


def build_scatter(conn, store, target_id, depth, min_x, min_y, min_citations,
                  max_citations, min_year, max_year, y_metric='sim'):
    """x = signed bibliographic coupling (shared references, backward).
    y = semantic similarity ('sim') or co-citation ('cocitation', shared citers, forward).

    Candidates are restricted to expanded papers - the x axis (coupling) needs each
    candidate's own references, which only expanded papers have.
    """
    target = db.get_paper(conn, target_id)
    if target is None:
        return None

    # neighbourhood via recursive CTE; expanded/citation/year filtering in SQL, so the
    # metrics only run on the survivors.
    neighbourhood = db.neighborhood(conn, target_id, depth)
    rows = db.scatter_candidates(
        conn, neighbourhood, min_citations, max_citations, min_year, max_year)
    if not rows:
        return {'target': target, 'points': [], 'candidates': 0,
                'dropped_no_abstract': 0, 'y_metric': y_metric}

    candidates = list(rows.keys())
    target_refs = db.refs_of(conn, target_id)
    target_cited_by = db.citers_of(conn, target_id)
    candidate_refs = db.refs_of_many(conn, candidates)

    # y-axis source: cosine of abstracts, or co-citation (cosine-normalized shared citers).
    if y_metric == 'cocitation':
        candidate_citers = db.citers_of_many(conn, candidates)
    else:
        similarities = store.similarity_to(target_id, candidates)

    points = []
    no_vector = 0
    for candidate_id in candidates:
        row = rows[candidate_id]

        if y_metric == 'cocitation':
            y = coupling(target_cited_by, candidate_citers[candidate_id])
        else:
            y = similarities[candidate_id]
            if y is None:
                # No abstract, so no position on the similarity axis. Dropped rather than
                # pinned to 0.0, which would read as "unrelated". (Not hit for co-citation.)
                no_vector += 1
                continue

        strength = coupling(target_refs, candidate_refs[candidate_id])
        # Axis-agnostic noise cut: |x| (coupling magnitude, both signs) and y value.
        if strength < min_x or y < min_y:
            continue

        sign, direction = _direction(
            candidate_id, target_refs, target_cited_by,
            target['published'], row['published'],
        )

        points.append({
            'paper_id': candidate_id,
            'title': row['title'],
            'arxiv_id': row['arxiv_id'],
            'published': row['published'],
            'citation_count': row['citation_count'],
            'x': sign * strength,
            'y': y,
            'coupling': strength,
            'direction': direction,
        })

    points.sort(key=lambda p: -p['coupling'])
    return {
        'target': target,
        'points': points,
        'candidates': len(candidates),
        'dropped_no_abstract': no_vector,
        'y_metric': y_metric,
    }


def _walk(conn, target_id, depth, top_k, min_citations, max_citations, min_year, max_year,
          forward):
    """One direction of the citation graph, pruned to top_k per level.

    forward=True follows papers citing the frontier; forward=False follows papers the
    frontier cites. The citation/year filter and the top_k selection run in SQL
    (db.top_papers); only the traversal bookkeeping stays in Python. The frontier is at
    most top_k wide each level, so building `step` is a handful of indexed lookups.
    """
    nodes = {}
    edges = []
    seen = {target_id}
    frontier = {target_id}

    for _ in range(depth):
        step = {}
        for paper_id in frontier:
            neighbors = (db.citers_of(conn, paper_id) if forward
                         else db.refs_of(conn, paper_id))
            for neighbor_id in neighbors - seen:
                step.setdefault(neighbor_id, set()).add(paper_id)

        if not step:
            break

        kept = db.top_papers(
            conn, step.keys(), min_citations, max_citations, min_year, max_year, top_k)
        if not kept:
            break

        for row in kept:
            neighbor_id = row['paper_id']
            nodes[neighbor_id] = row
            seen.add(neighbor_id)
            for parent_id in step[neighbor_id]:
                edges.append(
                    (neighbor_id, parent_id) if forward else (parent_id, neighbor_id)
                )

        frontier = {row['paper_id'] for row in kept}

    return nodes, edges


def build_graph(conn, target_id, back_depth, fwd_depth, top_k, min_citations,
                max_citations, min_year, max_year):
    """Chronological citation graph. The x axis is publication date, so there is no
    layout pass - the frontend just stacks within a date column.

    The target is always included regardless of the filters - it is the paper being
    viewed, not a candidate.
    """
    target = db.get_paper(conn, target_id)
    if target is None:
        return None

    back_nodes, back_edges = _walk(
        conn, target_id, back_depth, top_k, min_citations, max_citations, min_year,
        max_year, forward=False)
    fwd_nodes, fwd_edges = _walk(
        conn, target_id, fwd_depth, top_k, min_citations, max_citations, min_year,
        max_year, forward=True)

    nodes = {**back_nodes, **fwd_nodes, target_id: target}
    edges = [{'from': a, 'to': b} for a, b in back_edges + fwd_edges]

    return {
        'target': target,
        'nodes': list(nodes.values()),
        'edges': edges,
    }
