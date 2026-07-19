"""In-memory vector index.

sqlite is the persistence, this is the runtime. Treated as immutable: when a scrape job
adds papers the worker builds a whole new store and the server rebinds to it in one
assignment, so a search either sees the old matrix or the new one and never needs a lock.
"""

import numpy as np

from src import db


class VectorStore:
    """Parallel `ids` list and `matrix` rows. Kept in one object because they desync
    silently if held apart."""

    def __init__(self, ids, matrix):
        self.ids = list(ids)
        self.matrix = matrix
        self._row_of = {paper_id: i for i, paper_id in enumerate(self.ids)}

    @classmethod
    def from_db(cls, conn):
        ids, matrix = db.load_vectors(conn)
        return cls(ids, matrix)

    def __len__(self):
        return len(self.ids)

    def __contains__(self, paper_id):
        return paper_id in self._row_of

    def get(self, paper_id):
        if paper_id not in self._row_of:
            return None
        return self.matrix[self._row_of[paper_id]]

    def search(self, query_vector, top_k):
        """Brute force. Both sides are normalized, so the dot product is cosine."""
        if len(self.ids) == 0:
            return []
        scores = self.matrix @ query_vector
        k = min(top_k, len(self.ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top]

    def similarity_to(self, paper_id, candidate_ids):
        """Cosine of `paper_id` against each candidate.

        None (not 0.0) where a vector is missing - S2 withholds many abstracts, and
        "unknown" must not be confused with "dissimilar". 0.0 would silently anchor the
        low end of a min/max scaled axis.
        """
        anchor = self.get(paper_id)
        if anchor is None:
            return {candidate_id: None for candidate_id in candidate_ids}

        out = {}
        for candidate_id in candidate_ids:
            vector = self.get(candidate_id)
            out[candidate_id] = None if vector is None else float(vector @ anchor)
        return out
