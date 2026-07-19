"""Semantic Scholar Academic Graph API client.

Unauthenticated traffic shares a global pool and gets throttled unpredictably, so every
request goes through a self-imposed rate limit with jittered backoff on 429. The API
documents no Retry-After header and no backoff policy - the strategy here is judgment.
"""

import random
import time

import requests

BASE_URL = 'https://api.semanticscholar.org/graph/v1'
REC_BASE_URL = 'https://api.semanticscholar.org/recommendations/v1'

# Requested as bare names; on /references and /citations they arrive wrapped in
# citedPaper / citingPaper respectively.
PAPER_FIELDS = (
    'paperId,externalIds,title,abstract,publicationDate,year,citationCount,fieldsOfStudy'
)


class S2Error(RuntimeError):
    pass


def to_row(paper):
    """S2 paper JSON -> our `papers` row shape.

    S2 responses are sparse by design: abstracts are often withheld for licensing reasons
    and externalIds only carries keys it knows. Note the case asymmetry - requests use the
    ARXIV: prefix, responses use the ArXiv key.
    """
    external = paper.get('externalIds') or {}
    fields = paper.get('fieldsOfStudy') or []
    year = paper.get('year')

    published = paper.get('publicationDate')
    if published is None and year is not None:
        published = str(year)

    return {
        'paper_id': paper['paperId'],
        'arxiv_id': external.get('ArXiv'),
        'title': paper.get('title'),
        'abstract': paper.get('abstract'),
        'published': published,
        'categories': ','.join(fields) if fields else None,
        'citation_count': paper.get('citationCount'),
    }


class S2Client:
    def __init__(self, api_key, min_interval, max_retries, timeout, page_limit):
        self.api_key = api_key
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.page_limit = page_limit
        self.session = requests.Session()
        self._last_request_at = 0.0

    def _headers(self):
        return {'x-api-key': self.api_key} if self.api_key is not None else {}

    def _throttle(self):
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_at = time.monotonic()

    def _get(self, path, params, base=BASE_URL):
        url = f'{base}{path}'
        for attempt in range(self.max_retries + 1):
            self._throttle()
            response = self.session.get(
                url, params=params, headers=self._headers(), timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json()

            if response.status_code == 404:
                raise S2Error(f'not found: {url}')

            retryable = response.status_code == 429 or response.status_code >= 500
            if not retryable or attempt == self.max_retries:
                raise S2Error(f'{response.status_code} from {url}: {response.text[:200]}')

            backoff = (2 ** attempt) + random.uniform(0, 1)
            print(f'  {response.status_code}, retrying in {backoff:.1f}s '
                  f'({attempt + 1}/{self.max_retries})')
            time.sleep(backoff)

        raise S2Error(f'exhausted retries: {url}')

    def get_paper(self, paper_id):
        """paper_id may be a bare S2 id or a prefixed form such as ARXIV:1706.03762."""
        return self._get(f'/paper/{paper_id}', {'fields': PAPER_FIELDS})

    def _paged(self, path, wrapper_key, max_records):
        """Walk offset/next pagination, unwrapping citedPaper / citingPaper.

        `next` is absent (not null) on the final page - that absence is the stop signal.
        Entries whose paper has no paperId are records S2 could not resolve; drop them.

        max_records=None fetches everything.
        """
        papers = []
        offset = 0
        while True:
            payload = self._get(
                path,
                {'fields': PAPER_FIELDS, 'offset': offset, 'limit': self.page_limit},
            )
            # Undocumented: `data` comes back null (not []) for some papers rather than
            # an empty list, so this cannot be iterated blind.
            for entry in payload['data'] or []:
                paper = entry[wrapper_key]
                if paper is not None and paper.get('paperId') is not None:
                    papers.append(paper)

            if max_records is not None and len(papers) >= max_records:
                return papers[:max_records]
            if 'next' not in payload:
                return papers
            offset = payload['next']

    def get_references(self, paper_id):
        """Papers that `paper_id` cites. Deliberately uncapped.

        Bibliographic coupling is computed from the full reference list, and rare shared
        references carry more signal than common ones - two papers both citing an obscure
        work says far more than both citing Adam. Truncating here corrupts the metric.
        Real bibliographies are 20-60 entries, so the cost is bounded anyway.
        """
        return self._paged(f'/paper/{paper_id}/references', 'citedPaper', None)

    def get_recommendations(self, paper_id, limit):
        """Topically-related papers from S2's recommender - a single call, no pagination.

        These are NOT citation edges: a recommended paper need not cite the target. The
        caller adds them as bare corpus papers with no edge. Much stronger as a discovery
        source than newest-first citations, which for a landmark paper are all recent
        near-zero-citation preprints. `all-cs` pools from the whole CS corpus rather than
        recent papers only.
        """
        payload = self._get(
            f'/papers/forpaper/{paper_id}',
            {'fields': PAPER_FIELDS, 'limit': limit, 'from': 'all-cs'},
            base=REC_BASE_URL,
        )
        return [
            p for p in (payload.get('recommendedPapers') or [])
            if p.get('paperId') is not None
        ]

    def get_citations(self, paper_id, max_records):
        """Papers that cite `paper_id`. Capped - citers are a discovery channel, not a
        metric input.

        Note the ordering: S2 returns these newest-first, so a cap takes the most *recent*
        citers, not the most important ones. For a landmark paper that means brand-new
        preprints with near-zero citations of their own, which is what the caller's
        minimum-citation filter exists to clean up.
        """
        return self._paged(f'/paper/{paper_id}/citations', 'citingPaper', max_records)
