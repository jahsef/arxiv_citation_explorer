
# goal: create an arxiv oriented research tool

## dataset
sqlite. incremental writes + point lookups during scrape, indexes, no daemon.

```sql
papers    (arxiv_id PK, title, abstract, conclusion, published, categories)
citations (citing_id, cited_id, PK(citing_id, cited_id)) + INDEX on cited_id
vectors   (arxiv_id PK, abstract BLOB, conclusion BLOB, global BLOB)
papers_fts -- FTS5 over title + abstract
```

- **citations as edge table, not a list column** — need the reverse direction ("who cites A") indexed for the graph viz and builds-on score. list column = full scan + deserialize.
- **vectors: sqlite is persistence, memory is runtime** — load all into one numpy array at startup, brute-force matmul. no vector db needed (384d, ms at our scale). sqlite BLOB so writes stay transactional with the paper row.
- **fts5 alongside sim search, not instead of** — embeddings miss rare exact tokens (method/dataset names). complementary. ~3 lines, cheap.
- no stub rows. frontier is derivable: `cited_id NOT IN (SELECT arxiv_id FROM papers)`.

parquet/duckdb only if we pull the full arxiv metadata dump — separate read-only bulk workload.

## scraping
semantic scholar api, not pdf/latex parsing. `/paper/ARXIV:id/references` + `/citations` gives both
directions with resolved arxiv ids + citation counts. batch endpoint takes 500 ids/POST.
(openalex = no-key fallback, has a bulk snapshot if we want independence.)

input arxiv id => api call => insert rows + edges => embed abstract. recursive depth arg, cap at 2
(~40 refs/paper: depth 2 ≈ 1.6k papers, depth 3 ≈ 64k).

**mvp cuts:**
- no conclusion — only field needing full text, drags the whole parsing layer back in for one field.
  abstract-only is what most paper-sim systems use.
- no global vector — near-redundant with abstract vector once full text is gone. one abstract vector,
  `vectors` collapses to one column.
- no kaggle dump (5gb) — its only unique use was forward-edge candidates, s2 covers that.

revisit full text later if abstract embeddings prove too weak.

to generate forward edges, just find papers manually lmao. realistically shouldnt be that hard if you find like 2 main successors and then like 2 tangential papers that are future relative of the paper ur looking at.


## visualization
for visualization, we just rely on whatever was scraped previously
ie if the user wants more comprehensive results they must go back and scrape mroe themselves, nothing at this stage to scrape with

### citation graph
citation graph visualization: we can just show the citation graph from left to right (chronological). we can show by topk citations, (10 max papers, pruning the less citations papers) for the backwards (chronological) and forward. those should be differential args as well. 

### other novel ish approach

2d visualization for a given paper. x = builds on (citations), y = semantic sim.

builds-on = bibliographic coupling (kessler 1963): shared refs between candidate and given paper.
normalize by cosine `|A∩B| / sqrt(|A|·|B|)`, not `|A∩B| / |A|` — else 300-ref surveys pin to the top
of every plot. sign = direction (- = influences given paper, + = builds on it).

candidates = graph traversal only, with a depth arg. neighborhood is already bounded so we plot all
of it, no topk needed (topk on a large corpus would only surface both tails and leave the middle
empty). still want threshold settings on coupling + sim to cut noise.

mvp limitation: builds-on only defined for citation neighbors, so "high sim + disjoint literature"
(the missed-connection quadrant) is unreachable. needs corpus-wide sim search as a 2nd candidate
source — later.

compute: refs(A) => indexed lookup on cited_id. corpus-wide would be `C @ C.T` sparse.


## deployment concerns
so for local, we are able to use local llms like 7b but lets opt for llm free since relying on api is gay and if i do deploy obviously cant use local llm.
though, since ai agents are so hyped up it might be a good signal to just have them regardless like "look at my shitty research app it now has research agents!"


# models
(not sure if we even need rerankers for our usecase)
sentence-transformers/all-MiniLM-L6-v2
cross-encoder/ms-marco-TinyBERT-L-2-v2

# web interface
should be extremely minimal, im talking like raw html and js is fine i dont need some bullshit js frameworks here

# serving
uvicorn instance, fast api, http get/post for the relevant data.


# evals (encoder quality against our own citation ground truth)

citations are free ground truth. key trap: DON'T optimize sim->coupling correlation. the semantic
axis is valuable *because* it's orthogonal to citations — perfect correlation = embedding just
re-encodes the graph we already have. so these are floors / sanity checks, not maximization targets.
SPECTER2 / SciNCL are trained on citation prediction, so they top any of these by construction — use
them as the "cheat" upper bound, not the goal. if MiniLM lands close without seeing the graph, that's
the interesting result.

1. **citation-neighbor retrieval recall@k** — for a held-out paper, rank whole corpus by abstract
   cos sim; do its real citation neighbors (cites / cited-by) surface in top-k? recall@k / MRR.
   rejects broken or wrong-domain encoders. cheap, no labels needed.

2. **coupling correlation as a floor** — spearman between abstract cos sim and bibliographic coupling
   over candidate pairs. expect clearly positive; ~0 means the encoder is garbage for this domain.
   a floor check, not a target (see trap above). sweep a few encoders, read the *gap* to SPECTER, not
   the absolute.

3. **abstract embedding drift over scrape date** — centroid / distribution shift of abstract vectors
   bucketed by publication year. real signal (new fields, new terminology), honest MLOps story that
   fits the data rather than a bolted-on synthetic drift metric.