'use strict';

let currentPaper = new URLSearchParams(location.search).get('paper_id');
let graphAnim = null;  // requestAnimationFrame handle for the citation-graph force sim

const DIRECTION_COLOR = {
  target_cites: '#c4794a',   // influenced the target
  cites_target: '#4a9ec4',   // builds on the target
  older: '#8a6a4a',
  newer: '#4a7a8a',
  unknown: '#6b7280',
};

/* ---------- sidebar ---------- */

function renderHits(hits) {
  const box = $('hits');
  box.innerHTML = '';
  if (!hits.length) {
    box.innerHTML = '<div class="empty">no results</div>';
    return;
  }
  // All lookup results are shown most-cited first, regardless of search mode.
  for (const hit of [...hits].sort(byCitationsDesc)) {
    const row = document.createElement('div');
    row.className = 'hit';
    const score = hit.score === undefined ? '' : ` · ${hit.score.toFixed(3)}`;
    row.innerHTML =
      `<div>${shorten(hit.title, 90)}</div>` +
      `<div class="meta">${hit.published || '?'} · ${hit.citation_count ?? 0} cites${score}</div>`;
    row.onclick = () => selectPaper(hit.paper_id);
    box.appendChild(row);
  }
}

/* ---------- scatter ---------- */

function renderScatter(data, width, height) {
  const svg = svgEl('svg', { viewBox: `0 0 ${width} ${height}` });
  const pad = 58;

  if (!data.points.length) {
    svg.appendChild(svgEl('text', {
      class: 'axis-label', x: width / 2, y: height / 2, 'text-anchor': 'middle',
    }, `no plottable candidates — ${data.candidates} in range, ` +
       `${data.dropped_no_abstract} dropped for missing abstracts`));
    return svg;
  }

  // Absolute, not scaled to the visible set - position is the real tell.
  // x: signed coupling on [-1,1], zero (low coupling) at the centre.
  // y: semantic similarity on [min(minSim,0), 1]. Cosine is almost always positive, so the
  //    floor is normally 0 (low sim at the bottom); it only dips below 0 to reveal a
  //    genuinely negative similarity, which would otherwise be invisible.
  const yLo = Math.min(0, ...data.points.map((p) => p.y));
  const px = (v) => pad + ((v + 1) / 2) * (width - 2 * pad);
  const py = (v) => height - pad - ((v - yLo) / (1 - yLo)) * (height - 2 * pad);

  // Bounding box, with -1/+1 ticks on the edges.
  svg.appendChild(svgEl('line', {
    class: 'axis', x1: pad, y1: height - pad, x2: width - pad, y2: height - pad,
  }));
  svg.appendChild(svgEl('line', {
    class: 'axis', x1: pad, y1: pad, x2: pad, y2: height - pad,
  }));

  // Dashed vertical line at x=0 (low coupling). The horizontal sim=0 line is only drawn
  // when the floor dipped below 0 - otherwise sim=0 already is the bottom axis.
  svg.appendChild(svgEl('line', {
    class: 'axis', x1: px(0), y1: pad, x2: px(0), y2: height - pad,
    'stroke-dasharray': '3 4',
  }));
  if (yLo < 0) {
    svg.appendChild(svgEl('line', {
      class: 'axis', x1: pad, y1: py(0), x2: width - pad, y2: py(0),
      'stroke-dasharray': '3 4',
    }));
  }

  svg.appendChild(svgEl('text', {
    class: 'axis-tick', x: pad, y: height - pad + 15,
  }, '-1'));
  svg.appendChild(svgEl('text', {
    class: 'axis-tick', x: width - pad, y: height - pad + 15, 'text-anchor': 'end',
  }, '+1'));
  svg.appendChild(svgEl('text', {
    class: 'axis-label', x: width / 2, y: height - pad + 32, 'text-anchor': 'middle',
  }, '← influences it     signed bibliographic coupling     builds on it →'));

  svg.appendChild(svgEl('text', {
    class: 'axis-tick', x: pad - 6, y: height - pad, 'text-anchor': 'end',
  }, yLo === 0 ? '0' : yLo.toFixed(2)));
  svg.appendChild(svgEl('text', {
    class: 'axis-tick', x: pad - 6, y: pad + 4, 'text-anchor': 'end',
  }, '1'));
  const yLabel = data.y_metric === 'cocitation' ? 'co-citation' : 'semantic similarity';
  svg.appendChild(svgEl('text', {
    class: 'axis-label', x: 8, y: height / 2,
    transform: `rotate(-90 8 ${height / 2})`, 'text-anchor': 'middle',
  }, yLabel));

  for (const point of data.points) {
    const group = svgEl('g', { class: 'node' });
    group.appendChild(svgEl('circle', {
      cx: px(point.x), cy: py(point.y), r: blobRadius(point.citation_count),
      fill: DIRECTION_COLOR[point.direction], 'fill-opacity': 0.75,
    }));
    group.appendChild(svgEl('title', {},
      `${point.title}\n${point.published || '?'} · ${point.citation_count ?? 0} cites\n` +
      `coupling ${point.coupling.toFixed(3)} · ` +
      `${data.y_metric === 'cocitation' ? 'co-cite' : 'sim'} ${point.y.toFixed(3)} · ` +
      `${point.direction}`));
    group.onclick = () => selectPaper(point.paper_id);
    svg.appendChild(group);
  }

  if (data.dropped_no_abstract) {
    svg.appendChild(svgEl('text', {
      class: 'axis-tick', x: width - pad, y: pad - 6, 'text-anchor': 'end',
    }, `${data.dropped_no_abstract} hidden (no abstract)`));
  }
  return svg;
}

/* ---------- citation graph ---------- */

function renderGraph(data, width, height) {
  const svg = svgEl('svg', { viewBox: `0 0 ${width} ${height}` });
  const pad = 60;

  const dated = data.nodes.filter((n) => dateValue(n.published) !== null);
  if (!dated.length) {
    svg.appendChild(svgEl('text', {
      class: 'axis-label', x: width / 2, y: height / 2, 'text-anchor': 'middle',
    }, 'no dated nodes to plot'));
    return svg;
  }

  // x is pinned to publication year (±2-year padding so edge labels aren't clipped).
  // y is free and settled by a force sim - the Obsidian look, but chronology preserved.
  const rawDates = dated.map((n) => dateValue(n.published));
  const lo = Math.min(...rawDates) - 2;
  const hi = Math.max(...rawDates) + 2;
  const span = hi - lo || 1;
  const xAt = (v) => pad + ((v - lo) / span) * (width - 2 * pad);
  const midY = height / 2;

  // Simulation nodes: x fixed, y jittered so repulsion has a direction to resolve. The
  // target is pinned to the mid-line and never integrated, so it stays the visual anchor.
  const sim = dated.map((n) => {
    const isTarget = n.paper_id === data.target.paper_id;
    return {
      node: n, isTarget,
      x: xAt(dateValue(n.published)),
      y: isTarget ? midY : midY + (Math.random() - 0.5) * (height - 2 * pad) * 0.7,
      vy: 0,
    };
  });
  const byId = new Map(sim.map((s) => [s.node.paper_id, s]));
  const links = data.edges
    .map((e) => [byId.get(e.from), byId.get(e.to)])
    .filter(([a, b]) => a && b);

  // Build elements once; the sim mutates their positions each frame.
  const edgeEls = links.map(([a, b]) => {
    const line = svgEl('line', { stroke: '#333a45', 'stroke-width': 1 });
    svg.appendChild(line);
    return { a, b, line };
  });
  const nodeEls = sim.map((s) => {
    const g = svgEl('g', { class: 'node' });
    const circle = svgEl('circle', {
      r: s.isTarget ? 9 : blobRadius(s.node.citation_count),
      fill: s.isTarget ? '#c4794a' : '#4a7ec4', 'fill-opacity': 0.85,
    });
    const label = svgEl('text', { class: 'axis-label' }, shorten(s.node.title, 26));
    g.appendChild(circle);
    g.appendChild(label);
    g.appendChild(svgEl('title', {},
      `${s.node.title}\n${s.node.published || '?'} · ${s.node.citation_count ?? 0} cites`));
    g.onclick = () => selectPaper(s.node.paper_id);
    svg.appendChild(g);
    return { s, circle, label };
  });

  svg.appendChild(svgEl('text', {
    class: 'axis-tick', x: pad, y: height - 14 }, Math.floor(lo)));
  svg.appendChild(svgEl('text', {
    class: 'axis-tick', x: width - pad, y: height - 14, 'text-anchor': 'end',
  }, Math.floor(hi)));
  svg.appendChild(svgEl('text', {
    class: 'axis-label', x: width / 2, y: height - 14, 'text-anchor': 'middle',
  }, 'chronological →'));

  function paint() {
    for (const { a, b, line } of edgeEls) {
      line.setAttribute('x1', a.x); line.setAttribute('y1', a.y);
      line.setAttribute('x2', b.x); line.setAttribute('y2', b.y);
    }
    for (const { s, circle, label } of nodeEls) {
      circle.setAttribute('cx', s.x); circle.setAttribute('cy', s.y);
      label.setAttribute('x', s.x + 13); label.setAttribute('y', s.y + 4);
    }
  }

  // y-only force sim, x pinned. Repulsion is gated by horizontal proximity (only nodes
  // whose labels could actually collide push apart); links pull connected papers together;
  // a weak spring keeps the whole thing centred. Velocity is clamped so it can't explode.
  const yMin = pad;
  const yMax = height - pad;
  const xInfluence = 160;  // ~label width: beyond this, no vertical overlap is possible
  function tick() {
    for (let i = 0; i < sim.length; i++) {
      const a = sim[i];
      if (a.isTarget) continue;
      let fy = (midY - a.y) * 0.015;  // centring
      for (let j = 0; j < sim.length; j++) {
        if (i === j) continue;
        const b = sim[j];
        const dx = Math.abs(a.x - b.x);
        if (dx > xInfluence) continue;
        let dy = a.y - b.y;
        if (Math.abs(dy) < 1) dy = Math.sign(dy || 1);
        const prox = 1 - dx / xInfluence;         // 1 at same x, 0 at the edge
        fy += prox * 350 / (Math.sign(dy) * Math.max(Math.abs(dy), 6));
      }
      a.vy = Math.max(-20, Math.min(20, (a.vy + fy) * 0.85));
    }
    for (const [a, b] of links) {           // springs pull connected y together
      const pull = (b.y - a.y) * 0.015;
      if (!a.isTarget) a.vy += pull;
      if (!b.isTarget) b.vy -= pull;
    }
    for (const a of sim) {
      if (a.isTarget) continue;
      a.y = Math.max(yMin, Math.min(yMax, a.y + a.vy));
    }
  }

  let frame = 0;
  function loop() {
    tick();
    paint();
    frame += 1;
    graphAnim = frame < 220 ? requestAnimationFrame(loop) : null;
  }
  paint();
  loop();
  return svg;
}

/* ---------- orchestration ---------- */

async function draw() {
  if (!currentPaper) return;
  // Stop any in-flight graph sim before it mutates a soon-to-be-detached SVG.
  if (graphAnim) { cancelAnimationFrame(graphAnim); graphAnim = null; }
  const view = $('view');
  const width = view.clientWidth || 900;
  const height = view.clientHeight || 600;

  const minCites = Number($('minCitations').value) || 0;
  const maxCites = Number($('maxCitations').value) || 1000000000;  // blank = no upper limit
  const depth = Number($('depth').value) || 1;  // citation hops out from the base paper
  const minX = Number($('minX').value) || 0;  // scatter-only, axis-agnostic noise filters
  const minY = Number($('minY').value) || 0;
  // Blank year -> the endpoint's own no-filter default (0 / 9999).
  const minYear = Number($('minYear').value) || 0;
  const maxYear = Number($('maxYear').value) || 9999;
  const yearQs = `&min_year=${minYear}&max_year=${maxYear}`;
  const mode = $('viewmode').value;
  const isScatter = mode === 'scatter' || mode === 'cocite';
  const yMetric = mode === 'cocite' ? 'cocitation' : 'sim';
  // The axis-min filters only apply to the scatter - hide them on the graph.
  $('axisMinCtl').style.display = isScatter ? 'flex' : 'none';
  $('axishint').textContent = !isScatter ? ''
    : yMetric === 'cocitation'
      ? 'x = bib coupling (shared references, past) · y = co-citation (shared citers, future); low = middle/bottom'
      : 'middle of x axis = low bib coupling, bottom of y axis = low abstract semantic sim';
  status('loading…');
  try {
    const citeQs = `&min_citations=${minCites}&max_citations=${maxCites}`;
    const svg = isScatter
      ? renderScatter(await api(
          `/api/scatter/${currentPaper}?depth=${depth}${citeQs}` +
          `&min_x=${minX}&min_y=${minY}&y_metric=${yMetric}${yearQs}`), width, height)
      : renderGraph(await api(
          `/api/graph/${currentPaper}?back_depth=${depth}&fwd_depth=${depth}` +
          `${citeQs}${yearQs}`), width, height);
    view.innerHTML = '';
    view.appendChild(svg);
    status('');
  } catch (err) {
    status(err.message);
  }
}

async function selectPaper(paperId) {
  currentPaper = paperId;
  history.replaceState(null, '', `vis.html?paper_id=${paperId}`);
  const paper = await api(`/api/paper/${paperId}`);
  const link = paper.arxiv_id
    ? ` · <a href="https://arxiv.org/abs/${paper.arxiv_id}" target="_blank">arxiv:${paper.arxiv_id}</a>`
    : '';
  $('title').innerHTML =
    `<b>${paper.title || '(untitled)'}</b> — ${paper.published || '?'} · ` +
    `${paper.citation_count ?? 0} cites${link}`;
  await draw();
}

// Look up an already-scraped paper by S2 id or arXiv id, then open it directly.
// Tries the s2-id path first, falls back to arxiv resolution - both only see the corpus,
// which is what "lookup" means here (no scraping).
async function lookupId(q) {
  const asS2 = async () => (await api(`/api/paper/${encodeURIComponent(q)}`)).paper_id;
  const asArxiv = async () => (await api(`/api/resolve/${encodeURIComponent(q)}`)).paper_id;
  // An S2 paper id is 40 hex chars; anything else (e.g. 2501.12948) is an arXiv id. Probe
  // the matching endpoint first so a normal arXiv lookup doesn't log a spurious 404. The
  // other stays as a fallback for the odd case.
  const [first, second] = /^[0-9a-f]{40}$/i.test(q) ? [asS2, asArxiv] : [asArxiv, asS2];
  try {
    await selectPaper(await first());
  } catch (_) {
    await selectPaper(await second());
  }
}

async function runSearch() {
  const q = $('q').value.trim();
  if (!q) return;
  const mode = $('mode').value;
  status(mode === 'id' ? 'looking up…' : 'searching…');
  try {
    if (mode === 'id') {
      await lookupId(q);
    } else {
      renderHits((await api(`/api/search?q=${encodeURIComponent(q)}&mode=${mode}`)).results);
    }
    status('');
  } catch (err) {
    status(mode === 'id' ? `not in corpus: ${q}` : err.message);
  }
}

function onModeChange() {
  const idMode = $('mode').value === 'id';
  $('q').placeholder = idMode ? 'arxiv id or s2 id…' : 'search…';
  if ($('q').value.trim()) runSearch();
}

$('q').onkeydown = (e) => { if (e.key === 'Enter') runSearch(); };
$('mode').onchange = onModeChange;
$('viewmode').onchange = draw;
$('depth').onchange = draw;
$('minX').onchange = draw;
$('minY').onchange = draw;
$('minCitations').onchange = draw;
$('maxCitations').onchange = draw;
$('minYear').onchange = draw;
$('maxYear').onchange = draw;
window.onresize = draw;

if (currentPaper) selectPaper(currentPaper);
