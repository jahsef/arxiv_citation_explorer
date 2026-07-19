'use strict';

const SVGNS = 'http://www.w3.org/2000/svg';
const $ = (id) => document.getElementById(id);

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail || 'request failed');
  }
  return response.json();
}

function status(message) {
  const node = $('status');
  if (node) node.textContent = message;
}

function svgEl(tag, attrs, text) {
  const node = document.createElementNS(SVGNS, tag);
  for (const key in attrs) node.setAttribute(key, attrs[key]);
  if (text !== undefined) node.textContent = text;
  return node;
}

function shorten(text, n) {
  if (!text) return '(untitled)';
  return text.length > n ? text.slice(0, n - 1) + '…' : text;
}

// "2017-06-12" or "2017" -> sortable float. Bare years land mid-year.
function dateValue(published) {
  if (!published) return null;
  const parts = published.split('-').map(Number);
  const year = parts[0];
  const month = parts.length > 1 ? parts[1] : 6;
  const day = parts.length > 2 ? parts[2] : 15;
  return year + (month - 1) / 12 + day / 365;
}

// Citation-driven node radius. Scale is deliberately mild - citation counts span four
// orders of magnitude, so anything steeper makes landmark papers swallow the plot.
function blobRadius(citationCount) {
  return 4 + Math.min(10, Math.sqrt(citationCount || 0) / 4.8);
}

// Shared ordering for lookup results: most-cited first. Missing counts sort last.
function byCitationsDesc(a, b) {
  return (b.citation_count || 0) - (a.citation_count || 0);
}

function markActiveNav() {
  const here = location.pathname.split('/').pop() || 'index.html';
  for (const link of document.querySelectorAll('nav a')) {
    if (link.getAttribute('href') === here) link.classList.add('active');
  }
}

document.addEventListener('DOMContentLoaded', markActiveNav);
