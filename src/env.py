"""Minimal .env loader - no dependency. Call load_env() before building CONFIG so
os.environ has S2_API_KEY (and anything else) in place. Existing environment variables
win over the file, so an explicit `set`/`$env:` still overrides .env."""

import os
from pathlib import Path


def load_env(path='.env'):
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, val = line.partition('=')
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
