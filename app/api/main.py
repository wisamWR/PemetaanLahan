"""Titik masuk ASGI: `uvicorn api.main:app --app-dir app` (dari root repo).

Membaca `config` (LC_BACKEND, LC_DATA_DIR, LC_RUNS_DIR, LC_CORE_BIN) saat diimpor. Menyajikan
`web/dist` bila sudah dibangun (`npm run build` di `web/`): satu proses untuk demo.
"""

from __future__ import annotations

from api.server import create_app

app = create_app()
