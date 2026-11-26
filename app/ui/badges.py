"""Lencana kecil yang dipakai di banyak tempat (teks markdown Streamlit)."""

from __future__ import annotations

MOCK_BADGE = ":orange-badge[DATA TIRUAN]"


def backend_badge(backend: str) -> str:
    """Lencana backend aktif. Data tiruan harus selalu terlihat jelas."""
    if backend == "mock":
        return ":orange-badge[Backend: DATA TIRUAN (mock)]"
    return ":green-badge[Backend: core C++ (cli)]"
