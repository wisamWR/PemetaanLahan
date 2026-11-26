"""Fixture bersama. `make_run_dir` dipakai semua tes yang butuh folder run."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Palet centroid tetap untuk K <= 8 (tiap warna jelas berbeda).
_PALETTE = np.array(
    [[34, 61, 40], [201, 198, 190], [60, 120, 200], [180, 140, 60],
     [90, 160, 70], [120, 120, 120], [220, 60, 60], [10, 10, 10]],
    dtype=np.int64,
)


def default_labels(width: int, height: int, k: int) -> np.ndarray:
    """Label deterministik (h, w); semua k cluster muncul bila width >= k."""
    cols = (np.arange(width)[None, :] * k) // width
    rows = np.arange(height)[:, None]
    return ((cols + rows) % k).astype(np.int64)


def write_run_dir(
    target: Path,
    *,
    width: int = 64,
    height: int = 48,
    k: int = 4,
    meta: dict | None = None,
    timing: dict | None = None,
    centroids=None,
    counts=None,
    labels=None,
    drop: tuple[str, ...] = (),
) -> Path:
    """Tulis folder run yang valid (kontrak 4.3) lalu terapkan override.

    meta/timing: dict yang digabung ke bawaan; nilai None menghapus kuncinya.
    centroids/counts/labels: mengganti bawaan (counts bawaan = bincount(labels)).
    drop: nama file yang tidak ditulis.
    """
    target.mkdir(parents=True, exist_ok=True)

    labels_arr = default_labels(width, height, k) if labels is None else np.asarray(labels)
    centroids_arr = _PALETTE[:k] if centroids is None else np.asarray(centroids)
    counts_arr = (
        np.bincount(labels_arr.ravel(), minlength=k) if counts is None else np.asarray(counts)
    )
    # Peta warna selalu dari palet bawaan, supaya override centroid yang rusak
    # (mis. di luar 0-255) tidak membuat penulisan PNG gagal.
    colors = _PALETTE[np.clip(labels_arr, 0, len(_PALETTE) - 1)].astype(np.uint8)

    base_meta = {
        "contract_version": 1,
        "mock": True,
        "dataset": "mock_kecil",
        "mode": "opt",
        "kmeans_device": "cuda",
        "width": width,
        "height": height,
        "pixel_size_m": 10,
        "k": k,
        "iters": 20,
        "seed": 42,
        "threads": 8,
        "init_centroids": _PALETTE[:k].tolist(),
        "preview_width": width,
        "preview_height": height,
        "host": "uji: CPU ≥ 8 inti, GPU ±6 GB",  # non-ASCII: menjaga baca/tulis UTF-8
        "created_at": "2026-11-21T10:00:00+07:00",
    }
    base_timing = {
        "unit": "ms",
        "read_ms": 120.0, "init_ms": 1.0, "kmeans_ms": 800.0, "color_ms": 60.0, "write_ms": 200.0,
        "total_ms": 1200.0,
        "h2d_ms": 20.0, "kernel_ms": 700.0, "d2h_ms": 10.0,
    }
    for base, override in ((base_meta, meta), (base_timing, timing)):
        for key, value in (override or {}).items():
            if value is None:
                base.pop(key, None)
            else:
                base[key] = value

    def dump(name: str, data) -> None:
        if name not in drop:
            (target / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    dump("meta.json", base_meta)
    dump("timing.json", base_timing)
    dump("centroids.json", centroids_arr.tolist())
    dump("counts.json", counts_arr.tolist())
    if "labels_preview.png" not in drop:
        Image.fromarray(labels_arr.astype(np.uint8)).save(target / "labels_preview.png")
    if "map_preview.png" not in drop:
        Image.fromarray(colors).save(target / "map_preview.png")
    return target


@pytest.fixture
def make_run_dir(tmp_path):
    """Pabrik folder run valid: `make_run_dir(**override)` -> Path (folder baru tiap panggilan)."""
    counter = itertools.count()

    def factory(name: str | None = None, **override) -> Path:
        return write_run_dir(tmp_path / (name or f"run_{next(counter)}"), **override)

    return factory
