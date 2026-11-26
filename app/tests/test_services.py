"""Ekstraksi logika murni dari `ui/*` ke `services/*` (Fase 1 UI web).

Dua jaminan yang diperiksa mekanis:
  1. `services` tidak mengimpor Streamlit (juga tidak lewat `landcover`/`launchers`), jadi API FastAPI
     tidak menarik Streamlit.
  2. Nama yang dipakai ulang oleh `ui/*` adalah OBJEK YANG SAMA dengan yang di `services/*` (bukan
     salinan ganda): `services.X.f is ui.X.f`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SERVICE_MODULES = (
    "services.runs", "services.compare", "services.charts", "services.verification",
    "services.params", "services.maps", "services.labeling_panel", "services.datasets",
)

# (modul services, modul ui, nama yang harus identik)
SHARED = [
    ("services.runs", "ui.runs", name) for name in (
        "ACTIVE_KEY", "CLASS_MAPS_KEY", "DEVICE_LABELS", "KMEANS_LABELS", "MODE_LABELS", "NOTES_KEY",
        "Outcome", "RUNS_KEY", "TICK_SECONDS", "class_map_for", "created_text", "describe_run",
        "ensure_state", "execute", "find_saved_run", "obtain_run", "register_run", "run_id_of", "run_stamp",
    )
] + [
    ("services.compare", "ui.compare", name) for name in (
        "EMPTY", "MOCK_NOTICE", "comparable_runs", "format_seconds", "speedup", "timer_text",
    )
] + [
    ("services.charts", "ui.charts", name) for name in (
        "DEVICE_COLORS", "DEVICE_NAMES", "DEVICE_ORDER", "STAGE_NAMES", "TOTAL_NAME",
        "UNKNOWN_DEVICE_COLOR", "device_color", "device_name", "format_duration", "run_label", "run_labels",
    )
] + [
    ("services.verification", "ui.verify_badge", name) for name in (
        "ICON_BAD", "ICON_OK", "ICON_PARTIAL", "ICON_WARN", "MOCK_VERIFY_NOTICE", "TITLE", "detail_rows",
        "format_identical_pct", "format_pixels", "source_text", "verdict_text",
    )
] + [
    ("services.params", "ui.sidebar", name) for name in (
        "FormValues", "build_pair", "build_params", "K_DEFAULT", "K_MIN", "ITERS_DEFAULT", "SEED_DEFAULT",
        "THREADS_DEFAULT",
    )
] + [
    ("services.maps", "ui.maps", name) for name in (
        "VIEW_CENTROID", "VIEW_CLASS", "VIEW_LABELS", "class_map_image", "find_original",
    )
] + [
    ("services.labeling_panel", "ui.labeling_panel", name) for name in (
        "COL_CLASS", "COL_CLUSTER", "COL_COLOR", "COL_OTHER", "COL_PERCENT", "COL_PIXELS", "EDITOR_COLUMNS",
        "LABELS_FILE", "Summary", "area_csv", "class_map_from_frame", "class_map_png", "cluster_csv",
        "editor_frame", "export_full_map", "export_names", "format_ha", "format_percent", "labels_file_of",
        "pixel_size_of", "resolve_name", "split_name", "summary_of", "swatch_data_url",
    )
]


def test_services_tidak_mengimpor_streamlit():
    code = (
        "import sys\n"
        + "".join(f"import {m}\n" for m in SERVICE_MODULES)
        + "assert 'streamlit' not in sys.modules, 'streamlit ikut terimpor'\n"
        + "assert 'plotly' not in sys.modules, 'plotly ikut terimpor'\n"
        + "assert not any(m == 'ui' or m.startswith('ui.') for m in sys.modules), 'ui ikut terimpor'\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=None,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )
    assert proc.returncode == 0, proc.stderr[-800:]


@pytest.mark.parametrize("service_module, ui_module, name", SHARED)
def test_ui_memakai_objek_yang_sama_dengan_services(service_module, ui_module, name):
    import importlib

    service = importlib.import_module(service_module)
    ui = importlib.import_module(ui_module)
    assert getattr(ui, name) is getattr(service, name)


def test_device_names_dan_device_labels_satu_objek():
    from services.charts import DEVICE_NAMES
    from services.runs import DEVICE_LABELS

    assert DEVICE_NAMES is DEVICE_LABELS


def test_verdict_level_dan_gaya_streamlit_konsisten():
    from landcover.verify import SOURCE_LABELS, SOURCE_PREVIEW, Verification
    from services import verification as sv
    from ui.verify_badge import verdict_style

    def make(status, full):
        return Verification(
            status=status, source=SOURCE_LABELS if full else SOURCE_PREVIEW, n_compared=10, n_diff=0,
            identical_pct=100.0, centroids_equal=True, counts_equal=True, reasons=(), differences=(),
            notes=(), mock=False,
        )

    for status, full, level, color in (
        ("identical", True, sv.LEVEL_OK, "green"),
        ("identical", False, sv.LEVEL_PARTIAL, "blue"),
        ("different", True, sv.LEVEL_BAD, "red"),
        ("not_comparable", False, sv.LEVEL_WARN, "orange"),
    ):
        v = make(status, full)
        assert sv.verdict_level(v) == level
        assert verdict_style(v) == (sv.LEVEL_ICONS[level], color)


def test_kalimat_bantuan_dan_label_ekspor_hanya_satu_sumber():
    """`ui/labeling_panel.py` (Streamlit) mengimpor kalimat bantuan dan label ekspor dari `services`, tidak menyalin ganda."""
    import ui.labeling_panel as ui_panel
    from services import labeling_panel as svc_panel

    for name in ("HELP_PANEL", "HELP_OTHER", "HELP_CLASS", "EXPORT_LABELS"):
        assert getattr(ui_panel, name) is getattr(svc_panel, name), name
    # kalimat yang dulu tertulis ganda di ui/ memang masih sama dengan yang dipakai API
    assert svc_panel.EXPORT_LABELS["map"] == "Peta kelas (PNG pratinjau)"
    assert svc_panel.HELP_PANEL.startswith("Beri nama tiap cluster")
