"""Uji headless entry point Streamlit (tanpa server, tanpa browser)."""

from streamlit.testing.v1 import AppTest

import config
from ui.badges import backend_badge

APP_FILE = str(config.ROOT / "app" / "streamlit_app.py")


def jalankan_app(monkeypatch, backend: str) -> AppTest:
    # config.BACKEND dibekukan saat import; timpa langsung supaya hasil tes
    # tidak bergantung pada LC_BACKEND di shell yang menjalankan pytest.
    monkeypatch.setattr(config, "BACKEND", backend)
    at = AppTest.from_file(APP_FILE).run()
    assert not at.exception
    return at


def test_menampilkan_judul_dan_lencana_mock(monkeypatch):
    at = jalankan_app(monkeypatch, "mock")
    assert [t.value for t in at.title] == ["Pemetaan Tutupan Lahan"]
    assert any("DATA TIRUAN" in m.value for m in at.markdown)


def test_lencana_backend_cli_tidak_menyebut_data_tiruan(monkeypatch):
    at = jalankan_app(monkeypatch, "cli")
    texts = [m.value for m in at.markdown]
    assert any("core C++" in t for t in texts)
    assert not any("DATA TIRUAN" in t for t in texts)


def test_lencana_mock_dan_cli_berbeda():
    assert "DATA TIRUAN" in backend_badge("mock")
    assert "DATA TIRUAN" not in backend_badge("cli")
