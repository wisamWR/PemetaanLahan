"""API FastAPI untuk UI web (Fase 1: mock dan cli, halaman Kinerja).

Lapisan tipis di atas `services/`, `landcover/`, dan `launchers/`; tidak mengimpor Streamlit. Semua
kalimat vonis, pemformatan angka, dan aturan perbandingan datang dari `services/` (satu sumber dengan
aplikasi Streamlit), sehingga frontend hanya menampilkan.

Jalankan (dari root repo):  .venv\\Scripts\\python.exe -m uvicorn api.main:app --app-dir app --port 8000
"""
