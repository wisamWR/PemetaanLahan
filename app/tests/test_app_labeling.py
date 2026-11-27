"""Alur layar panel pelabelan dan tabel luas dengan AppTest (plan Langkah 8, "Cek" dan "Selesai bila").

AppTest tidak punya `set_value` untuk `st.data_editor`; suntingan disimulasikan dengan mengisi state
editor (`edited_rows`, kumulatif seperti yang dikirim frontend) lewat `at.session_state` sebelum
`run()`. Klik sel sungguhan diuji di browser (lihat catatan Sesi H di docs/PROGRESS.md).
Pembantu dan fixture `sandbox` dipakai bersama `test_app_flow.py`.
"""

from __future__ import annotations

import json

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

from landcover.labeling import UNNAMED
from test_app_flow import (  # noqa: F401  (sandbox = fixture pytest)
    CPU_ID, OPT_ID, captions, metrics, no_errors, run_mode, sandbox, start,
)


def editor_key_of(rid: str) -> str:
    return f"label_editor__{rid}"


def seed_edits(at: AppTest, rid: str, edited_rows: dict) -> AppTest:
    """Meniru frontend: kirim kumpulan suntingan KUMULATIF editor lalu jalankan ulang skrip."""
    at.session_state[editor_key_of(rid)] = {"edited_rows": edited_rows, "added_rows": [], "deleted_rows": []}
    at.run()
    assert not at.exception, at.exception
    return at


def image_urls(at: AppTest) -> list[str]:
    return [i.proto.imgs[0].url for i in at.get("image")]          # URL = hash isi: isi sama, URL sama


def area_frame(at: AppTest):
    (f,) = [f for f in at.dataframe if "kelas" in f.value.columns]
    return f.value


def editor_frame_of(at: AppTest):
    (ed,) = at.data_editor
    return ed.value


def counts_of(runs, rid: str) -> list[int]:
    return json.loads((runs / "mock" / rid / "counts.json").read_text(encoding="utf-8"))


def download_labels(at: AppTest) -> list[str]:
    return [d.proto.label for d in at.get("download_button")]


def class_map_of(at: AppTest, rid: str = CPU_ID) -> dict:
    return dict(at.session_state["class_maps"][rid])


# --- keadaan awal ---------------------------------------------------------------------- #

def test_kotak_kosong_sebelum_ada_hasil(sandbox):
    at = start()
    assert any("Pelabelan cluster dan tabel luas: belum ada hasil" in c for c in captions(at))
    assert not at.data_editor and not at.metric and download_labels(at) == []


def test_panel_tampil_setelah_run_dengan_semua_cluster_tanpa_nama(sandbox):
    data, runs = sandbox
    at = start()
    run_mode(at, "cpu")
    no_errors(at)
    assert [ed.key for ed in at.data_editor] == [editor_key_of(CPU_ID)]
    assert at.data_editor[0].editing_mode == "fixed"
    frame = editor_frame_of(at)
    assert list(frame["Cluster"]) == list(range(8)) and list(frame["Piksel"]) == counts_of(runs, CPU_ID)
    assert frame["Kelas"].isna().all() and (frame["Nama lain"] == "").all()
    assert class_map_of(at) == {i: UNNAMED for i in range(8)}              # render pertama tidak mengubahnya
    m = metrics(at)
    assert (m["Luas total"], m["Kelas bernama"], m["Kelas terbesar"], m["Belum diberi nama"]) == (
        "12.00 ha", "0", "Belum ada", "100.00 %")                          # 40x30 piksel @10 m = 12 ha
    area = area_frame(at)
    assert area["kelas"].tolist() == [UNNAMED, "Total"] and area["piksel"].tolist() == [1200, 1200]
    assert any(s.value.startswith("Pelabelan cluster dan luas") and "DATA TIRUAN" in s.value for s in at.subheader)
    assert download_labels(at) == ["Peta kelas (PNG pratinjau)", "Tabel luas per kelas (CSV)", "Tabel per cluster (CSV)"]
    assert any("PNG resolusi penuh tidak tersedia" in c for c in captions(at))
    assert not any(b.key == "btn_full_export" for b in at.button)
    assert len(at.get("image")) == 2                                       # swatch ada di dalam tabel, bukan st.image
    assert not any("Pelabelan cluster dan tabel luas: belum ada hasil" in c for c in captions(at))


# --- "Cek" plan: gabungkan dua cluster ke Air -------------------------------------------- #

def test_gabungkan_dua_cluster_ke_air_tabel_satu_baris_dan_peta_berubah(sandbox):
    data, runs = sandbox
    at = start()
    run_mode(at, "cpu")
    counts = counts_of(runs, CPU_ID)
    before = image_urls(at)
    assert at.radio(key="map_view").value == "centroid"

    seed_edits(at, CPU_ID, {0: {"Kelas": "Air"}, 1: {"Kelas": "Air"}})
    no_errors(at)
    assert class_map_of(at) == {0: "Air", 1: "Air", **{i: UNNAMED for i in range(2, 8)}}
    area = area_frame(at)
    air = area[area["kelas"] == "Air"]
    assert len(air) == 1 and air["cluster"].item() == "0, 1" and air["piksel"].item() == counts[0] + counts[1]
    assert area["kelas"].tolist() == ["Air", UNNAMED, "Total"] and area["piksel"].iloc[-1] == 1200
    m = metrics(at)
    assert m["Kelas bernama"] == "1" and m["Kelas terbesar"].startswith("Air (")
    assert at.radio(key="map_view").value == "class"                       # peta pindah ke warna kelas
    after = image_urls(at)
    assert after[0] == before[0] and after[1] != before[1]                 # citra asli tetap, peta berubah
    assert len(at.get("image")) == 2 and list(at.session_state["runs"]) == [CPU_ID]    # tidak ada run baru


def test_peta_memakai_nama_terbaru_pada_eksekusi_yang_sama(sandbox, monkeypatch):
    """Panel digambar setelah peta di skrip; peta harus tetap melihat nama dari suntingan yang baru
    dikirim (bukan nama sebelumnya, yang baru terlihat satu eksekusi kemudian)."""
    import ui.maps as maps

    seen: list[dict] = []
    real = maps.render_maps

    def spy(run, class_map, data_dir):
        seen.append(dict(class_map))
        return real(run, class_map, data_dir)

    at = start()
    monkeypatch.setattr(maps, "render_maps", spy)        # skrip mengambil ulang nama ini tiap eksekusi
    run_mode(at, "cpu")
    seen.clear()
    seed_edits(at, CPU_ID, {2: {"Nama lain": "Sawah"}})
    assert seen and seen[-1][2] == "Sawah"
    urls_1 = image_urls(at)
    seed_edits(at, CPU_ID, {2: {"Nama lain": "Sawah"}, 3: {"Kelas": "Vegetasi"}})
    assert seen[-1][2] == "Sawah" and seen[-1][3] == "Vegetasi"
    urls_2 = image_urls(at)
    assert urls_2[1] != urls_1[1]                                           # peta mengikuti suntingan kedua
    at.run()                                                                # eksekusi tanpa suntingan baru
    assert image_urls(at) == urls_2                                         # peta tidak "menyusul" lagi


def test_perubahan_nama_menggeser_metrik_ringkasan(sandbox):
    at = start()
    run_mode(at, "cpu")
    seed_edits(at, CPU_ID, {i: {"Kelas": "Vegetasi"} for i in range(8)})
    m = metrics(at)
    assert m["Kelas bernama"] == "1" and m["Kelas terbesar"] == "Vegetasi (100.0 %)" and m["Belum diberi nama"] == "0.00 %"
    assert area_frame(at)["kelas"].tolist() == ["Vegetasi", "Total"]


def test_huruf_besar_kecil_dibedakan_dan_cluster_gabungan_tampil_dalam_satu_baris(sandbox):
    at = start()
    run_mode(at, "cpu")
    edits = {i: {"Kelas": "Air"} for i in (0, 3, 6)} | {1: {"Nama lain": "air"}}
    seed_edits(at, CPU_ID, edits)
    area = area_frame(at)
    assert area["kelas"].tolist() == ["Air", "air", UNNAMED, "Total"]
    assert area.loc[0, "cluster"] == "0, 3, 6" and area.loc[1, "cluster"] == "1"


# --- suntingan kumulatif dan state ---------------------------------------------------------- #

def test_suntingan_lama_yang_masih_tersimpan_tidak_mengubah_nama_diam_diam(sandbox):
    """Frontend mengirim suntingan KUMULATIF dan isi awal editor dibangun ulang dari peta kelas;
    suntingan lama yang masih tersimpan harus tetap menghasilkan nama yang sama (ada tesnya
    supaya aturan 'Nama lain mengalahkan Kelas' tidak diganti dengan 'yang terakhir menang')."""
    at = start()
    run_mode(at, "cpu")

    seed_edits(at, CPU_ID, {0: {"Kelas": "Air"}})
    assert class_map_of(at)[0] == "Air"
    seed_edits(at, CPU_ID, {0: {"Kelas": "Air", "Nama lain": "Sawah"}})
    assert class_map_of(at)[0] == "Sawah"
    frame = editor_frame_of(at)                                             # dasar eksekusi ini: dibangun SEBELUM suntingan
    assert frame.loc[0, "Kelas"] == "Air" and frame.loc[0, "Nama lain"] == ""      # diterapkan (nama lama: Air)
    at.run()                                                                # eksekusi berikutnya: dasar dari peta kelas
    frame = editor_frame_of(at)                                             # (kosong, "Sawah"); suntingan lama masih menempel
    assert frame.loc[0, "Nama lain"] == "Sawah" and frame["Kelas"].isna().all()
    assert class_map_of(at)[0] == "Sawah"
    seed_edits(at, CPU_ID, {0: {"Kelas": "Air", "Nama lain": "Sawah"}, 1: {"Kelas": "Vegetasi"}})
    assert class_map_of(at)[0] == "Sawah" and class_map_of(at)[1] == "Vegetasi"   # baris 1 tidak mengubah baris 0
    seed_edits(at, CPU_ID, {0: {"Kelas": "Air", "Nama lain": ""}, 1: {"Kelas": "Vegetasi"}})
    assert class_map_of(at)[0] == "Air"                                     # Nama lain dikosongkan: Kelas dipakai
    for _ in range(2):                                                      # rerun tanpa suntingan baru: tetap sama
        at.run()
        assert not at.exception and class_map_of(at)[0] == "Air" and class_map_of(at)[1] == "Vegetasi"


def test_tampilan_peta_tidak_dipaksa_ke_warna_kelas_pada_eksekusi_tanpa_suntingan(sandbox):
    at = start()
    run_mode(at, "cpu")
    seed_edits(at, CPU_ID, {0: {"Kelas": "Air"}})
    assert at.radio(key="map_view").value == "class"
    at.radio(key="map_view").set_value("centroid").run()
    assert not at.exception and at.radio(key="map_view").value == "centroid"
    at.run()
    assert at.radio(key="map_view").value == "centroid"                     # pengguna boleh kembali ke warna centroid
    seed_edits(at, CPU_ID, {0: {"Kelas": "Air"}, 1: {"Kelas": "Lainnya"}})  # suntingan baru: pindah lagi
    assert at.radio(key="map_view").value == "class"


def test_nama_total_tampil_sebagai_galat_dan_nama_lama_dipertahankan(sandbox):
    at = start()
    run_mode(at, "cpu")
    seed_edits(at, CPU_ID, {0: {"Nama lain": "Vegetasi"}})
    seed_edits(at, CPU_ID, {0: {"Nama lain": "Total"}, 1: {"Kelas": "Air"}})
    assert class_map_of(at)[0] == "Vegetasi"                                # ditolak, nama lama tetap
    assert class_map_of(at)[1] == "Air"                                     # suntingan lain tetap berlaku
    assert len(at.error) == 1 and "Cluster 0" in at.error[0].value and "Total" in at.error[0].value
    assert area_frame(at)["kelas"].tolist() == ["Vegetasi", "Air", UNNAMED, "Total"]
    at.run()                                                                # galat tetap tampil sampai diperbaiki
    assert len(at.error) == 1
    seed_edits(at, CPU_ID, {0: {"Nama lain": "Sawah"}, 1: {"Kelas": "Air"}})
    no_errors(at)
    assert class_map_of(at)[0] == "Sawah"


def test_run_lain_dengan_k_sama_tidak_berbagi_nama_dan_nama_kembali_saat_run_dipilih_lagi(sandbox):
    at = start()
    run_mode(at, "cpu")
    seed_edits(at, CPU_ID, {0: {"Kelas": "Air"}, 5: {"Nama lain": "Sawah"}})
    assert at.data_editor[0].key == editor_key_of(CPU_ID)

    run_mode(at, "opt")                                                     # K=8 yang sama, run lain
    no_errors(at)
    assert at.session_state["active_run"] == OPT_ID
    assert at.data_editor[0].key == editor_key_of(OPT_ID)                   # kunci editor memuat run_id
    assert class_map_of(at, OPT_ID) == {i: UNNAMED for i in range(8)}
    assert area_frame(at)["kelas"].tolist() == [UNNAMED, "Total"]
    assert editor_frame_of(at)["Kelas"].isna().all() and (editor_frame_of(at)["Nama lain"] == "").all()

    at.selectbox(key="active_run_pick").select_index(1).run()               # kembali ke run CPU
    assert not at.exception and at.session_state["active_run"] == CPU_ID
    assert area_frame(at)["kelas"].tolist() == ["Air", UNNAMED, "Sawah", "Total"]
    frame = editor_frame_of(at)
    assert frame.loc[0, "Kelas"] == "Air" and frame.loc[5, "Nama lain"] == "Sawah"
    assert class_map_of(at)[0] == "Air"


# --- ekspor resolusi penuh -------------------------------------------------------------------- #

def test_ekspor_resolusi_penuh_lewat_tombol_hanya_bila_labels_u8_ada(sandbox):
    data, runs = sandbox
    at = start()
    run_mode(at, "cpu")
    assert not any(b.key == "btn_full_export" for b in at.button)
    labels = (np.arange(40 * 30) % 8).astype(np.uint8)
    (runs / "mock" / CPU_ID / "labels.u8").write_bytes(labels.tobytes())
    seed_edits(at, CPU_ID, {0: {"Kelas": "Air"}})
    assert any(b.key == "btn_full_export" for b in at.button)
    assert not any("PNG resolusi penuh tidak tersedia" in c for c in captions(at))

    at.button(key="btn_full_export").click().run()
    no_errors(at)
    out = runs / "mock" / CPU_ID / f"TIRUAN_peta-kelas-penuh_{CPU_ID}.png"
    assert out.is_file()
    assert any(s.value.startswith("Disimpan di ") and str(out) in s.value for s in at.success)
    assert "Unduh PNG resolusi penuh" in download_labels(at)
    with Image.open(out) as png:
        assert png.size == (40, 30) and png.mode == "P"                      # lebar 40, tinggi 30 (dari meta)
        assert np.array_equal(np.array(png).ravel(), labels)
        assert tuple(png.getpalette()[:3]) == (30, 100, 200)                 # cluster 0 diberi nama Air


def test_ekspor_resolusi_penuh_ukuran_salah_tampil_sebagai_galat(sandbox):
    data, runs = sandbox
    at = start()
    run_mode(at, "cpu")
    (runs / "mock" / CPU_ID / "labels.u8").write_bytes(bytes(100))           # seharusnya 1200 byte
    at.run()
    at.button(key="btn_full_export").click().run()
    assert not at.exception
    assert len(at.error) == 1 and "gagal dibuat" in at.error[0].value and "1200" in at.error[0].value
    assert not list((runs / "mock" / CPU_ID).glob("*.png.tmp"))


# --- hasil janggal dan hasil asli --------------------------------------------------------------- #

def test_hasil_janggal_counts_tidak_cocok_panel_menampilkan_galat_bukan_mogok(sandbox):
    data, runs = sandbox
    at = start()
    run_mode(at, "cpu")
    counts = runs / "mock" / CPU_ID / "counts.json"
    counts.write_text(json.dumps(counts_of(runs, CPU_ID)[:5]), encoding="utf-8")      # K=8 tetapi 5 hitungan
    (runs / "mock" / CPU_ID / "meta.json").touch()                                     # cache dibaca ulang
    at.run()
    assert not at.exception
    assert any("janggal" in w.value for w in at.warning)
    assert len(at.error) == 1 and "Panel pelabelan tidak bisa dibuat" in at.error[0].value
    assert not at.data_editor and len(at.get("image")) == 2                            # peta dan kartu tetap tampil
    assert metrics(at)["Total: CPU saja (L0)"] != "—"


def test_centroid_di_luar_rentang_tidak_membuat_peta_atau_unduhan_mogok(sandbox):
    data, runs = sandbox
    at = start()
    run_mode(at, "cpu")
    cents = runs / "mock" / CPU_ID / "centroids.json"
    values = json.loads(cents.read_text(encoding="utf-8"))
    values[0][0] = 300                                                                 # di luar 0..255
    cents.write_text(json.dumps(values), encoding="utf-8")
    (runs / "mock" / CPU_ID / "meta.json").touch()
    at.run()
    assert not at.exception
    seed_edits(at, CPU_ID, {0: {"Kelas": "Air"}})                                      # memicu tampilan warna kelas
    assert at.radio(key="map_view").value == "class"
    assert any("Peta warna kelas tidak bisa dibuat" in w.value for w in at.warning)
    assert any("Peta kelas (PNG) tidak bisa dibuat" in c for c in captions(at))
    assert "Peta kelas (PNG pratinjau)" not in download_labels(at)
    assert "Tabel luas per kelas (CSV)" in download_labels(at)
    assert len(at.get("image")) == 2


def test_hasil_bukan_tiruan_tanpa_lencana_dan_awalan_tiruan(sandbox, make_run_dir):
    data, runs = sandbox
    path = make_run_dir("asli", width=40, height=30, k=8, meta={"mock": False, "dataset": "kecil"})
    at = start()
    at.session_state["runs"] = {"asli": str(path)}
    at.run()
    assert not at.exception
    assert at.session_state["active_run"] == "asli" and len(at.data_editor) == 1
    assert any(s.value == "Pelabelan cluster dan luas" for s in at.subheader)
    assert not any("TIRUAN" in s.value for s in at.subheader)
    assert not any("TIRUAN" in c for c in captions(at))
    assert not any("TIRUAN" in w.value for w in at.warning)
    assert len(download_labels(at)) == 3
