"""Bentuk JSON untuk frontend. Semua kalimat dan angka tampilan datang dari `services/`.

Aturan yang dijaga di sini (sama dengan aplikasi Streamlit; frontend tidak menghitung ulang):
  * total resmi dari `timing.json` (`run.total`), bukan jumlah tahap;
  * pangsa tahap dihitung dari JUMLAH waktu tahap (persen tampilan kedua grafik);
  * vonis verifikasi berupa kalimat + tingkat (`ok` hanya untuk resolusi penuh, `partial` untuk pratinjau);
  * hasil tiruan membawa `mock: true` dan kalimat DATA TIRUAN;
  * NaN/tak hingga tidak pernah keluar (JSON tidak punya NaN): jadi `null`.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Mapping

from landcover.contract import STAGES, RunResult, validate
from landcover.labeling import CLASS_PRESETS, NODATA_LABEL, TOTAL_LABEL, UNNAMED, ClassMap, area_table, cluster_table
from landcover.reference_kmeans import KMEANS_MAX_K, MAX_SEED
from landcover.verify import Verification
from launchers.base import KMEANS_CHOICES, MODES
from services import verification as sv
from services.labeling_panel import (
    HELP_CLASS,
    HELP_OTHER,
    HELP_PANEL,
    NAME_MAX_CHARS,
    area_caption,
    centroid_hex,
    cluster_error_ids,
    export_items,
    export_note,
    format_ha,
    format_percent,
    names_of,
    nodata_note,
    nodata_percent,
    palette_hex,
    pixel_size_of,
    preview_caption,
    run_fingerprint,
    split_name,
    summary_of,
)
from services.charts import (
    DEVICE_COLORS,
    DEVICE_NAMES,
    DEVICE_ORDER,
    STAGE_NAMES,
    TOTAL_NAME,
    UNKNOWN_DEVICE_COLOR,
    clean_seconds,
    clean_seconds_or_none,
    device_color,
    device_name,
    format_duration,
    run_cells,
    run_label,
    run_labels,
    share_text,
    stage_records,
)
from services.compare import EMPTY, MOCK_NOTICE, device_caption, format_seconds, speedup
from services.params import (
    ITERS_DEFAULT,
    ITERS_MAX,
    K_DEFAULT,
    K_MIN,
    SEED_DEFAULT,
    THREADS_DEFAULT,
    THREADS_MAX,
    THREADS_MIN,
)
from services.runs import KMEANS_LABELS, MODE_LABELS, created_text, run_id_of, run_stamp


def num(value: Any) -> float | None:
    """Float yang aman untuk JSON; None untuk NaN, tak hingga, atau bukan angka."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def iso(ts: float | None) -> str | None:
    if ts is None or not math.isfinite(ts) or ts <= 0:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


# --- run ------------------------------------------------------------------------- #

def stage_cell(stage: str, cell) -> dict:
    detail = []
    for key in ("h2d", "kernel", "d2h"):
        if cell.detail and key in cell.detail:
            seconds = clean_seconds(cell.detail[key])
            detail.append({"key": key, "seconds": seconds, "text": format_duration(seconds)})
    return {
        "stage": stage,
        "name": STAGE_NAMES[stage],
        "device": cell.device,
        "device_name": device_name(cell.device),
        "color": device_color(cell.device),
        "seconds": cell.seconds,
        "duration_text": format_duration(cell.seconds),
        "share": cell.share,
        "share_text": share_text(cell.share),
        "detail": detail,
    }


def run_summary(run: RunResult, created_ts: float | None = None, note: str | None = None) -> dict:
    m = run.meta
    total = clean_seconds_or_none(run.total)
    return {
        "run_id": run_id_of(run),
        "dataset": m.get("dataset"),
        "mode": run.mode,
        "mode_label": MODE_LABELS.get(run.mode, str(run.mode)),
        "kmeans_device": run.kmeans_device,
        "engine_label": run_label(run),
        "caption": device_caption(run),
        "k": run.k,
        "iters": m.get("iters"),
        "seed": m.get("seed"),
        "threads": m.get("threads"),
        "width": run.width,
        "height": run.height,
        "n_pixels": run.n_pixels,
        "total_s": total,
        "total_text": format_seconds(total),
        "created_at": iso(created_ts),
        "created_text": created_text(m, run.path),
        "mock": run.mock,
        "has_labels": (run.path / "labels.u8").is_file(),
        "problems": validate(run),
        "note": note,
    }


def run_detail(run: RunResult, created_ts: float | None = None, note: str | None = None) -> dict:
    cells = run_cells(run)
    return {
        **run_summary(run, created_ts, note),
        "stages": [stage_cell(s, cells[s]) for s in STAGES],
        "sum_s": sum(c.seconds for c in cells.values()),
        "counts": [int(c) for c in run.counts],
        "centroids": [[int(v) for v in row] for row in run.centroids],
        "meta": dict(run.meta),
        "mock_notice": MOCK_NOTICE if run.mock else None,
    }


# --- perbandingan ------------------------------------------------------------------ #

def chart_payload(cpu: RunResult | None, opt: RunResult | None) -> dict | None:
    """Data grafik waktu per tahap (CPU saja di atas, lalu Optimalisasi); None bila tidak ada run."""
    runs, labels = run_labels(cpu, opt)
    if not runs:
        return None
    rows = []
    for run, label in zip(runs, labels):
        cells = run_cells(run)
        total = clean_seconds_or_none(run.total)
        rows.append({
            "label": label,
            "run_id": run_id_of(run),
            "mode": run.mode,
            "total_s": total,
            "total_text": None if total is None else format_duration(total),
            "sum_s": sum(c.seconds for c in cells.values()),
            "sum_text": format_duration(sum(c.seconds for c in cells.values())),
            "stages": [stage_cell(s, cells[s]) for s in STAGES],
        })
    present = {c["device"] for row in rows for c in row["stages"]}
    ordered = [d for d in DEVICE_ORDER if d in present] + sorted(present - set(DEVICE_ORDER))
    longest = max([*(r["sum_s"] for r in rows), *(r["total_s"] for r in rows if r["total_s"] is not None)], default=0.0)
    table = []
    for rec in stage_records(runs, labels):
        is_total = rec["Tahap"] == TOTAL_NAME
        table.append({
            "run": rec["Run"],
            "stage": rec["Tahap"],
            "device": rec["Perangkat"] or None,
            "seconds": num(rec["Waktu (s)"]),
            "percent": num(rec["Persen dari jumlah tahap"]),
            "is_total": is_total,
        })
    stage_speedup: dict[str, float | None] | None = None
    stage_speedup_text: dict[str, str] | None = None
    if cpu is not None and opt is not None:
        cpu_cells, opt_cells = run_cells(cpu), run_cells(opt)
        # sama dengan kartu speedup: T_cpu / T_opt, kosong bila salah satu nol (tidak dibagi nol)
        stage_speedup = {s: speedup(cpu_cells[s].seconds or None, opt_cells[s].seconds or None) for s in STAGES}
        stage_speedup_text = {s: EMPTY if v is None else f"{v:.2f}×" for s, v in stage_speedup.items()}
    return {
        "labels": labels,
        "rows": rows,
        "stage_speedup": stage_speedup,
        "stage_speedup_text": stage_speedup_text,
        "stages": [{"stage": s, "name": STAGE_NAMES[s]} for s in STAGES],
        "devices": [{"device": d, "name": device_name(d), "color": device_color(d)} for d in ordered],
        "max_seconds": longest,
        "total_name": TOTAL_NAME,
        "table": table,
        "mock": any(r.mock for r in runs),
    }


def compare_payload(
    cpu: RunResult | None,
    opt: RunResult | None,
    created: Mapping[str, float],
    notes: Mapping[str, str],
) -> dict:
    s = speedup(cpu.total if cpu else None, opt.total if opt else None)
    anchor = cpu or opt

    def side(run: RunResult | None) -> dict | None:
        if run is None:
            return None
        rid = run_id_of(run)
        return run_summary(run, created.get(rid), notes.get(rid))

    mock = any(r.mock for r in (cpu, opt) if r is not None)
    return {
        "empty": anchor is None,
        "key": None if anchor is None else {
            "dataset": anchor.meta.get("dataset"), "k": anchor.k,
            "iters": anchor.meta.get("iters"), "seed": anchor.meta.get("seed"),
        },
        "cpu": side(cpu),
        "opt": side(opt),
        "speedup": None if s is None else s,
        "speedup_text": EMPTY if s is None else f"{s:.2f}×",
        "mock": mock,
        "mock_notice": MOCK_NOTICE if mock else None,
        "chart": chart_payload(cpu, opt),
    }


def verification_payload(v: Verification, cpu: RunResult, opt: RunResult) -> dict:
    level = sv.verdict_level(v)
    pct = num(v.identical_pct)
    return {
        "title": sv.TITLE,
        "status": v.status,
        "level": level,
        "icon": sv.LEVEL_ICONS[level],
        "verdict": sv.verdict_text(v),
        "source": v.source,
        "source_text": sv.source_text(v),
        "full": v.full,
        "n_compared": v.n_compared,
        "n_diff": v.n_diff,
        "identical_pct": pct,
        "identical_pct_text": None if pct is None else sv.format_identical_pct(pct),
        "centroids_equal": v.centroids_equal,
        "counts_equal": v.counts_equal,
        "reasons": list(v.reasons),
        "differences": list(v.differences),
        "notes": list(v.notes),
        "rows": [{"what": what, "result": result} for what, result in sv.detail_rows(v)],
        "pair_label": f"{run_label(cpu)} ↔ {run_label(opt)}",
        "mock": v.mock,
        "mock_notice": sv.MOCK_VERIFY_NOTICE if v.mock else None,
    }


def verification_error_payload(message: str) -> dict:
    """Verifikasi gagal dijalankan (OSError/ValueError): tampil sebagai peringatan, bukan galat halaman."""
    return {
        "title": sv.TITLE, "status": "error", "level": sv.LEVEL_WARN, "icon": sv.ICON_WARN,
        "verdict": f"Verifikasi tidak bisa dijalankan: {message}", "source": None, "source_text": None,
        "full": False, "n_compared": None, "n_diff": None, "identical_pct": None,
        "identical_pct_text": None, "centroids_equal": None, "counts_equal": None,
        "reasons": [], "differences": [], "notes": [], "rows": [], "pair_label": None,
        "mock": False, "mock_notice": None,
    }


# --- pemetaan: luas dan pelabelan ------------------------------------------------------ #

def area_payload(run: RunResult, class_map: ClassMap, errors: list[str], created_ts: float | None = None) -> dict:
    """Tabel luas, tabel cluster, ringkasan, palet, dan daftar ekspor untuk satu `class_map`.

    Semua angka dan kalimat berasal dari `landcover.labeling` dan `services.labeling_panel` (persen dari
    `counts` resolusi penuh, bukan dari pratinjau). `ValueError` bila hasil janggal (centroid atau
    `counts` tidak cocok); pemanggil memetakannya ke 422."""
    size, size_note = pixel_size_of(run)
    k = run.k
    nd_pixels, nd_cluster = run.nodata_pixels, run.nodata_cluster
    area = area_table(run.counts, class_map, size, nodata_pixels=nd_pixels, nodata_cluster=nd_cluster)
    clusters = cluster_table(run.counts, run.centroids, size, nodata_pixels=nd_pixels, nodata_cluster=nd_cluster)
    summary = summary_of(area)
    names = names_of(class_map, k)
    class_hex = palette_hex(run, class_map)
    centroid = centroid_hex(run)
    invalid = cluster_error_ids(errors)
    pure_nodata = nd_cluster if (nd_pixels > 0 and int(run.counts[nd_cluster]) == nd_pixels) else None

    cluster_rows = []
    for idx in range(k):
        row = clusters.iloc[idx]
        kelas, other = split_name(names[idx])
        is_nd_cluster = nd_pixels > 0 and idx == nd_cluster
        all_nodata = idx == pure_nodata
        nodata_badge = (
            "Seluruhnya nodata"
            if all_nodata
            else (f"Memuat {nd_pixels:,} piksel nodata" if is_nd_cluster else None)
        )
        cluster_rows.append({
            "cluster": idx,
            "centroid_hex": centroid[idx],
            "class_hex": class_hex[idx],
            "name": names[idx],
            "named": names[idx] != UNNAMED,
            "kelas": kelas,
            "nama_lain": other,
            "pixels": int(row["piksel"]),
            "pixels_text": f"{int(row['piksel']):,}",
            "percent": float(row["persen"]),
            "percent_text": format_percent(float(row["persen"])),
            "area_ha": float(row["luas_ha"]),
            "area_text": format_ha(float(row["luas_ha"])),
            "invalid": idx in invalid,
            "all_nodata": all_nodata,
            "nodata_pixels": nd_pixels if is_nd_cluster else 0,
            "nodata_badge": nodata_badge,
        })

    members: dict[str, list[int]] = {}
    for idx, name in enumerate(names):
        if idx == pure_nodata:
            continue
        members.setdefault(name, []).append(idx)
    area_rows = []
    for _, row in area.iterrows():
        name = str(row["kelas"])
        is_total = name == TOTAL_LABEL
        is_nodata = name == NODATA_LABEL
        ids = [] if is_total else ([nd_cluster] if is_nodata else members.get(name, []))
        if is_total or is_nodata:
            colors: list[str] = []
        elif name == UNNAMED:                 # beberapa cluster tanpa nama: tiap cluster memakai warna centroid
            colors = [centroid[i] for i in ids]
        else:
            colors = [class_hex[ids[0]]]      # cluster ber-Kelas sama berwarna sama di peta
        area_rows.append({
            "name": name,
            "clusters": str(row["cluster"]),
            "cluster_ids": ids,
            "colors": colors,
            "unnamed": name == UNNAMED,
            "is_total": is_total,
            "is_nodata": is_nodata,
            "pixels": int(row["piksel"]),
            "pixels_text": f"{int(row['piksel']):,}",
            "percent": float(row["persen"]),
            "percent_text": format_percent(float(row["persen"])),
            "area_ha": float(row["luas_ha"]),
            "area_text": format_ha(float(row["luas_ha"])),
        })

    total_all = int(run.counts.sum())
    valid_total = total_all - nd_pixels
    total_caption = (
        f"{valid_total:,} piksel valid ({nd_pixels:,} nodata dikecualikan) · {size:g} m per piksel"
        if nd_pixels > 0
        else f"{total_all:,} piksel · {size:g} m per piksel"
    )
    named_caption = (
        f"dari {k - 1} cluster valid (1 nodata)"
        if pure_nodata is not None
        else f"dari {k} cluster"
    )
    kpis = [
        {"key": "total", "label": "Luas total", "value": format_ha(summary.total_ha),
         "caption": total_caption},
        {"key": "named", "label": "Kelas bernama", "value": str(summary.named_classes),
         "caption": named_caption},
        {"key": "largest", "label": "Kelas terbesar",
         "value": "Belum ada" if summary.largest is None else f"{summary.largest[0]} ({summary.largest[1]:.1f} %)",
         "caption": "Beri nama cluster untuk melihatnya." if summary.largest is None else "Dari kelas yang sudah diberi nama."},
        {"key": "unnamed", "label": "Belum diberi nama", "value": format_percent(summary.unnamed_percent),
         "caption": "Piksel pada cluster tanpa nama."},
    ]
    preview_h, preview_w = run.labels_preview.shape[:2]
    return {
        "run_id": run_id_of(run),
        "mock": run.mock,
        "mock_notice": MOCK_NOTICE if run.mock else None,
        "k": k,
        "fingerprint": run_fingerprint(run),
        "stamp": str(run_stamp(run.path)),
        "created_at": iso(created_ts),
        "preview": {"width": int(preview_w), "height": int(preview_h)},
        "pixel_size_m": size,
        "pixel_size_note": size_note,
        "presets": list(CLASS_PRESETS),
        "unnamed_label": UNNAMED,
        "total_label": TOTAL_LABEL,
        "nodata_label": NODATA_LABEL,
        "nodata": {
            "pixels": nd_pixels,
            "cluster": nd_cluster,
            "percent": num(nodata_percent(nd_pixels, total_all)),
            "note": nodata_note(nd_pixels, total_all),
        },
        "name_max_chars": NAME_MAX_CHARS,
        "names": names,
        "palette": class_hex,
        "clusters": cluster_rows,
        "area": area_rows,
        "kpis": kpis,
        "errors": list(errors),
        "help": {"panel": HELP_PANEL, "other": HELP_OTHER, "kelas": HELP_CLASS},
        "preview_caption": preview_caption(run),
        "area_caption": area_caption(size, size_note, has_nodata=nd_pixels > 0),
        "exports": export_items(run),
        "export_note": export_note(run),
    }


# --- konfigurasi -------------------------------------------------------------------- #

def core_payload(probe) -> dict | None:
    """Hasil `landcover --info` (backend cli); None untuk backend lain."""
    if probe is None:
        return None
    info = probe.info
    return {
        "ok": info is not None,
        "problem": probe.problem,
        "info": None if info is None else {
            "cpu": info.cpu, "omp": info.omp, "cuda": info.cuda, "openmp": info.openmp,
            "max_threads": info.max_threads,
            "auto_cuda_threshold_pixels": info.auto_cuda_threshold_pixels,
        },
        "notice": None if info is None else info.notice,
        "kmeans_choices": list(KMEANS_CHOICES if info is None else info.kmeans_choices),
    }


def config_payload(backend: str, probe, active_job: str | None) -> dict:
    mock = backend == "mock"
    return {
        "backend": backend,
        "mock": mock,
        "backend_label": "Backend: DATA TIRUAN (mock)" if mock else "Backend: core C++ (cli)",
        "mock_notice": MOCK_NOTICE if mock else None,
        "core": core_payload(probe),
        "defaults": {
            "k": K_DEFAULT, "iters": ITERS_DEFAULT, "seed": SEED_DEFAULT, "threads": THREADS_DEFAULT,
            "mode": "cpu", "kmeans": "auto",
        },
        "limits": {
            "k_min": K_MIN, "k_max": KMEANS_MAX_K, "iters_max": ITERS_MAX, "threads_min": THREADS_MIN,
            "threads_max": THREADS_MAX, "seed_max": MAX_SEED,
        },
        "modes": list(MODES),
        "labels": {
            "modes": dict(MODE_LABELS),
            "kmeans": dict(KMEANS_LABELS),
            "devices": dict(DEVICE_NAMES),
            "stages": dict(STAGE_NAMES),
        },
        "device_colors": dict(DEVICE_COLORS),
        "unknown_device_color": UNKNOWN_DEVICE_COLOR,
        "stage_order": list(STAGES),
        "active_job": active_job,
    }
