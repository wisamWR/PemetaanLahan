"""Aplikasi FastAPI: rute `/api/*` dan penyajian frontend hasil build (`web/dist`).

`create_app(settings)` murni dari `Settings` (tes memberi folder sementara); `api.main.app` memakai
`config`. Semua rute `def` biasa (dijalankan di thread pool, jadi `probe_core` dan pembacaan folder run
tidak memblokir loop kejadian) kecuali aliran SSE.

Bentuk galat seragam: `{"detail": "<teks>"}` (422 untuk isian yang ditolak, 404 untuk id yang tidak
ada, 409 bila ada run yang masih berjalan, disertai `job_id`).
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import tempfile
import threading
import time
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, StrictStr
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from landcover.datasets import DatasetExistsError, list_datasets
from landcover.export import png_bytes
from landcover.verify import Verification, verification_stamp, verify_pair
from launchers import make_launcher
from launchers.cli import CoreProbe, probe_core
from services.datasets import (
    CARD_THUMB_SIDE,
    MAX_UPLOAD_BYTES,
    PreviewUnavailableError,
    UploadTooLargeError,
    clamp_size,
    dataset_card,
    dataset_cards,
    dataset_preview_jpeg,
    store_upload,
    upload_limits,
)
from services.labeling_panel import (
    EXPORT_KINDS,
    EXPORT_MEDIA_TYPES,
    area_csv,
    class_map_from_names,
    class_map_from_rows,
    cluster_csv,
    export_full_map,
    export_items,
    export_names,
    labels_file_of,
)
from services.maps import (
    THUMB_MAX_SIDE,
    THUMB_MIN_SIDE,
    VIEW_CENTROID,
    VIEW_CLASS,
    class_map_image,
    jpeg_bytes,
    original_preview,
    thumbnail,
)
from services.params import (
    ITERS_DEFAULT,
    K_DEFAULT,
    SEED_DEFAULT,
    THREADS_DEFAULT,
    FormValues,
    build_pair,
    build_params,
)
from services.runs import MODE_LABELS, run_id_of, run_stamp

from api import library, views
from api.jobs import JobBusyError, JobManager, step_label
from api.settings import Settings

VERIFY_CACHE_SIZE = 32
TICK_EVERY_S = 0.25
POLL_EVERY_S = 0.1


class RunRequest(BaseModel):
    """Isian satu run. Tipe ketat (angka JSON harus integer); rentang diperiksa `RunParams`."""

    model_config = ConfigDict(extra="forbid")

    dataset: StrictStr
    mode: StrictStr = "cpu"
    kmeans: StrictStr = "auto"
    k: StrictInt = K_DEFAULT
    iters: StrictInt = ITERS_DEFAULT
    seed: StrictInt = SEED_DEFAULT
    threads: StrictInt = THREADS_DEFAULT
    force: StrictBool = False

    def values(self) -> FormValues:
        return FormValues(
            dataset=self.dataset, k=self.k, iters=self.iters, mode=self.mode, kmeans=self.kmeans,
            threads=self.threads, seed=self.seed, force=self.force,
        )


class CompareRequest(RunRequest):
    """Bandingkan: mode ditentukan server (CPU saja lalu Optimalisasi); `mode` diabaikan bila ada."""


class ClusterNameCell(BaseModel):
    """Dua kolom editor satu cluster (sama dengan editor Streamlit): Kelas (pilihan) dan Nama lain (bebas)."""

    model_config = ConfigDict(extra="forbid")

    kelas: StrictStr | None = None
    nama_lain: StrictStr | None = None


class AreaRequest(BaseModel):
    """`class_map`: id cluster (kunci JSON = teks angka) -> teks (dianggap "Nama lain") atau `{kelas, nama_lain}`.
    Aturan "Nama lain mengalahkan Kelas" diterapkan server (`services.labeling_panel`), bukan klien."""

    model_config = ConfigDict(extra="forbid")

    class_map: dict[str, StrictStr | ClusterNameCell] = {}


_CLUSTER_KEY = re.compile(r"[0-9]{1,3}")


def _cluster_rows(class_map: dict[str, Any]) -> dict[int, tuple]:
    rows: dict[int, tuple] = {}
    for key, value in class_map.items():
        if not _CLUSTER_KEY.fullmatch(key):
            raise HTTPException(422, f"class_map: id cluster harus bilangan bulat, ditemukan {key!r}")
        rows[int(key)] = (None, value) if isinstance(value, str) else (value.kelas, value.nama_lain)
    return rows


def _parse_names(raw: str | None) -> list | None:
    """`names` di URL peta/ekspor: daftar JSON nama akhir per cluster (hasil `POST .../area`)."""
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        raise HTTPException(422, "names harus daftar JSON berisi nama kelas per cluster") from None
    if not isinstance(value, list):
        raise HTTPException(422, "names harus daftar JSON berisi nama kelas per cluster")
    return value


def _content_disposition(filename: str) -> str:
    fallback = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename, safe='')}"


def _cached_bytes(
    request: Request, data: bytes, media_type: str, tag: str, filename: str | None = None
) -> Response:
    """Respons berisi bytes dengan ETag (304 bila klien sudah punya); `no-cache` = selalu tanya server."""
    etag = f'"{tag}"'
    headers = {"ETag": etag, "Cache-Control": "no-cache"}
    if filename is not None:
        headers["Content-Disposition"] = _content_disposition(filename)
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    return Response(data, media_type=media_type, headers=headers)


def _sse(event: str, data: Any, event_id: int | None = None) -> str:
    head = f"id: {event_id}\n" if event_id is not None else ""
    return f"{head}event: {event}\ndata: {json.dumps(data, ensure_ascii=False, allow_nan=False)}\n\n"


class _VerifyCache:
    """Cache LRU hasil `verify_pair`, berkunci (path, waktu ubah, ukuran) berkas yang dibaca: membaca
    dua `labels.u8` 120 MB di tiap permintaan akan lambat."""

    def __init__(self, size: int = VERIFY_CACHE_SIZE) -> None:
        self._size = size
        self._items: OrderedDict[tuple, Verification] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, cpu, opt) -> Verification:
        key = (verification_stamp(cpu), verification_stamp(opt))
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        result = verify_pair(cpu, opt)
        with self._lock:
            self._items[key] = result
            while len(self._items) > self._size:
                self._items.popitem(last=False)
        return result


def create_app(
    settings: Settings | None = None,
    *,
    launcher_factory: Callable[..., object] | None = None,
    probe: Callable[[Any], CoreProbe] | None = None,
) -> FastAPI:
    settings = settings or Settings.from_config()
    jobs = JobManager(settings, launcher_factory or make_launcher)
    verify_cache = _VerifyCache()
    probe_fn = probe or probe_core

    app = FastAPI(title="Pemetaan Tutupan Lahan API", docs_url="/api/docs", redoc_url=None,
                  openapi_url="/api/openapi.json")
    app.state.settings = settings
    app.state.jobs = jobs

    # --- galat seragam ------------------------------------------------------------- #

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        parts = []
        for err in exc.errors():
            where = ".".join(str(p) for p in err.get("loc", ()) if p not in ("body", "query"))
            parts.append(f"{where}: {err.get('msg')}" if where else str(err.get("msg")))
        return JSONResponse({"detail": "; ".join(parts) or "isian tidak valid"}, status_code=422)

    @app.exception_handler(JobBusyError)
    async def _busy(_: Request, exc: JobBusyError) -> JSONResponse:
        return JSONResponse({"detail": str(exc), "job_id": exc.job_id}, status_code=409)

    # --- pembantu --------------------------------------------------------------------- #

    def lib() -> library.Library:
        return library.scan_library(settings.runs_dir, settings.backend)

    def run_or_404(run_id: str):
        """(pustaka, run). `run_id` hanya dicocokkan dengan hasil pindai, tidak pernah membentuk path."""
        found = lib()
        run = found.get(run_id)
        if run is None:
            why = found.unreadable.get(run_id)
            raise HTTPException(404, f"run '{run_id}' tidak ada" if why is None else f"run '{run_id}' tidak terbaca: {why}")
        return found, run

    def core_probe() -> CoreProbe | None:
        return probe_fn(settings.core_bin) if settings.backend == "cli" else None

    def checked_values(req: RunRequest) -> FormValues:
        scan = list_datasets(settings.data_dir, settings.backend)
        if scan.get(req.dataset) is None:
            raise HTTPException(422, f"dataset '{req.dataset}' tidak ditemukan; pilih dari GET /api/datasets")
        return req.values()

    def check_engine(req_kmeans: str, modes: tuple[str, ...]) -> None:
        """Backend cli: tolak pilihan K-Means yang tidak terhubung pada build core ini (lebih jelas
        daripada kode keluar 3 saat run)."""
        if "opt" not in modes or settings.backend != "cli" or req_kmeans == "auto":
            return
        probe_result = core_probe()
        if probe_result is not None and probe_result.info is not None and req_kmeans not in probe_result.info.kmeans_choices:
            raise HTTPException(
                422, f"K-Means '{req_kmeans}' tidak terhubung pada build landcover ini "
                     f"(tersedia: {', '.join(probe_result.info.kmeans_choices)})",
            )

    def start_job(kind: str, steps: list[tuple[str, Any]], force: bool) -> JSONResponse:
        try:
            job = jobs.start(kind, steps, force)
        except ValueError as exc:                       # launcher tidak bisa dibuat
            raise HTTPException(422, str(exc)) from None
        return JSONResponse(
            {"job_id": job.id, "run_ids": [s.run_id for s in job.steps], "status": job.status},
            status_code=202,
        )

    # --- rute ---------------------------------------------------------------------------- #

    @app.get("/api/config")
    def get_config() -> dict:
        active = jobs.active()
        return views.config_payload(settings.backend, core_probe(), None if active is None else active.id)

    @app.get("/api/datasets")
    def get_datasets() -> dict:
        cards, skipped = dataset_cards(settings.data_dir, settings.backend)
        return {"datasets": cards, "skipped": list(skipped), "upload": upload_limits()}

    @app.post("/api/datasets", status_code=201)
    async def post_dataset(request: Request, filename: str = Query(..., min_length=1, max_length=255)) -> JSONResponse:
        """Unggah PNG: badan mentah (`Content-Type: image/png`) dan `?filename=`. Bukan multipart
        (python-multipart tidak termasuk stack); batas byte dicek sebelum dan selama pembacaan."""
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"berkas melebihi batas {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        chunks: list[bytes] = []
        received = 0
        async for chunk in request.stream():
            received += len(chunk)
            if received > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"berkas melebihi batas {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
            chunks.append(chunk)
        try:
            ds = await run_in_threadpool(store_upload, settings.data_dir, filename, b"".join(chunks))
        except DatasetExistsError as exc:
            raise HTTPException(409, str(exc)) from None
        except UploadTooLargeError as exc:
            raise HTTPException(413, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        return JSONResponse(dataset_card(settings.data_dir, ds), status_code=201)

    @app.get("/api/datasets/{name}/preview")
    def get_dataset_preview(
        name: str, request: Request, size: int | None = Query(None, ge=THUMB_MIN_SIDE, le=THUMB_MAX_SIDE)
    ) -> Response:
        """Pratinjau JPEG (sisi terpanjang <= `size`, bawaan 480). Nama dicocokkan dengan hasil pindai,
        tidak pernah dipakai sebagai path."""
        ds = list_datasets(settings.data_dir, settings.backend).get(name)
        if ds is None:
            raise HTTPException(404, f"dataset '{name}' tidak ditemukan")
        try:
            data, tag = dataset_preview_jpeg(settings.data_dir, ds, clamp_size(size, CARD_THUMB_SIDE))
        except PreviewUnavailableError as exc:
            raise HTTPException(404, str(exc)) from None
        return _cached_bytes(request, data, "image/jpeg", tag)

    @app.post("/api/runs", status_code=202)
    def post_run(req: RunRequest) -> JSONResponse:
        values = checked_values(req)
        try:
            params = build_params(values)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        check_engine(params.kmeans, (params.mode,))
        return start_job("run", [(step_label(params.mode), params)], req.force)

    @app.post("/api/runs/compare", status_code=202)
    def post_compare(req: CompareRequest) -> JSONResponse:
        values = checked_values(req)
        try:
            cpu, opt = build_pair(values)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        check_engine(opt.kmeans, ("opt",))
        return start_job("compare", [(MODE_LABELS["cpu"], cpu), (MODE_LABELS["opt"], opt)], req.force)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"pekerjaan '{job_id}' tidak ada (server dimulai ulang?)")
        return job.snapshot()

    @app.get("/api/jobs/{job_id}/events")
    async def get_job_events(job_id: str) -> StreamingResponse:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"pekerjaan '{job_id}' tidak ada (server dimulai ulang?)")

        async def stream():
            yield "retry: 2000\n\n"
            yield _sse("snapshot", job.snapshot(), 0)
            sent, last_tick = 0, 0.0
            while True:
                events, finished, tick = job.poll(sent)
                for ev in events:
                    yield _sse(ev["type"], ev["data"], ev["id"])
                    sent = ev["id"]
                if finished:
                    return                              # peristiwa penutup (done/error) sudah terkirim
                now = time.monotonic()
                if now - last_tick >= TICK_EVERY_S:
                    yield _sse("tick", tick)
                    last_tick = now
                await asyncio.sleep(POLL_EVERY_S)

        return StreamingResponse(
            stream(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/runs")
    def get_runs() -> dict:
        found = lib()
        notes = jobs.notes
        return {
            "runs": [
                views.run_summary(run, found.created[rid], notes.get(rid))
                for rid, run in reversed(found.runs.items())
            ],
            "unreadable": [{"run_id": rid, "reason": why} for rid, why in found.unreadable.items()],
            "groups": library.groups(found),
        }

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        found = lib()
        run = found.get(run_id)
        if run is None:
            why = found.unreadable.get(run_id)
            raise HTTPException(404, f"run '{run_id}' tidak ada" if why is None else f"run '{run_id}' tidak terbaca: {why}")
        return views.run_detail(run, found.created.get(run_id), jobs.notes.get(run_id))

    def _pair(
        dataset: str | None, k: int | None, iters: int | None, seed: int | None,
        cpu: str | None, opt: str | None,
    ):
        found = lib()
        given = [v is not None for v in (dataset, k, iters, seed)]
        if any(given) and not all(given):
            raise HTTPException(422, "dataset, k, iters, dan seed harus diberikan bersamaan (atau tidak sama sekali)")
        key = library.PairKey(dataset, k, iters, seed) if all(given) else None
        try:
            cpu_run, opt_run = library.select_pair(found, key=key, cpu=cpu, opt=opt)
        except library.LibraryError as exc:
            raise HTTPException(422, str(exc)) from None
        return found, cpu_run, opt_run

    @app.get("/api/compare")
    def get_compare(
        dataset: str | None = None, k: int | None = Query(None, ge=1), iters: int | None = Query(None, ge=1),
        seed: int | None = Query(None, ge=0), cpu: str | None = None, opt: str | None = None,
    ) -> dict:
        found, cpu_run, opt_run = _pair(dataset, k, iters, seed, cpu, opt)
        return views.compare_payload(cpu_run, opt_run, found.created, jobs.notes)

    @app.get("/api/verify")
    def get_verify(cpu: str, opt: str) -> dict:
        found = lib()
        runs = {}
        for side, rid in (("cpu", cpu), ("opt", opt)):
            run = found.get(rid)
            if run is None:
                raise HTTPException(404, f"run {side} '{rid}' tidak ada")
            runs[side] = run
        try:
            result = verify_cache.get(runs["cpu"], runs["opt"])
        except (OSError, ValueError) as exc:           # batas API: verifikasi gagal bukan galat server
            return views.verification_error_payload(str(exc))
        return views.verification_payload(result, runs["cpu"], runs["opt"])

    # --- pemetaan: gambar, luas, ekspor ------------------------------------------------------------ #

    @app.get("/api/runs/{run_id}/original")
    def get_run_original(
        run_id: str, request: Request, size: int | None = Query(None, ge=THUMB_MIN_SIDE, le=THUMB_MAX_SIDE)
    ) -> Response:
        """Citra asli (JPEG, sisi terpanjang <= 1024) untuk run ini, dari logika yang sama dengan Streamlit."""
        _, run = run_or_404(run_id)
        image, why = original_preview(run, settings.data_dir)
        if image is None:
            raise HTTPException(404, why or "Citra asli tidak tersedia")
        if size is not None:
            image = thumbnail(image, size)
        data = jpeg_bytes(image)
        return _cached_bytes(request, data, "image/jpeg", f"{run_stamp(run.path)}-{size}-{len(data)}")

    def _run_png(run_id: str, filename: str) -> FileResponse:
        _, run = run_or_404(run_id)
        return FileResponse(run.path / filename, media_type="image/png", headers={"Cache-Control": "no-cache"})

    @app.get("/api/runs/{run_id}/labels_preview")
    def get_labels_preview(run_id: str) -> FileResponse:
        """`labels_preview.png` apa adanya (id cluster per piksel, abu-abu 8 bit)."""
        return _run_png(run_id, "labels_preview.png")

    @app.get("/api/runs/{run_id}/map_preview")
    def get_map_preview(run_id: str) -> FileResponse:
        """`map_preview.png` apa adanya (warna centroid)."""
        return _run_png(run_id, "map_preview.png")

    @app.get("/api/runs/{run_id}/map")
    def get_run_map(
        run_id: str, request: Request, names: str | None = Query(None, max_length=4000),
        view: str = Query(VIEW_CLASS), v: str | None = None,
    ) -> Response:
        """Peta pratinjau berwarna kelas: `names` = daftar JSON nama akhir per cluster (dari `POST .../area`),
        `view=centroid` untuk warna centroid. `v` hanya pemutus cache di URL (cap run)."""
        _, run = run_or_404(run_id)
        if view not in (VIEW_CLASS, VIEW_CENTROID):
            raise HTTPException(422, f"view harus {VIEW_CLASS!r} atau {VIEW_CENTROID!r}")
        try:
            class_map = class_map_from_names(_parse_names(names), run.k)
            data = png_bytes(class_map_image(run, class_map, view))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        return _cached_bytes(request, data, "image/png", f"{run_stamp(run.path)}-{zlib.crc32(data):08x}")

    @app.post("/api/runs/{run_id}/area")
    def post_area(run_id: str, req: AreaRequest) -> dict:
        """Tabel luas, tabel cluster, ringkasan, palet, dan daftar ekspor untuk `class_map` yang dikirim.
        Tanpa keadaan: nama yang ditolak (mis. `Total`) kembali ke "Belum diberi nama" dan galatnya dilaporkan."""
        found, run = run_or_404(run_id)
        try:
            class_map, errors = class_map_from_rows(_cluster_rows(req.class_map), run.k)
            return views.area_payload(run, class_map, errors, found.created.get(run_id))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/runs/{run_id}/export/{kind}")
    def get_export(run_id: str, kind: str, names: str | None = Query(None, max_length=4000)) -> Response:
        """Unduhan: `map` (PNG pratinjau), `area` (CSV luas), `clusters` (CSV per cluster), `full` (PNG resolusi
        penuh dari `labels.u8`, ditulis ke berkas sementara). Nama berkas dari `export_names` (awalan TIRUAN_)."""
        if kind not in EXPORT_KINDS:
            raise HTTPException(404, f"jenis ekspor '{kind}' tidak dikenal (pilih: {', '.join(EXPORT_KINDS)})")
        _, run = run_or_404(run_id)
        filename = export_names(run_id, run.mock)[kind]
        try:
            class_map = class_map_from_names(_parse_names(names), run.k)
            if kind == "full":
                if labels_file_of(run) is None:
                    reason = next(i["reason"] for i in export_items(run) if i["kind"] == "full")
                    raise HTTPException(404, reason)
                folder = Path(tempfile.mkdtemp(prefix="lc_full_"))
                try:
                    target = export_full_map(run, class_map, out_path=folder / filename)
                except BaseException:
                    shutil.rmtree(folder, ignore_errors=True)
                    raise
                return FileResponse(
                    target, media_type=EXPORT_MEDIA_TYPES[kind], filename=filename,
                    background=BackgroundTask(shutil.rmtree, folder, ignore_errors=True),
                )
            if kind == "map":
                data = png_bytes(class_map_image(run, class_map, VIEW_CLASS))
            elif kind == "area":
                data = area_csv(run, class_map)
            else:
                data = cluster_csv(run)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from None
        except (OSError, ValueError) as exc:
            raise HTTPException(422, f"ekspor gagal: {exc}") from None
        return Response(
            data, media_type=EXPORT_MEDIA_TYPES[kind],
            headers={"Content-Disposition": _content_disposition(filename), "Cache-Control": "no-store"},
        )

    # --- frontend hasil build ---------------------------------------------------------------- #

    dist = settings.web_dist
    if dist is not None and (dist / "index.html").is_file():
        dist = dist.resolve()
        if (dist / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            if path == "api" or path.startswith("api/"):
                raise HTTPException(404, "rute API tidak ada")
            candidate = (dist / path).resolve()
            if path and candidate.is_file() and dist in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        def no_frontend() -> JSONResponse:
            return JSONResponse(
                {"detail": "Frontend belum dibangun: jalankan `npm run build` di web/ "
                           "(atau `npm run dev` untuk mode pengembangan; API ada di /api/docs)."},
                status_code=404,
            )

    return app
