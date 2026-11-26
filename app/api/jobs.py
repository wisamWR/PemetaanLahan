"""Pekerjaan latar: satu run, atau pasangan (CPU saja lalu Optimalisasi) untuk "Bandingkan".

Aturan:
  * SATU pekerjaan aktif pada satu waktu (`JobBusyError` bila ada yang berjalan): menjalankan dua
    benchmark bersamaan akan merusak angka waktu, dan folder run tidak boleh punya dua penulis.
  * Run dijalankan lewat `services.runs.obtain_run` (hasil CPU tersimpan dipakai ulang bila ada dan
    tidak `force`; Optimalisasi SELALU dijalankan), jadi perilakunya sama dengan aplikasi Streamlit.
  * Waktu berjalan yang dikirim ke klien adalah PROGRES (jam dinding), bukan hasil; angka kinerja
    selalu dari `timing.json` di folder run.
  * Tiap pekerjaan selalu berakhir dengan satu peristiwa penutup (`done` atau `failed`; bukan `error`,
    karena `EventSource` sudah memakai nama itu untuk putusnya koneksi), supaya
    klien SSE bisa menutup koneksinya dan tidak tersambung ulang tanpa henti.
  * Galat meniru aplikasi Streamlit: `CliLauncherError` = ringkasan + stderr/daftar masalah; galat
    lain = "<Tipe>: pesan". Langkah sesudah yang gagal berstatus `skipped`; langkah sebelumnya tetap
    tersimpan di disk dan tampil.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Sequence

from launchers import make_launcher
from launchers.base import RunParams, run_id
from launchers.cli import CliLauncherError
from services.runs import MODE_LABELS, obtain_run

from api.settings import Settings

MAX_LINES = 200
KEEP_JOBS = 20

STATUS_QUEUED, STATUS_RUNNING, STATUS_DONE, STATUS_ERROR = "queued", "running", "done", "error"
STEP_PENDING, STEP_RUNNING, STEP_DONE, STEP_ERROR, STEP_SKIPPED = "pending", "running", "done", "error", "skipped"
FINISHED = (STATUS_DONE, STATUS_ERROR)


class JobBusyError(RuntimeError):
    """Sudah ada pekerjaan yang berjalan."""

    def __init__(self, job_id: str) -> None:
        super().__init__("Masih ada run yang berjalan; tunggu selesai (angka waktu tidak boleh bercampur).")
        self.job_id = job_id


@dataclass
class Step:
    index: int
    label: str
    mode: str
    params: RunParams
    run_id: str
    status: str = STEP_PENDING
    note: str | None = None
    reused: bool = False
    started: float | None = None           # monotonic, hanya untuk progres
    seconds: float | None = None           # lama langkah (jam dinding), bukan hasil pengukuran

    def as_dict(self) -> dict:
        return {
            "index": self.index, "label": self.label, "mode": self.mode, "run_id": self.run_id,
            "status": self.status, "note": self.note, "reused": self.reused, "seconds": self.seconds,
        }


class Job:
    def __init__(self, kind: str, steps: Sequence[tuple[str, RunParams]], force: bool) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind                               # "run" | "compare"
        self.force = force
        self.steps = [
            Step(i, label, p.mode, p, run_id(p)) for i, (label, p) in enumerate(steps)
        ]
        self.status = STATUS_QUEUED
        self.error: dict | None = None
        self.created_at = time.time()
        self.t0: float | None = None                   # monotonic
        self.t1: float | None = None
        self.lines: deque[str] = deque(maxlen=MAX_LINES)
        self._events: list[dict] = []
        self._lock = threading.Lock()

    # --- penulisan (dari thread pekerja) -------------------------------------------- #

    def emit(self, type_: str, **data) -> None:
        with self._lock:
            self._append(type_, data)

    def _append(self, type_: str, data: dict) -> None:
        self._events.append({"id": len(self._events) + 1, "type": type_, "data": data})

    def finish(self, status: str, type_: str, **data) -> None:
        """Atomik: status akhir dan peristiwa penutup tampak bersamaan bagi pembaca SSE."""
        with self._lock:
            self.status = status
            self.t1 = time.monotonic()
            self._append(type_, {**data, "job": self._snapshot_locked()})

    def add_line(self, step: Step, line: str) -> None:
        with self._lock:
            self.lines.append(line)
            self._append("progress", {"step": step.index, "line": line})

    def set_step(self, step: Step, status: str, **fields) -> None:
        with self._lock:
            step.status = status
            for key, value in fields.items():
                setattr(step, key, value)
            self._append("step", step.as_dict())

    # --- pembacaan ----------------------------------------------------------------------- #

    def elapsed(self) -> float:
        if self.t0 is None:
            return 0.0
        return (self.t1 if self.t1 is not None else time.monotonic()) - self.t0

    def _snapshot_locked(self) -> dict:
        running = next((s for s in self.steps if s.status == STEP_RUNNING), None)
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "force": self.force,
            "created_at": datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat(timespec="seconds"),
            "elapsed_s": self.elapsed(),
            "running_step": None if running is None else running.index,
            "steps": [s.as_dict() for s in self.steps],
            "lines": list(self.lines),
            "error": self.error,
            "run_ids": [s.run_id for s in self.steps],
        }

    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot_locked()

    def poll(self, after: int) -> tuple[list[dict], bool, dict]:
        """(peristiwa dengan id > `after`, pekerjaan sudah selesai?, info progres). Atomik."""
        with self._lock:
            events = self._events[after:]
            finished = self.status in FINISHED
            running = next((s for s in self.steps if s.status == STEP_RUNNING), None)
            tick = {
                "elapsed_s": self.elapsed(),
                "step": None if running is None else running.index,
                "step_elapsed_s": None if running is None or running.started is None
                else time.monotonic() - running.started,
            }
        return events, finished, tick


LauncherFactory = Callable[..., object]


class JobManager:
    def __init__(self, settings: Settings, launcher_factory: LauncherFactory = make_launcher) -> None:
        self.settings = settings
        self._factory = launcher_factory
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self.notes: dict[str, str] = {}          # run_id -> "hasil tersimpan, ..." (sesi server ini)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def active(self) -> Job | None:
        with self._lock:
            return self._active_locked()

    def _active_locked(self) -> Job | None:
        for job in self._jobs.values():
            if job.status not in FINISHED:
                return job
        return None

    def start(self, kind: str, steps: Sequence[tuple[str, RunParams]], force: bool) -> Job:
        """Mulai pekerjaan di thread latar. `JobBusyError` bila ada yang aktif; `ValueError` bila
        launcher tidak bisa dibuat (parameter backend salah), sebelum apa pun dimulai."""
        with self._lock:
            busy = self._active_locked()
            if busy is not None:
                raise JobBusyError(busy.id)
            s = self.settings
            launcher = self._factory(s.backend, s.data_dir, s.core_bin)
            job = Job(kind, steps, force)
            job.status = STATUS_RUNNING
            job.t0 = time.monotonic()
            self._jobs[job.id] = job
            while len(self._jobs) > KEEP_JOBS:
                oldest = next((j for j, v in self._jobs.items() if v.status in FINISHED), None)
                if oldest is None:
                    break
                del self._jobs[oldest]
        threading.Thread(target=self._work, args=(job, launcher), name=f"job-{job.id}", daemon=True).start()
        return job

    # --- thread pekerja -------------------------------------------------------------------- #

    def _fail(self, job: Job, step: Step, message: str, detail: str | None) -> None:
        job.set_step(step, STEP_ERROR, seconds=None if step.started is None else time.monotonic() - step.started)
        for later in job.steps[step.index + 1:]:
            job.set_step(later, STEP_SKIPPED)
        job.error = {"message": message, "detail": detail or None, "step": step.index}
        job.finish(STATUS_ERROR, "failed", message=message, detail=detail or None, step=step.index)

    def _work(self, job: Job, launcher: object) -> None:
        s = self.settings
        current: Step | None = None
        try:
            for step in job.steps:
                current = step
                step.started = time.monotonic()
                job.set_step(step, STEP_RUNNING)
                try:
                    outcome = obtain_run(
                        launcher, s.runs_dir, s.backend, step.params, force=job.force,
                        on_progress=lambda line, _step=step: job.add_line(_step, line),
                    )
                except CliLauncherError as exc:        # program landcover: ringkasan + stderr/daftar masalah
                    self._fail(job, step, f"Run '{step.label}' gagal: {exc.summary}", exc.detail)
                    return
                except Exception as exc:               # batas API: galat launcher jadi pesan, bukan crash
                    self._fail(job, step, f"Run '{step.label}' gagal: {type(exc).__name__}: {exc}", None)
                    return
                if outcome.note:
                    self.notes[outcome.run_id] = outcome.note
                else:
                    self.notes.pop(outcome.run_id, None)
                job.set_step(
                    step, STEP_DONE, note=outcome.note, reused=outcome.reused,
                    seconds=time.monotonic() - step.started,
                )
            job.finish(STATUS_DONE, "done")
        except BaseException as exc:                   # jaring pengaman: klien harus menerima peristiwa penutup
            if job.status not in FINISHED:
                step = current or job.steps[0]
                self._fail(job, step, f"Run '{step.label}' gagal: {type(exc).__name__}: {exc}", None)


def step_label(mode: str) -> str:
    return MODE_LABELS[mode]
