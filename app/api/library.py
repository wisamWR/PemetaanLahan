"""Pustaka hasil di disk: pindai `runs/<backend>/`, kelompokkan menurut kunci sebanding, pilih pasangan.

Folder run adalah sumber kebenaran (hasil bertahan antar muat ulang). Urutan "terbaru" = waktu
`created_at` di `meta.json` (cadangan: waktu ubah berkas), bukan urutan folder. Pasangan (CPU saja,
Optimalisasi) dipilih dengan `services.compare.comparable_runs` yang sama dengan aplikasi Streamlit.
`run_id` dari klien TIDAK PERNAH dipakai untuk membentuk path: hanya dicocokkan dengan hasil pindai.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from landcover.contract import ContractError, RunResult
from launchers.base import BACKEND_DIRS
from services.compare import comparable_runs, pair_key
from services.runs import load_run, run_id_of


class LibraryError(ValueError):
    """Permintaan pasangan yang tidak masuk akal (run tidak ada, kunci beda)."""


@dataclass(frozen=True)
class Library:
    runs: dict[str, RunResult]              # run_id -> hasil, urut dari terlama ke terbaru
    unreadable: dict[str, str]              # run_id -> alasan tidak terbaca
    created: dict[str, float]               # run_id -> detik epoch (untuk urutan dan tampilan)

    def get(self, rid: str) -> RunResult | None:
        return self.runs.get(rid)


def _created_ts(run: RunResult) -> float:
    try:
        return datetime.fromisoformat(str(run.meta["created_at"])).timestamp()
    except (KeyError, ValueError, OSError, OverflowError):
        try:
            return (run.path / "meta.json").stat().st_mtime
        except OSError:
            return 0.0


def _meta_mtime_ns(run: RunResult) -> int:
    try:
        return (run.path / "meta.json").stat().st_mtime_ns
    except OSError:
        return 0


def runs_root(runs_dir: Path | str, backend: str) -> Path:
    return Path(runs_dir) / BACKEND_DIRS[backend]


def scan_library(runs_dir: Path | str, backend: str) -> Library:
    """Baca semua folder run di `runs/<backend>/` yang punya `meta.json`."""
    root = runs_root(runs_dir, backend)
    loaded: dict[str, RunResult] = {}
    unreadable: dict[str, str] = {}
    if root.is_dir():
        for folder in sorted(root.iterdir()):
            if not folder.is_dir() or not (folder / "meta.json").is_file():
                continue
            try:
                loaded[folder.name] = load_run(folder)
            except (ContractError, OSError) as exc:
                unreadable[folder.name] = str(exc)
    created = {rid: _created_ts(run) for rid, run in loaded.items()}
    # `created_at` hanya teliti sampai detik: run yang selesai dalam detik yang sama dibedakan oleh
    # waktu ubah `meta.json` (ditulis terakhir oleh launcher), baru nama folder.
    ordered = sorted(loaded, key=lambda rid: (created[rid], _meta_mtime_ns(loaded[rid]), rid))
    return Library(
        runs={rid: loaded[rid] for rid in ordered},
        unreadable=unreadable,
        created={rid: created[rid] for rid in ordered},
    )


@dataclass(frozen=True)
class PairKey:
    dataset: str
    k: int
    iters: int
    seed: int

    def as_dict(self) -> dict:
        return {"dataset": self.dataset, "k": self.k, "iters": self.iters, "seed": self.seed}


def key_of(run: RunResult) -> PairKey:
    dataset, k, iters, seed = pair_key(run)
    return PairKey(str(dataset), int(k), int(iters), int(seed))


def groups(lib: Library) -> list[dict]:
    """Kelompok run yang sebanding (dataset, K, iterasi, seed sama), yang terbaru di atas."""
    by_key: dict[PairKey, list[str]] = {}
    rank = {rid: i for i, rid in enumerate(lib.runs)}            # urutan terlama -> terbaru
    for rid, run in lib.runs.items():
        by_key.setdefault(key_of(run), []).append(rid)
    out = []
    for key, rids in by_key.items():
        cpu = [r for r in rids if lib.runs[r].mode == "cpu"]
        opt = [r for r in rids if lib.runs[r].mode != "cpu"]
        out.append({
            **key.as_dict(),
            "cpu_run_id": cpu[-1] if cpu else None,
            "opt_run_ids": list(reversed(opt)),
            "latest_run_id": rids[-1],
            "latest_ts": lib.created[rids[-1]],
            "complete": bool(cpu and opt),
        })
    out.sort(key=lambda g: rank[g["latest_run_id"]], reverse=True)
    return out


def select_pair(
    lib: Library,
    *,
    key: PairKey | None = None,
    cpu: str | None = None,
    opt: str | None = None,
) -> tuple[RunResult | None, RunResult | None]:
    """(CPU saja, Optimalisasi) sebanding.

    `cpu`/`opt` eksplisit (run_id) menang; sisi yang tidak disebut = run TERBARU dengan kunci yang
    sama. Tanpa petunjuk apa pun dipakai kunci run terbaru di pustaka. `LibraryError` bila run yang
    disebut tidak ada, mode-nya salah, atau kuncinya tidak sama dengan `key`/sisi lain."""
    if not lib.runs:
        return None, None
    chosen: dict[str, RunResult] = {}
    for side, rid, want_cpu in (("cpu", cpu, True), ("opt", opt, False)):
        if rid is None:
            continue
        run = lib.get(rid)
        if run is None:
            raise LibraryError(f"run {side} '{rid}' tidak ada di pustaka hasil")
        if (run.mode == "cpu") != want_cpu:
            raise LibraryError(f"run '{rid}' bukan mode {'CPU saja' if want_cpu else 'Optimalisasi'}")
        chosen[side] = run
    if len(chosen) == 2 and key_of(chosen["cpu"]) != key_of(chosen["opt"]):
        raise LibraryError("run CPU saja dan Optimalisasi yang dipilih tidak sebanding (dataset, K, iterasi, atau seed beda)")

    anchor = chosen.get("opt") or chosen.get("cpu")
    if anchor is not None:
        wanted = key_of(anchor)
        if key is not None and key != wanted:
            raise LibraryError("run yang dipilih tidak cocok dengan dataset, K, iterasi, dan seed yang diminta")
    elif key is not None:
        wanted = key
    else:
        wanted = key_of(next(reversed(lib.runs.values())))
    members = {rid: run for rid, run in lib.runs.items() if key_of(run) == wanted}
    if not members:
        return None, None
    anchor_id = run_id_of(anchor) if anchor is not None else next(reversed(members))
    cpu_run, opt_run = comparable_runs(members, anchor_id)
    if "cpu" in chosen:
        cpu_run = chosen["cpu"]
    if "opt" in chosen:
        opt_run = chosen["opt"]
    return cpu_run, opt_run
