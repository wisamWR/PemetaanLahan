"""Penemuan dataset di `data/` dan penyimpanan unggahan (tanpa Streamlit).

Sumber dataset (plan 4.2 dan Langkah 6), urut prioritas bila namanya sama:
  1. `data/<nama>/manifest.json`   folder tile yang sudah dipotong-potong (jalur utama)
  2. `data/<nama>.png`             satu PNG kecil tanpa manifest
  3. `data/uploads/<nama>.png`     citra kecil yang diunggah lewat UI

Di backend `mock` ditambahkan dataset sintetis (tanpa berkas): MockLauncher sudah
menangani nama yang tidak punya folder dengan citra sintetis ukuran tile penuh.
Dengan begitu alur UI bisa dicoba walau `data/` masih kosong.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from launchers.base import is_safe_dataset_name

KIND_FOLDER = "folder"
KIND_PNG = "png"
KIND_SYNTHETIC = "synthetic"

UPLOADS_DIR = "uploads"
SYNTHETIC_NAMES = ("tiruan_sleman", "tiruan_kulonprogo")

# Nama berkas yang dicadangkan Windows: tidak boleh jadi nama dataset unggahan.
_WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)
_UNSAFE_CHARS = re.compile(r"[^\w .-]+")


class DatasetExistsError(ValueError):
    """Nama dataset unggahan sudah dipakai."""


@dataclass(frozen=True)
class Dataset:
    name: str
    kind: str              # KIND_FOLDER | KIND_PNG | KIND_SYNTHETIC
    path: Path | None      # folder atau berkas PNG; None untuk dataset sintetis

    @property
    def label(self) -> str:
        if self.kind == KIND_FOLDER:
            return f"{self.name} (folder tile)"
        if self.kind == KIND_PNG:
            return f"{self.name} (PNG)"
        return f"{self.name} (sintetis, tiruan)"


@dataclass(frozen=True)
class DatasetScan:
    datasets: tuple[Dataset, ...]
    skipped: tuple[str, ...] = ()      # "<item>: <alasan>" untuk yang tidak bisa dipakai

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(d.name for d in self.datasets)

    def get(self, name: str) -> Dataset | None:
        for d in self.datasets:
            if d.name == name:
                return d
        return None


def scan_datasets(data_dir: Path | str) -> DatasetScan:
    """Cari dataset nyata di `data_dir`. Folder yang tidak ada = tidak ada dataset."""
    root = Path(data_dir)
    found: dict[str, Dataset] = {}
    skipped: list[str] = []

    def add(name: str, kind: str, path: Path, shown: str) -> None:
        if not is_safe_dataset_name(name):
            skipped.append(f"{shown}: nama tidak aman (pakai huruf, angka, _ . - dan spasi di tengah)")
        elif name in found:
            skipped.append(f"{shown}: nama '{name}' sudah dipakai dataset lain ({found[name].kind})")
        else:
            found[name] = Dataset(name, kind, path)

    if not root.is_dir():
        return DatasetScan(())

    entries = sorted(root.iterdir(), key=lambda p: p.name.lower())
    for p in entries:
        if p.is_dir() and (p / "manifest.json").is_file():
            add(p.name, KIND_FOLDER, p, f"{p.name}/")
    for p in entries:
        if p.is_file() and p.suffix.lower() == ".png":
            add(p.stem, KIND_PNG, p, p.name)
    uploads = root / UPLOADS_DIR
    if uploads.is_dir():
        for p in sorted(uploads.iterdir(), key=lambda p: p.name.lower()):
            if p.is_file() and p.suffix.lower() == ".png":
                add(p.stem, KIND_PNG, p, f"{UPLOADS_DIR}/{p.name}")
    return DatasetScan(tuple(found.values()), tuple(skipped))


def list_datasets(data_dir: Path | str, backend: str) -> DatasetScan:
    """Dataset yang ditawarkan UI: yang nyata, plus yang sintetis bila backend `mock`."""
    scan = scan_datasets(data_dir)
    if backend != "mock":
        return scan
    extra = tuple(
        Dataset(n, KIND_SYNTHETIC, None) for n in SYNTHETIC_NAMES if scan.get(n) is None
    )
    return DatasetScan(scan.datasets + extra, scan.skipped)


def resolve_dataset(data_dir: Path | str, name: str) -> Dataset | None:
    """Dataset nyata bernama `name`, atau None (mis. nama sintetis atau tidak ada)."""
    return scan_datasets(data_dir).get(name)


# --- unggahan ------------------------------------------------------------- #

def sanitize_upload_name(filename: str) -> str:
    """Nama dataset dari nama berkas unggahan: tanpa ekstensi, karakter asing jadi `_`.

    ValueError bila hasilnya kosong, tidak aman, atau nama cadangan Windows."""
    stem = Path(str(filename).replace("\\", "/")).stem
    name = _UNSAFE_CHARS.sub("_", stem).strip(" ._-")
    if not name or not is_safe_dataset_name(name):
        raise ValueError(f"nama berkas {filename!r} tidak bisa dijadikan nama dataset")
    if name.lower() in _WINDOWS_RESERVED:
        raise ValueError(f"nama {name!r} dicadangkan sistem; ganti nama berkasnya")
    return name


def _check_png(data: bytes) -> None:
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format != "PNG":
                raise ValueError(f"berkas bukan PNG (terdeteksi: {im.format})")
            im.verify()
    except ValueError:
        raise
    except Exception as exc:  # Pillow: UnidentifiedImageError, DecompressionBombError, dst.
        raise ValueError(f"berkas bukan PNG yang valid ({type(exc).__name__}: {exc})") from None


def save_upload(data_dir: Path | str, filename: str, data: bytes) -> Dataset:
    """Simpan PNG kecil ke `data/uploads/<nama>.png` dan kembalikan datasetnya.

    Tidak menimpa dataset yang sudah ada (hasil tersimpan di runs/ diberi kunci dari nama
    dataset, jadi isi baru dengan nama lama membuat hasil itu basi): `DatasetExistsError`."""
    name = sanitize_upload_name(filename)
    _check_png(data)
    root = Path(data_dir)
    if scan_datasets(root).get(name) is not None:
        raise DatasetExistsError(f"dataset '{name}' sudah ada; ganti nama berkas atau hapus yang lama")
    target = root / UPLOADS_DIR / f"{name}.png"
    if target.exists():     # mis. '.PNG' besar-kecil huruf yang tidak terdeteksi scan
        raise DatasetExistsError(f"berkas {target.name} sudah ada di {UPLOADS_DIR}/")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
    return Dataset(name, KIND_PNG, target)


__all__ = [
    "DatasetExistsError", "Dataset", "DatasetScan", "KIND_FOLDER", "KIND_PNG",
    "KIND_SYNTHETIC", "SYNTHETIC_NAMES", "UPLOADS_DIR", "list_datasets", "resolve_dataset",
    "sanitize_upload_name", "save_upload", "scan_datasets",
]
