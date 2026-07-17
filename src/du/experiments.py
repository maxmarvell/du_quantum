from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

RUN_RE = re.compile(r"^run_(\d+)")
_PAD = 4
MAIN_FILE = "data.npz"
METADATA_FILE = "metadata.json"
PLOTS_DIR = "plots"


# --------------------------------------------------------------------------- #
# Creating / opening runs
# --------------------------------------------------------------------------- #
def create_run(
    canonical_folder: str,
    run_name: str | None = None,
    *,
    data_root: str | os.PathLike = "data",
    params: dict | None = None,
    provenance: bool = True,
) -> "Run":
    """Create the next ``run_00XX`` folder under ``data/{canonical_folder}/``.

    ``run_name`` is slugified and appended (``run_0003_bond2000``). ``params``
    is written to ``metadata.json`` under a ``"params"`` key; when
    ``provenance`` is true, timestamp/git/host info is captured too. Retries on
    collision so concurrent jobs get distinct numbers.
    """
    base = Path(data_root) / canonical_folder
    base.mkdir(parents=True, exist_ok=True)
    suffix = f"_{_slugify(run_name)}" if run_name else ""

    for _ in range(50):
        idx = _next_index(base)
        path = base / f"run_{idx:0{_PAD}d}{suffix}"
        try:
            path.mkdir()
            break
        except FileExistsError:
            continue
    else:  # pragma: no cover
        raise RuntimeError(f"could not allocate a run folder under {base}")

    run = Run(path)
    meta: dict = {
        "run": {
            "index": idx,
            "name": run_name,
            "folder": path.name,
            "canonical_folder": canonical_folder,
        }
    }
    if provenance:
        meta["provenance"] = _provenance()
    if params:
        meta["params"] = params
    run.save_metadata(meta, replace=True)
    return run


def open_run(path: str | os.PathLike) -> "Run":
    """Wrap an existing run folder."""
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    return Run(path)


def latest_run(
    canonical_folder: str, *, data_root: str | os.PathLike = "data"
) -> "Run | None":
    """Return the highest-numbered run under ``data/{canonical_folder}/``."""
    base = Path(data_root) / canonical_folder
    if not base.is_dir():
        return None
    runs = [
        (int(m.group(1)), p)
        for p in base.iterdir()
        if p.is_dir() and (m := RUN_RE.match(p.name))
    ]
    if not runs:
        return None
    return Run(max(runs, key=lambda t: t[0])[1])


# --------------------------------------------------------------------------- #
# Run handle
# --------------------------------------------------------------------------- #
class Run:
    """Handle to one ``run_00XX`` folder with canonical save/load helpers."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Run({self.path})"

    @property
    def plots(self) -> Path:
        """The ``plots/`` subfolder (created on first access)."""
        d = self.path / PLOTS_DIR
        d.mkdir(exist_ok=True)
        return d

    def subdir(self, name: str) -> Path:
        """Create and return a heavy-data subfolder (e.g. ``two_body_rdm``)."""
        d = self.path / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_data(self, **arrays) -> Path:
        """Write the main output arrays to ``data.npz`` (atomic)."""
        out = self.path / MAIN_FILE
        _atomic_savez(out, arrays)
        return out

    def save_array(self, subfolder: str, name: str, **arrays) -> Path:
        """Write heavy arrays to ``{subfolder}/{name}.npz`` (atomic).

        e.g. ``save_array("two_body_rdm", "L_20", rdm=rdm)``.
        """
        out = self.subdir(subfolder) / f"{name}.npz"
        _atomic_savez(out, arrays)
        return out

    def load_data(self) -> dict:
        """Load ``data.npz`` as a plain dict of arrays."""
        with np.load(self.path / MAIN_FILE, allow_pickle=False) as npz:
            return {k: npz[k] for k in npz.files}

    @property
    def metadata(self) -> dict:
        p = self.path / METADATA_FILE
        return json.loads(p.read_text()) if p.exists() else {}

    def save_metadata(self, extra: dict, *, replace: bool = False) -> Path:
        """Merge ``extra`` into ``metadata.json`` (recursive dict merge)."""
        p = self.path / METADATA_FILE
        data = {} if replace else self.metadata
        _deep_merge(data, extra)
        _atomic_write_text(p, json.dumps(data, indent=2, default=str) + "\n")
        return p


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _next_index(base: Path) -> int:
    idx = 0
    for p in base.iterdir():
        if p.is_dir() and (m := RUN_RE.match(p.name)):
            idx = max(idx, int(m.group(1)))
    return idx + 1


def _slugify(name: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_")
    return slug or "run"


def _deep_merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def _provenance() -> dict:
    import datetime
    import getpass
    import socket
    import sys

    prov = {
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
    }
    try:
        prov["hostname"] = socket.gethostname()
        prov["user"] = getpass.getuser()
    except Exception:
        pass
    commit = _git_commit()
    if commit:
        prov["git_commit"] = commit
    return prov


def _git_commit() -> str | None:
    import subprocess

    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        if head.returncode != 0:
            return None
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip()
        return head.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return None


def _atomic_savez(path: Path, arrays: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    # Write via a file object so numpy doesn't re-append ".npz" to the tmp name.
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
