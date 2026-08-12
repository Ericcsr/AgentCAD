#!/usr/bin/env python3
"""Design version history: snapshot + rollback for CAD + conversation state."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


VERSIONS_DIRNAME = "versions"


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class VersionMeta:
    id: str
    label: str
    created_at: str
    note: str = ""
    requirements: str = ""
    design_mode: str = "draft"
    parent_id: str | None = None
    n_triangles: int = 0
    n_history: int = 0

    def short_label(self) -> str:
        base = self.label or self.id
        when = self.created_at
        if "T" in when:
            when = when.replace("T", " ")[:19]
        return f"{self.id} · {base}"


@dataclass
class VersionSnapshot:
    meta: VersionMeta
    code: str
    history: list[dict[str, str]] = field(default_factory=list)
    features: list[dict[str, str]] = field(default_factory=list)
    chat_log: str = ""
    worksheet_md: str = ""
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None

    @property
    def id(self) -> str:
        return self.meta.id


class DesignVersionStore:
    """Filesystem-backed version history under generated/versions/."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.versions_dir = self.root / "generated" / VERSIONS_DIRNAME
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.versions_dir / "index.json"

    def list_versions(self) -> list[VersionMeta]:
        index = self._load_index()
        out: list[VersionMeta] = []
        for item in index.get("versions", []):
            try:
                out.append(
                    VersionMeta(
                        **{k: item[k] for k in VersionMeta.__dataclass_fields__ if k in item}
                    )
                )
            except Exception:
                continue
        return out

    def latest(self) -> VersionMeta | None:
        versions = self.list_versions()
        return versions[-1] if versions else None

    def get(self, version_id: str) -> VersionSnapshot | None:
        path = self._version_dir(version_id)
        meta_path = path / "meta.json"
        if not meta_path.exists():
            return None
        raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta = VersionMeta(
            **{k: raw_meta[k] for k in VersionMeta.__dataclass_fields__ if k in raw_meta}
        )
        code = (path / "design.py").read_text(encoding="utf-8") if (path / "design.py").exists() else ""
        history: list[dict[str, str]] = []
        if (path / "history.json").exists():
            history = json.loads((path / "history.json").read_text(encoding="utf-8"))
        features: list[dict[str, str]] = []
        if (path / "features.json").exists():
            features = json.loads((path / "features.json").read_text(encoding="utf-8"))
        chat_log = (
            (path / "chat_log.txt").read_text(encoding="utf-8")
            if (path / "chat_log.txt").exists()
            else ""
        )
        worksheet_md = (
            (path / "worksheet.md").read_text(encoding="utf-8")
            if (path / "worksheet.md").exists()
            else ""
        )
        vertices = faces = None
        mesh_path = path / "mesh.npz"
        if mesh_path.exists():
            data = np.load(mesh_path)
            vertices = np.asarray(data["vertices"], dtype=np.float64)
            faces = np.asarray(data["faces"], dtype=np.int32)
        return VersionSnapshot(
            meta=meta,
            code=code,
            history=history,
            features=features,
            chat_log=chat_log,
            worksheet_md=worksheet_md,
            vertices=vertices,
            faces=faces,
        )

    def commit(
        self,
        *,
        code: str,
        history: Sequence[dict[str, str]],
        features: Sequence[Any],
        requirements: str,
        design_mode: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        chat_log: str = "",
        worksheet_md: str = "",
        label: str = "",
        note: str = "",
        parent_id: str | None = None,
    ) -> VersionMeta:
        index = self._load_index()
        seq = int(index.get("next_seq", 1))
        version_id = f"v{seq:03d}"
        created = _now_iso()
        if not label:
            label = note.strip()[:60] if note.strip() else f"snapshot {_now_stamp()}"
        versions = list(index.get("versions", []))
        meta = VersionMeta(
            id=version_id,
            label=label,
            created_at=created,
            note=note,
            requirements=requirements,
            design_mode=design_mode,
            parent_id=parent_id or (versions[-1]["id"] if versions else None),
            n_triangles=int(len(faces)),
            n_history=len(history),
        )
        path = self._version_dir(version_id)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

        (path / "meta.json").write_text(json.dumps(asdict(meta), indent=2) + "\n", encoding="utf-8")
        (path / "design.py").write_text(code.rstrip() + "\n", encoding="utf-8")
        (path / "history.json").write_text(
            json.dumps(list(history), indent=2) + "\n", encoding="utf-8"
        )
        feat_payload = []
        for f in features:
            if hasattr(f, "id"):
                feat_payload.append(
                    {
                        "id": f.id,
                        "text": f.text,
                        "status": getattr(f, "status", "pending"),
                        "evidence": getattr(f, "evidence", ""),
                    }
                )
            elif isinstance(f, dict):
                feat_payload.append(f)
        (path / "features.json").write_text(
            json.dumps(feat_payload, indent=2) + "\n", encoding="utf-8"
        )
        (path / "chat_log.txt").write_text(chat_log or "", encoding="utf-8")
        if worksheet_md:
            (path / "worksheet.md").write_text(worksheet_md.rstrip() + "\n", encoding="utf-8")
        np.savez_compressed(
            path / "mesh.npz",
            vertices=np.asarray(vertices, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int32),
        )

        versions.append(asdict(meta))
        index["versions"] = versions
        index["next_seq"] = seq + 1
        index["current_id"] = version_id
        self._save_index(index)
        return meta

    def set_current(self, version_id: str) -> None:
        index = self._load_index()
        index["current_id"] = version_id
        self._save_index(index)

    def current_id(self) -> str | None:
        return self._load_index().get("current_id")

    def delete_after(self, version_id: str) -> int:
        """Remove versions newer than version_id (truncate forward history on rollback)."""
        index = self._load_index()
        versions = index.get("versions", [])
        ids = [v["id"] for v in versions]
        if version_id not in ids:
            return 0
        keep_upto = ids.index(version_id)
        keep = versions[: keep_upto + 1]
        drop = versions[keep_upto + 1 :]
        for item in drop:
            path = self._version_dir(item["id"])
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
        index["versions"] = keep
        index["current_id"] = version_id
        self._save_index(index)
        return len(drop)

    def _version_dir(self, version_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_\-]", "", version_id)
        return self.versions_dir / safe

    def _load_index(self) -> dict[str, Any]:
        if not self._index_path.exists():
            return {"next_seq": 1, "versions": [], "current_id": None}
        try:
            return json.loads(self._index_path.read_text(encoding="utf-8"))
        except Exception:
            return {"next_seq": 1, "versions": [], "current_id": None}

    def _save_index(self, index: dict[str, Any]) -> None:
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self._index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
