"""Append-only acquisition sessions; raw observations are never overwritten."""
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .geometry import validate_transform


def serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(value, f, default=serializable, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


class SessionWriter:
    def __init__(self, parent, metadata):
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]
        self.path = Path(parent)/name
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path/"images").mkdir()
        self.count = 0
        self.metadata = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                         "translation_unit": "m", "transform_convention": "T_a2b maps column a to b",
                         **metadata}
        write_json(self.path/"session.json", self.metadata)

    def append(self, image, sample):
        sample = dict(sample)
        validate_transform(sample["G"])
        validate_transform(sample["C"])
        sid = f"{self.count:05d}"
        path = self.path/"images"/(sid+".png")
        if path.exists():
            raise FileExistsError(path)
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("PNG encoding failed")
        with path.open("xb") as f:
            f.write(encoded.tobytes())
            f.flush()
            os.fsync(f.fileno())
        sample.update(id=sid, image=str(path.relative_to(self.path)), image_sha256=digest(path))
        # A crash can leave an unreferenced image, but never a reference to a partial image.
        with (self.path/"samples.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(sample, default=serializable, ensure_ascii=False, allow_nan=False)+"\n")
            f.flush()
            os.fsync(f.fileno())
        self.count += 1
        return sample


def load_session(path):
    path = Path(path)
    meta = json.loads((path/"session.json").read_text())
    samples = []
    for line in (path/"samples.jsonl").read_text().splitlines():
        sample = json.loads(line)
        for key in ["G", "C"]:
            sample[key] = validate_transform(sample[key])
        for key in ["object_points", "image_points"]:
            sample[key] = np.asarray(sample[key], dtype=float)
        raw = path/sample["image"]
        if digest(raw) != sample["image_sha256"]:
            raise ValueError(f"Image changed: {raw}")
        samples.append(sample)
    if len(samples) < 6:
        raise ValueError("Need at least six recorded samples")
    return meta, samples
