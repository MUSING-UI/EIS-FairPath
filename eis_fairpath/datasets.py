"""Rebuild the three complete DBLP domains from the local verified BigDND archive."""
from __future__ import annotations
import gzip
import hashlib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path


DBLP_SOURCE_URL = "https://projects.csail.mit.edu/dnd/DBLP/dblp.json.gz"

DBLP_SOURCE_SHA256 = "1503ae44f951af5567c57db477f0598f2efeccd2cca27371be1912612089037e"

@dataclass(frozen=True)
class CSDatasetSpec:
    dataset_id: str
    family: str
    start: int
    end: int
    prefixes: tuple[str, ...] = ()

_DBLP = {
'dblp_mining_ir': CSDatasetSpec("dblp_mining_ir", "dblp", 2003, 2010,
                                     ("conf/kdd/", "conf/icdm/", "conf/sdm/", "conf/sigir/",
                                      "conf/cikm/", "conf/wsdm/")),
'dblp_nlp': CSDatasetSpec("dblp_nlp", "dblp", 1991, 2013,
                               ("conf/acl/", "conf/emnlp/", "conf/naacl/", "conf/coling/",
                                "conf/eacl/")),
'dblp_vision': CSDatasetSpec("dblp_vision", "dblp", 2002, 2009,
                                  ("conf/cvpr/", "conf/iccv/", "conf/eccv/", "conf/bmvc/",
                                   "conf/icpr/"))
}
DATASETS = dict(_DBLP)

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def verify_source(path: str | Path, expected_sha256: str) -> str:
    """Return the source digest, rejecting an unexpected download."""
    actual = _sha256(Path(path))
    if actual.casefold() != str(expected_sha256).casefold():
        raise ValueError(f"source SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    return actual

def author_id(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("author names must be nonempty strings")
    normalized = unicodedata.normalize("NFKC", name).casefold()
    normalized = " ".join(normalized.split())
    return "dblp-name:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()

def _manifest(source: Path, *, dataset_id: str, source_url: str, source_sha256: str,
              author_id_mode: str, selector: dict) -> dict:
    return {
        "dataset_id": dataset_id,
        "source_url": source_url,
        "source_sha256": source_sha256,
        "source_bytes": source.stat().st_size,
        "parser": "src.data.cs_datasets",
        "parser_version": "cs-v1",
        "author_id_mode": author_id_mode,
        "selector": selector,
    }

def _dblp_row(line: str):
    value = json.loads(line.rstrip(","))
    if isinstance(value, list) and len(value) == 3:
        return value[0], value[1], value[2]
    if isinstance(value, dict):
        return value.get("id", value.get("key")), value.get("authors"), value.get("year")
    raise ValueError("DBLP row must be a three-element list or object")

def _venue_for(key: str, prefixes: tuple[str, ...]) -> str:
    prefix = next((p for p in prefixes if key.startswith(p)), None)
    if prefix is None:
        raise ValueError(f"key does not match configured venue prefix: {key}")
    parts = prefix.rstrip("/").split("/")
    return parts[-1].upper()

def build_dblp(path: str | Path, dataset_id: str, *, verify: bool = True) -> dict:
    """Build one fixed DBLP venue/year family from the BigDND JSON stream."""
    if dataset_id not in _DBLP:
        raise ValueError(f"unknown DBLP dataset: {dataset_id}")
    spec = _DBLP[dataset_id]
    source = Path(path)
    digest = verify_source(source, DBLP_SOURCE_SHA256) if verify else _sha256(source)
    records = {}
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line or line in {"[", "]"}:
                continue
            key, names, year = _dblp_row(line)
            if not isinstance(key, str) or not key:
                continue
            if not isinstance(year, int) or not spec.start <= year <= spec.end:
                continue
            if not key.startswith(spec.prefixes):
                continue
            if not isinstance(names, list) or not names:
                continue
            authors = sorted({author_id(name) for name in names})
            venue = _venue_for(key, spec.prefixes)
            normalized = {
                "id": "dblp:" + key,
                "year": year,
                "venue": venue,
                "authors": authors,
                "event_kind": "publication",
            }
            old = records.get(key)
            if old is not None and old != normalized:
                raise ValueError(f"conflicting DBLP record for key {key} at line {line_number}")
            records[key] = normalized
    papers = sorted(records.values(), key=lambda row: (row["year"], row["id"]))
    return {
        "dataset": "cs_" + dataset_id,
        "years": list(range(spec.start, spec.end + 1)),
        "papers": papers,
        "source_manifest": _manifest(
            source, dataset_id=dataset_id, source_url=DBLP_SOURCE_URL,
            source_sha256=digest, author_id_mode="name_hash",
            selector={"prefixes": list(spec.prefixes), "start": spec.start, "end": spec.end}),
    }

def write_dataset(data: dict, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    # Write bytes so the manifest hash is independent of Windows newline translation.
    temporary.write_bytes(payload.encode("utf-8"))
    temporary.replace(destination)
    return destination
