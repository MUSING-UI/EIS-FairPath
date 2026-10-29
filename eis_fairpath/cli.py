"""Portable command line entry points for complete DBLP data and explanations."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


DOMAINS = {
    "mining": ("cs_dblp_mining_ir", 2008),
    "nlp": ("cs_dblp_nlp", 2004),
    "vision": ("cs_dblp_vision", 2007),
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(root):
    manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = root / entry["path"]
        if path.stat().st_size != entry["bytes"] or _sha256(path) != entry["sha256"]:
            raise ValueError(f"Data integrity mismatch: {entry['path']}")
    print(f"Verified {len(manifest['files'])} complete data files (SHA-256).")


def _load_graph(root, dataset):
    from .graph import load_graph
    return load_graph(root / "data/processed" / f"{dataset}.json")


def _candidate_pairs(graph, year, limit):
    names = list(graph.node_index["author"])
    groups = graph.queries(year, seed=42, max_positives=limit, negatives=1)["groups"]
    return [{"author_a": names[group[0]], "author_b": names[group[1]], "year": year}
            for group in groups]


def _author(graph, value):
    if value in graph.node_index["author"]:
        return value
    from .datasets import author_id
    normalized = author_id(value)
    if normalized not in graph.node_index["author"]:
        raise ValueError(f"Author is absent from this domain: {value!r}")
    return normalized


def _explain(args, root, dataset, year):
    import torch
    from .checkpoints import load_teacher
    from .explainer import prepare_query, explain

    torch.set_num_threads(args.threads)
    graph = _load_graph(root, dataset)
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else root / "checkpoints" / f"{dataset}.pt"
    if not args.checkpoint:
        manifest = json.loads((root / "checkpoints/manifest.json").read_text(encoding="utf-8"))
        entry = manifest["checkpoints"][dataset]
        if _sha256(checkpoint_path) != entry["sha256"]:
            raise ValueError("Teacher checkpoint SHA-256 mismatch")
        if _sha256(root / "data/processed" / f"{dataset}.json") != entry["graph_sha256"]:
            raise ValueError("Teacher was bound to different graph bytes")
    teacher = load_teacher(checkpoint_path, dataset=dataset)
    if bool(args.author_a) != bool(args.author_b):
        raise ValueError("Specify both --author-a and --author-b")
    if args.author_a:
        pair = (_author(graph, args.author_a), _author(graph, args.author_b))
    else:
        candidates = _candidate_pairs(graph, year, 1)
        if not candidates:
            raise ValueError("No warm-start example query; supply an author pair explicitly")
        pair = (candidates[0]["author_a"], candidates[0]["author_b"])
    # Stable IDs make the evidence probes independent of optimizer mask seeds.
    query_id = args.query_id or f"{dataset}|{year}|{min(pair)}|{max(pair)}"
    prepared = prepare_query(graph, teacher.model, pair, year, qid=query_id,
                             sampler_config=teacher.sampler)
    result = explain(prepared, seed=args.seed)
    names = {kind: list(index) for kind, index in graph.node_index.items()}
    result["events"] = {edge.event_id: dict(year=edge.timestamp, relation=list(edge.relation),
        source=dict(type=edge.relation[0], id=names[edge.relation[0]][edge.src], index=edge.src),
        target=dict(type=edge.relation[2], id=names[edge.relation[2]][edge.dst], index=edge.dst))
        for edge in prepared.ctx.edges}
    result["provenance"] = dict(dataset=dataset, year=year, author_pair=list(pair),
        query_id=query_id, checkpoint_sha256=teacher.sha256, sampler=asdict(teacher.sampler))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Explanation saved to {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="EIS-FairPath: edge-importance-supervised fair path explanations")
    commands = parser.add_subparsers(dest="command", required=True)
    default_root = Path(__file__).resolve().parents[1]
    for command, help_text in (("verify-data", "Verify complete data with SHA-256"),
                               ("rebuild-data", "Rebuild the three domains from the bundled raw archive"),
                               ("queries", "List fresh example pairs from the full graph"),
                               ("explain", "Optimize a fresh mask and extract H4 fair paths")):
        sub = commands.add_parser(command, help=help_text)
        sub.add_argument("--root", type=Path, default=default_root, help="Repository containing data/ and checkpoints/")
        if command in ("queries", "explain"):
            sub.add_argument("--dataset", choices=tuple(DOMAINS), default="mining")
            sub.add_argument("--year", type=int, help="Prediction year; history uses the preceding five calendar years")
        if command == "queries":
            sub.add_argument("--limit", type=int, default=5)
        if command == "rebuild-data":
            sub.add_argument("--output", type=Path, required=True, help="Directory for the three regenerated JSON files")
        if command == "explain":
            sub.add_argument("--author-a", help="External author ID or the original author name")
            sub.add_argument("--author-b", help="External author ID or the original author name")
            sub.add_argument("--query-id", help="Stable query identifier used to seed evidence probes")
            sub.add_argument("--checkpoint", help="Optional portable teacher checkpoint")
            sub.add_argument("--seed", type=int, default=42, help="Fresh explanation mask seed")
            sub.add_argument("--threads", type=int, default=1)
            sub.add_argument("--output", required=True, help="Destination JSON for this newly computed explanation")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.command == "verify-data":
            _verify(root)
        elif args.command == "rebuild-data":
            from .datasets import build_dblp, write_dataset
            for dataset, _ in DOMAINS.values():
                data = build_dblp(root / "data/raw/dblp.json.gz", dataset.removeprefix("cs_"))
                path = write_dataset(data, args.output / f"{dataset}.json")
                original = root / "data/processed" / path.name
                if _sha256(path) != _sha256(original):
                    raise ValueError(f"Rebuilt data differ from bundled domain: {dataset}")
                print(f"Rebuilt {dataset}: exact SHA-256 match")
        else:
            dataset, default_year = DOMAINS[args.dataset]
            year = args.year if args.year is not None else default_year
            if args.command == "queries":
                if args.limit < 1:
                    raise ValueError("--limit must be positive")
                print(json.dumps(_candidate_pairs(_load_graph(root, dataset), year, args.limit), indent=2))
            else:
                if args.threads < 1:
                    raise ValueError("--threads must be positive")
                _explain(args, root, dataset, year)
    except (ValueError, OSError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0
