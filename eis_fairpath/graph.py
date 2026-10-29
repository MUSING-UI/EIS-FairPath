"""Causal bibliographic snapshots and fixed next-year collaboration candidates."""
from dataclasses import dataclass
import hashlib
import itertools
import random

import torch

RELATIONS = [("author", "writes", "paper"), ("paper", "written_by", "author"),
             ("paper", "published_in", "venue"), ("venue", "publishes", "paper"),
             ("author", "collaborates", "author")]


def temporal_split(years):
    years = sorted(set(years))
    if len(years) < 5:
        raise ValueError("At least five distinct years are needed for three nonempty splits")
    a, b = int(len(years) * .6), int(len(years) * .8)
    return {"train": years[:a], "valid": years[a:b], "test": years[b:]}


@dataclass
class Snapshot:
    year: int
    x_dict: dict
    edge_index_dict: dict
    edge_event_index: dict
    event_ids: list
    active: dict

    @property
    def num_events(self):
        return len(self.event_ids)

    def to(self, device):
        return Snapshot(self.year, {k: v.to(device) for k, v in self.x_dict.items()},
                        {k: v.to(device) for k, v in self.edge_index_dict.items()},
                        {k: v.to(device) for k, v in self.edge_event_index.items()},
                        self.event_ids, {k: v.to(device) for k, v in self.active.items()})


class ExperimentGraph:
    def __init__(self, data, input_dim=16):
        self.dataset = data["dataset"]
        self.years = sorted(data["years"])
        self.papers = data["papers"]
        self.node_index, self.first_seen, self.features = {}, {}, {}
        self.metadata = (["author", "paper", "venue"], RELATIONS)
        nodes = {"author": {}, "paper": {}, "venue": {}}
        ids = set()
        for p in self.papers:
            if p["id"] in ids or p["year"] not in self.years:
                raise ValueError("Duplicate paper or year outside declared dataset")
            ids.add(p["id"])
            for typ, names in [("author", p["authors"]), ("paper", [p["id"]]), ("venue", [p["venue"]])]:
                for name in names:
                    nodes[typ][name] = min(nodes[typ].get(name, p["year"]), p["year"])
        for typ, entries in nodes.items():
            ordered = sorted(entries)
            self.node_index[typ] = {name: i for i, name in enumerate(ordered)}
            self.first_seen[typ] = torch.tensor([entries[n] for n in ordered])
            # Fixed identity features: no future attributes, vocabulary fit, or trainable ID table.
            values = [[1.] + [(b / 127.5 - 1) for b in hashlib.shake_256(
                (typ + ":" + name).encode()).digest(input_dim - 1)] for name in ordered]
            self.features[typ] = torch.tensor(values, dtype=torch.float32)
        self.events = {year: [] for year in self.years}
        self.pairs = {year: set() for year in self.years}
        for p in self.papers:
            y = p["year"]
            for a in sorted(set(p["authors"])):
                self.events[y].append(("writes", a, p["id"]))
            self.events[y].append(("published_in", p["id"], p["venue"]))
            for a, b in itertools.combinations(sorted(set(p["authors"])), 2):
                self.pairs[y].add(tuple(sorted((self.node_index["author"][a], self.node_index["author"][b]))))
        author_names = list(self.node_index["author"])
        for year in self.years:
            self.events[year] += [("collaborates", author_names[a], author_names[b])
                                  for a, b in sorted(self.pairs[year])]
            self.events[year] = sorted(set(self.events[year]))
        self._cache = {}

    def empty_snapshot(self, year):
        active = {typ: self.first_seen[typ] <= year for typ in self.features}
        return Snapshot(year, {typ: x * active[typ][:, None] for typ, x in self.features.items()},
                        {r: torch.empty((2, 0), dtype=torch.long) for r in RELATIONS},
                        {r: torch.empty(0, dtype=torch.long) for r in RELATIONS}, [], active)

    def snapshot(self, year, removed=frozenset()):
        snap = self.empty_snapshot(year)
        indices = {r: [] for r in RELATIONS}
        event_indices = {r: [] for r in RELATIONS}
        for relation, a, b in self.events[year]:
            event_id = hashlib.sha256(f"{year}|{relation}|{a}|{b}".encode()).hexdigest()
            if event_id in removed:
                continue
            event_index = len(snap.event_ids)
            snap.event_ids.append(event_id)
            if relation == "writes":
                types = ("author", "paper"); routes = [RELATIONS[0], RELATIONS[1]]
            elif relation == "published_in":
                types = ("paper", "venue"); routes = [RELATIONS[2], RELATIONS[3]]
            else:
                types = ("author", "author"); routes = [RELATIONS[4], RELATIONS[4]]
            u, v = self.node_index[types[0]][a], self.node_index[types[1]][b]
            for route, edge in zip(routes, [(u, v), (v, u)]):
                indices[route].append(edge)
                event_indices[route].append(event_index)
        snap.edge_index_dict = {r: torch.tensor(e, dtype=torch.long).reshape(-1, 2).T.contiguous()
                                for r, e in indices.items()}
        snap.edge_event_index = {r: torch.tensor(e, dtype=torch.long) for r, e in event_indices.items()}
        return snap

    def history(self, query_time, window=5, device="cpu", removed=frozenset()):
        times = [y for y in self.years if y < query_time][-window:]
        if not times:
            return [self.empty_snapshot(query_time - 1).to(device)]
        result = []
        for year in times:
            key = (year, str(device))
            if removed:
                result.append(self.snapshot(year, removed).to(device))
            else:
                if key not in self._cache:
                    self._cache[key] = self.snapshot(year).to(device)
                result.append(self._cache[key])
        return result

    def queries(self, year, seed=42, max_positives=256, negatives=49):
        known = set(torch.where(self.first_seen["author"] < year)[0].tolist())
        positives = sorted(self.pairs[year])
        neighbors = {u: set() for u in known}
        for u, v in positives:
            if u in known: neighbors[u].add(v)
            if v in known: neighbors[v].add(u)
        eligible = [(u, v) for u, v in positives if u in known and v in known]
        rng = random.Random(seed + year)
        rng.shuffle(eligible)
        groups = []
        for u, v in eligible[:max_positives]:
            if rng.random() < .5: u, v = v, u
            pool = sorted(known - {u} - neighbors[u])
            if len(pool) >= negatives:
                groups.append([u, v, *rng.sample(pool, negatives)])
        return {"year": year, "groups": groups, "all_positive_pairs": len(positives),
                "eligible_positive_pairs": len(eligible), "cold_start_excluded": len(positives) - len(eligible),
                "sampled_positive_pairs": len(groups), "negative_count": negatives,
                "seed": seed, "task": "collaboration_occurs_next_year_in_sampled_corpus"}


def load_graph(path, *, input_dim=16):
    """Load one processed publication graph from an explicit JSON path."""
    import json
    from pathlib import Path
    return ExperimentGraph(json.loads(Path(path).read_text(encoding="utf-8-sig")), input_dim=input_dim)


def build_snapshot(graph_or_data, year, *, input_dim=16, removed=frozenset()):
    """Construct one annual snapshot from a graph or processed data mapping."""
    graph = graph_or_data if isinstance(graph_or_data, ExperimentGraph) else ExperimentGraph(graph_or_data, input_dim=input_dim)
    return graph.snapshot(year, removed=removed)
