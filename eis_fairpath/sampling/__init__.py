"""Deterministic chronological fixed physical-event support."""
from dataclasses import dataclass
from .types import SamplerConfig, SampledQuery, SamplingLimitError, StateKey, support_digest
from .sampler import HistoryIndex, sample_query


@dataclass(frozen=True)
class Query:
    query_id: str
    source: tuple[str, int]
    target: tuple[str, int]
    time: int


def sample_pair(history_or_index, author_pair, year, *, qid="query", dataset="", config=None):
    """Sample an unordered pair of author indices with the production fixed quota policy."""
    source, target = author_pair
    source = ("author", source) if isinstance(source, int) else source
    target = ("author", target) if isinstance(target, int) else target
    return sample_query(history_or_index, Query(qid, source, target, year), dataset=dataset,
                        config=SamplerConfig() if config is None else config, quota_mode="fixed")


__all__ = ["SamplerConfig", "SampledQuery", "SamplingLimitError", "StateKey", "support_digest",
           "HistoryIndex", "sample_query", "sample_pair", "Query"]
