"""Trainable causal, physical-event HGT prediction models."""
from .structure import StructureAwareHGTTeacher
from .count_aware import CountAwareHGTTeacher
from .teacher_v2 import CausalHGTTeacher
from .adapter import FixedSupportTeacher

__all__ = ["StructureAwareHGTTeacher", "CountAwareHGTTeacher", "CausalHGTTeacher", "FixedSupportTeacher"]
