"""oddball anomaly detection datasets."""

from .datasets import (
    clear_cache,
    get_cache_location,
    list_available,
    load,
    split_by_label,
)
from .enums import Dataset

__all__ = [
    "Dataset",
    "clear_cache",
    "get_cache_location",
    "list_available",
    "load",
    "split_by_label",
]
__version__ = "1.0.0"
