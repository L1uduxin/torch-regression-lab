"""SMS data preparation, packaged with AI assistance; not a student assessment.

This package performs no model training or feature fitting.
"""

from .data import (
    check_line,
    read_records,
    split_record_ids,
    summarize_split,
    validate_split,
)

__all__ = [
    "check_line", "read_records", "split_record_ids", "summarize_split",
    "validate_split",
]
