"""AI-assisted engineering of the submitted SMS reading/splitting workflow.

The split semantics come from the submitted 10/12 implementation. Validation
restores the submitted ID/text/reproducibility checks. The six-field summary
and API boundary checks were completed by AI, not independently by the learner.
"""

import random
from pathlib import Path

from .text import (
    find_duplicate_groups,
    find_label_conflicts,
    group_rows_by_text,
    parse_labeled_line,
    to_text_rows,
)

DEFAULT_SEED = 20261012
SPLIT_NAMES = ("train", "val", "test")
SPLIT_KEYS = tuple(name + "_ids" for name in SPLIT_NAMES)
LABEL_TO_Y = {"ham": 0, "spam": 1}


def _validate_seed(seed):
    if type(seed) is not int:
        raise ValueError("seed must be an integer, not bool")


def _validate_records(records):
    """Enforce the public list-of-immutable-records input contract."""
    if not isinstance(records, list):
        raise ValueError("records must be a list of (row_id, text, y) tuples")
    seen = set()
    for position, record in enumerate(records, 1):
        if not isinstance(record, tuple) or len(record) != 3:
            raise ValueError(f"record {position} must be a three-field tuple")
        row_id, text, y = record
        if type(row_id) is not int or row_id < 1:
            raise ValueError(f"record {position} needs a positive integer ID")
        if row_id in seen:
            raise ValueError(f"duplicate source ID: {row_id}")
        seen.add(row_id)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"record {position} needs nonblank text")
        if type(y) is not int or y not in (0, 1):
            raise ValueError(f"record {position} needs label 0 or 1, not bool")


def _reject_conflicts(records):
    conflicts = find_label_conflicts(records)
    if conflicts:
        raise ValueError(f"label conflicts in {len(conflicts)} exact-text groups")


def check_line(line):
    """Return (ham/spam, original text), or raise one primary ValueError.

    Blank/no-TAB/unknown-label/blank-body is the fixed validation order. Error
    reasons deliberately omit text so CLI diagnostics cannot expose messages.
    """
    if not isinstance(line, str):
        raise ValueError("line must be decoded text")
    if not line.strip():
        raise ValueError("blank line")
    if "\t" not in line:
        raise ValueError("missing TAB separator")
    label, text = parse_labeled_line(line)
    if label not in LABEL_TO_Y:
        raise ValueError("unknown label (expected ham or spam)")
    if not text.strip():
        raise ValueError("blank body")
    return label, text


def read_records(path):
    """Read UTF-8 text with universal newlines, retaining original line IDs.

    Return n_lines, records[(row_id, text, y)], issues[(row_id, reason, line)].
    Issue lines are the unchanged strings yielded by text mode, not raw bytes.
    File-open and decoding failures propagate; partial counts are not returned.
    """
    records, issues = [], []
    n_lines = 0
    with Path(path).open("r", encoding="utf-8") as stream:
        for row_id, line in enumerate(stream, 1):
            n_lines = row_id
            try:
                label, text = check_line(line)
            except ValueError as error:
                issues.append((row_id, str(error), line))
            else:
                records.append((row_id, text, LABEL_TO_Y[label]))
    return {"n_lines": n_lines, "records": records, "issues": issues}


def split_record_ids(records, seed=DEFAULT_SEED):
    """Split exact-text groups; preserve the submitted seed/order semantics.

    One local Random shuffles copies of the ham then spam group pools. Each
    pool needs >=7 groups. Train/val use integer 70%/15%; test gets the rest.
    Output IDs retain class/group expansion order, not sorted row-number order.
    """
    _validate_records(records)
    _validate_seed(seed)
    _reject_conflicts(records)
    groups = group_rows_by_text(to_text_rows(records))
    id_to_y = {row_id: y for row_id, _, y in records}
    pools = {0: [], 1: []}
    for row_ids in groups.values():
        pools[id_to_y[row_ids[0]]].append(row_ids)

    rng = random.Random(seed)
    result = {key: [] for key in SPLIT_KEYS}
    for y in (0, 1):
        pool = pools[y]
        n = len(pool)
        if n < 7:
            raise ValueError(f"label {y} needs at least 7 text groups; got {n}")
        shuffled = pool.copy()
        rng.shuffle(shuffled)
        n_train, n_val = n * 70 // 100, n * 15 // 100
        parts = (
            shuffled[:n_train],
            shuffled[n_train:n_train + n_val],
            shuffled[n_train + n_val:],
        )
        for key, part in zip(SPLIT_KEYS, parts):
            result[key].extend(row_id for group in part for row_id in group)
    return result


def validate_split(records, split):
    """Check exact ID coverage and all three exact-text intersections.

    Raises ValueError on failure. This validates the supplied membership; the
    artifact verifier additionally regenerates the split to verify seed/rules.
    It does not claim detection of approximate text or other related samples.
    """
    _validate_records(records)
    _reject_conflicts(records)
    if not isinstance(split, dict) or set(split) != set(SPLIT_KEYS):
        raise ValueError("split must contain exactly train_ids, val_ids, test_ids")
    for key in SPLIT_KEYS:
        if not isinstance(split[key], list):
            raise ValueError(f"{key} must be an ID list")
        if any(type(row_id) is not int or row_id < 1 for row_id in split[key]):
            raise ValueError(f"{key} needs positive integer IDs, not bool")
    all_ids = [row_id for key in SPLIT_KEYS for row_id in split[key]]
    source_ids = {row_id for row_id, _, _ in records}
    if len(all_ids) != len(records) or set(all_ids) != source_ids:
        raise ValueError("split IDs must cover each source ID exactly once")

    id_to_text = {row_id: text for row_id, text, _ in records}
    texts = {
        name: {id_to_text[row_id] for row_id in split[name + "_ids"]}
        for name in SPLIT_NAMES
    }
    checks = {"ids_cover_source_exactly_once": True}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if not texts[left].isdisjoint(texts[right]):
            raise ValueError(f"exact text overlaps between {left} and {right}")
        checks[f"{left}_{right}_text_disjoint"] = True
    return checks


def summarize_split(records, split):
    """AI-completed six-count summary for train/val/test; never include text."""
    validate_split(records, split)
    by_id = {row_id: (text, y) for row_id, text, y in records}
    summary = {}
    for name in SPLIT_NAMES:
        selected = [by_id[row_id] for row_id in split[name + "_ids"]]
        ham_texts = {text for text, y in selected if y == 0}
        spam_texts = {text for text, y in selected if y == 1}
        summary[name] = {
            "n_records": len(selected),
            "ham_records": sum(y == 0 for _, y in selected),
            "spam_records": sum(y == 1 for _, y in selected),
            "n_groups": len(ham_texts | spam_texts),
            "ham_groups": len(ham_texts),
            "spam_groups": len(spam_texts),
        }
    return summary


def summarize_quality(data):
    """AI-completed numeric-only report; issues and message text stay local."""
    records = data["records"]
    _validate_records(records)
    duplicates = find_duplicate_groups(to_text_rows(records))
    conflicts = find_label_conflicts(records)
    return {
        "n_lines": data["n_lines"],
        "n_records": len(records),
        "n_issues": len(data["issues"]),
        "ham_records": sum(y == 0 for _, _, y in records),
        "spam_records": sum(y == 1 for _, _, y in records),
        "duplicate_groups": len(duplicates),
        "duplicate_records": sum(len(ids) for ids in duplicates.values()),
        "conflict_groups": len(conflicts),
        "conflict_records": sum(len(ids) for ids in conflicts.values()),
    }
