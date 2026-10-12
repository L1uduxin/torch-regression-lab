"""AI-completed local artifact persistence, hashing and verification.

These engineering functions were authorized as AI completion. They are not
evidence that the learner implemented or passed the saving/loading concepts.
Artifacts contain IDs/counts/configuration only, never source SMS or issues.
"""

import hashlib
import json
import platform
from pathlib import Path

from .data import (
    DEFAULT_SEED, SPLIT_KEYS, SPLIT_NAMES, _validate_seed, read_records, split_record_ids,
    summarize_quality, summarize_split, validate_split,
)

SCHEMA_VERSION = 1
RULES_VERSION = "sms-exact-groups-v1"
ARTIFACT_KEYS = {
    "schema_version", "rules_version", "seed", "source_sha256",
    "python_version", "rules", "quality", "summary", "split",
}
SUMMARY_KEYS = {"n_records", "ham_records", "spam_records", "n_groups", "ham_groups", "spam_groups"}
QUALITY_KEYS = {
    "n_lines", "n_records", "n_issues", "ham_records", "spam_records",
    "duplicate_groups", "duplicate_records", "conflict_groups", "conflict_records",
}


def get_rules():
    """Return a fresh serializable description of the actual implementation."""
    return {
        "encoding": "utf-8",
        "newline_mode": "universal",
        "row_ids": "one-based physical lines; never renumber valid rows",
        "parser": "rstrip CR/LF; split first TAB; preserve remaining text",
        "labels": {"ham": 0, "spam": 1},
        "invalid_lines": "one issue per line; continue; keep source unchanged",
        "blank_check": "whitespace-only line or body is invalid",
        "duplicate_policy": "retain all; group by exact original text",
        "label_conflicts": "reject before splitting",
        "initial_group_order": "first occurrence in input records",
        "class_order": [0, 1],
        "rng": "one local random.Random(seed); shuffle class-pool copies",
        "minimum_groups_per_class": 7,
        "train_group_numerator": 70,
        "val_group_numerator": 15,
        "denominator": 100,
        "rounding": "integer floor; test receives remainder",
        "output_id_order": "class then shuffled group then original within-group order",
        "features_or_training": "none",
    }


def file_sha256(path):
    """Hash the exact source bytes without changing the source file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    # Unlike Python ==, canonical JSON distinguishes true from integer 1.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read_unchanged_source(data_path):
    before = file_sha256(data_path)
    data = read_records(data_path)
    if file_sha256(data_path) != before:
        raise ValueError("source changed while being read")
    return before, data


def prepare_artifact(data_path, seed=DEFAULT_SEED):
    """Read, split twice, check input/IDs/text, and build a numeric-only artifact."""
    _validate_seed(seed)
    digest, data = _read_unchanged_source(data_path)
    records = data["records"]
    records_before = records.copy()
    split = split_record_ids(records, seed)
    if records != records_before:
        raise ValueError("split modified the input records")
    repeated = split_record_ids(records, seed)
    if repeated != split or records != records_before:
        raise ValueError("split reproducibility or input-preservation check failed")
    validate_split(records, split)
    validate_split(records, repeated)
    return {
        "schema_version": SCHEMA_VERSION,
        "rules_version": RULES_VERSION,
        "seed": seed,
        "source_sha256": digest,
        "python_version": platform.python_version(),
        "rules": get_rules(),
        "quality": summarize_quality(data),
        "summary": summarize_split(records, split),
        "split": split,
    }


def _check_metadata(artifact):
    if not isinstance(artifact, dict) or set(artifact) != ARTIFACT_KEYS:
        raise ValueError("artifact schema fields do not match schema version 1")
    if type(artifact["schema_version"]) is not int or artifact["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported artifact schema version")
    if artifact["rules_version"] != RULES_VERSION:
        raise ValueError("unsupported split rules version")
    _validate_seed(artifact["seed"])
    digest = artifact["source_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("source_sha256 must be a lowercase SHA256 digest")
    if artifact["python_version"] != platform.python_version():
        raise ValueError("Python version differs; cross-version reproduction is not claimed")
    if _canonical(artifact["rules"]) != _canonical(get_rules()):
        raise ValueError("stored split rules do not match the implemented rules")
    # Enforce numeric-only public payloads even when this API is called directly.
    def check_counts(value, keys, name):
        if not isinstance(value, dict) or set(value) != keys:
            raise ValueError(f"invalid {name} fields")
        if any(type(count) is not int or count < 0 for count in value.values()):
            raise ValueError(f"{name} counts must be nonnegative integers, not bool")

    check_counts(artifact["quality"], QUALITY_KEYS, "quality")
    summary = artifact["summary"]
    if not isinstance(summary, dict) or set(summary) != set(SPLIT_NAMES):
        raise ValueError("summary must contain exactly train, val, test")
    for name in SPLIT_NAMES:
        check_counts(summary[name], SUMMARY_KEYS, name + " summary")
    split = artifact["split"]
    if not isinstance(split, dict) or set(split) != set(SPLIT_KEYS):
        raise ValueError("invalid split ID fields")
    for key in SPLIT_KEYS:
        if not isinstance(split[key], list) or any(type(row_id) is not int or row_id < 1 for row_id in split[key]):
            raise ValueError(f"{key} must contain positive integer IDs, not bool")


def verify_artifact(artifact, data_path):
    """Verify schema, current source hash, IDs/text, summaries and exact ID order.

    The source path is required, and current Python must match the saved version.
    A content hash detects drift, not authorship or cryptographic authenticity.
    """
    _check_metadata(artifact)
    digest = file_sha256(data_path)
    if digest != artifact["source_sha256"]:
        raise ValueError("source SHA256 differs from the saved artifact")
    checked_digest, data = _read_unchanged_source(data_path)
    if checked_digest != digest:
        raise ValueError("source changed during verification")
    records = data["records"]
    checks = validate_split(records, artifact["split"])
    expected = split_record_ids(records, artifact["seed"])
    if artifact["split"] != expected:
        raise ValueError("saved ID membership/order does not match seed and rules")
    summary = summarize_split(records, artifact["split"])
    quality = summarize_quality(data)
    if _canonical(artifact["summary"]) != _canonical(summary):
        raise ValueError("saved split summary does not match the current source")
    if _canonical(artifact["quality"]) != _canonical(quality):
        raise ValueError("saved quality counts do not match the current source")
    return {"status": "verified", "checks": checks, "quality": quality, "summary": summary}


def save_artifact(artifact, output_path):
    """Save a prepared artifact with exclusive creation; never overwrite.

    Call prepare_artifact first. Full source-bound verification is performed by
    load_artifact/verify_artifact; the CLI always verifies the written artifact.
    """
    _check_metadata(artifact)
    payload = json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output_path = Path(output_path)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key in artifact")
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError("non-finite JSON number in artifact")


def load_artifact(artifact_path, data_path):
    """Load and fully verify against the required original data file."""
    with Path(artifact_path).open("r", encoding="utf-8") as stream:
        artifact = json.load(stream, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    verify_artifact(artifact, data_path)
    return artifact
