"""AI-assisted extraction of the existing 10/11 text helpers, without demos.

Exact text equality and original row IDs are preserved. Packaging and small
engineering changes are AI assistance, not evidence of independent learning.
"""


def parse_labeled_line(line):
    """Remove trailing CR/LF only, then split at the first TAB."""
    clean_line = line.rstrip("\r\n")
    label, text = clean_line.split("\t", 1)
    return label, text


def to_text_rows(records):
    """Project (original_id, text, y) records without renumbering."""
    return [(row_id, text) for row_id, text, _ in records]


def group_rows_by_text(records):
    """Return every exact-text group, in order of first occurrence."""
    groups = {}
    for row_id, text in records:
        if text not in groups:
            groups[text] = []
        groups[text].append(row_id)
    return groups


def find_duplicate_groups(records):
    """Return groups with more than one row, including their first row."""
    return {
        text: ids for text, ids in group_rows_by_text(records).items()
        if len(ids) > 1
    }


def has_label_conflict(labels):
    """For an existing nonempty group, report multiple distinct labels."""
    return len(set(labels)) > 1


def find_label_conflicts(records):
    """Return exact-text conflicts and all their original IDs.

    AI engineering completion groups once instead of rescanning every record;
    the existing helper's conflict definition and returned IDs are unchanged.
    """
    groups = group_rows_by_text(to_text_rows(records))
    id_to_y = {row_id: y for row_id, _, y in records}
    return {
        text: ids for text, ids in groups.items()
        if has_label_conflict([id_to_y[row_id] for row_id in ids])
    }
