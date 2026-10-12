"""AI-written engineering tests using artificial messages only.

These tests do not count as learner implementation, independent assessment or
learning time. No real SMS file, model, training script or third-party library
is imported or run here.
"""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from sms_classifier.artifacts import (
    file_sha256, get_rules, load_artifact, prepare_artifact,
    save_artifact, verify_artifact,
)
from sms_classifier.data import (
    check_line, read_records, split_record_ids, summarize_quality,
    summarize_split, validate_split,
)
from sms_classifier.text import find_label_conflicts, group_rows_by_text, to_text_rows


REPO_ROOT = Path(__file__).resolve().parents[1]


def artificial_records(groups_per_class=20):
    """Unique IDs with gaps, mixed duplicate sizes, synthetic text only."""
    records = []
    row_id = 2
    for group_number in range(groups_per_class):
        for y in (1, 0):  # Input interleaving differs from the specified RNG order.
            repetitions = 3 if group_number == 0 else 2 if group_number == 1 else 1
            for _ in range(repetitions):
                records.append((row_id, f"ARTIFICIAL_{y}_{group_number}", y))
                row_id += 3
    return records


def write_artificial_file(path, groups_per_class=20, with_bad_line=False):
    lines = []
    for _, text, y in artificial_records(groups_per_class):
        lines.append(("spam" if y else "ham") + "\t" + text + "\n")
    if with_bad_line:
        lines.insert(3, "PRIVATE_SENTINEL_UNKNOWN_LABEL\tPRIVATE_SENTINEL_BODY\n")
    Path(path).write_text("".join(lines), encoding="utf-8")


class ReadingTests(unittest.TestCase):
    def test_normal_bad_lines_and_original_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.txt"
            path.write_bytes(
                b"ham\t  Meet\tsoon  \r\nspam\tOffer\n\nmissing separator\n"
                b"other\tMessage\nham\t \t\n"
            )
            result = read_records(path)
        self.assertEqual(result["n_lines"], 6)
        self.assertEqual(result["records"], [(1, "  Meet\tsoon  ", 0), (2, "Offer", 1)])
        self.assertEqual([issue[0] for issue in result["issues"]], [3, 4, 5, 6])
        self.assertEqual(result["issues"][0], (3, "blank line", "\n"))
        self.assertEqual(len(result["records"]) + len(result["issues"]), result["n_lines"])
        self.assertEqual(result["issues"][3][2], "ham\t \t\n")

    def test_empty_and_last_line_without_newline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.txt"
            path.write_bytes(b"")
            self.assertEqual(read_records(path), {"n_lines": 0, "records": [], "issues": []})
            path.write_text("\nham\tLast line", encoding="utf-8")
            result = read_records(path)
        self.assertEqual(result["n_lines"], 2)
        self.assertEqual(result["records"], [(2, "Last line", 0)])

    def test_invalid_encoding_propagates_instead_of_reporting_partial_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.txt"
            path.write_bytes(b"ham\tGood\nspam\t\xff\n")
            with self.assertRaises(UnicodeDecodeError):
                read_records(path)

    def test_strict_label_blank_priority_and_body_preservation(self):
        self.assertEqual(check_line("spam\t A\tB  \r\n"), ("spam", " A\tB  "))
        for line, expected in (
            (" \t\r\n", "blank line"),
            ("spam", "missing TAB"),
            ("Spam\tBody", "unknown label"),
            ("ham \tBody", "unknown label"),
            ("ham\t \t\n", "blank body"),
        ):
            with self.subTest(line=line), self.assertRaisesRegex(ValueError, expected):
                check_line(line)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.records = artificial_records()

    def test_complete_groups_seed_repeat_and_unchanged_input(self):
        before = self.records.copy()
        split = split_record_ids(self.records)
        self.assertEqual(split, split_record_ids(self.records))
        self.assertEqual(self.records, before)
        self.assertTrue(all(validate_split(self.records, split).values()))
        self.assertEqual(set(split), {"train_ids", "val_ids", "test_ids"})
        source = {row_id for row_id, _, _ in self.records}
        all_ids = split["train_ids"] + split["val_ids"] + split["test_ids"]
        self.assertEqual(len(all_ids), len(source))
        self.assertEqual(set(all_ids), source)
        summary = summarize_split(self.records, split)
        for label in ("ham", "spam"):
            self.assertEqual([summary[name][label + "_groups"] for name in ("train", "val", "test")], [14, 3, 3])
        owner = {row_id: name for name, ids in split.items() for row_id in ids}
        for ids in group_rows_by_text(to_text_rows(self.records)).values():
            self.assertEqual(len({owner[row_id] for row_id in ids}), 1)

    def test_conflicts_rejected_before_insufficient_class_groups(self):
        with self.assertRaisesRegex(ValueError, "label conflicts"):
            split_record_ids([(1, "same", 0), (2, "same", 1)])

    def test_empty_and_small_class_pool_rejected(self):
        for records in ([], artificial_records(6)):
            with self.subTest(size=len(records)), self.assertRaisesRegex(ValueError, "at least 7"):
                split_record_ids(records)
        split = split_record_ids(artificial_records(7))
        self.assertTrue(all(split.values()))

    def test_duplicate_source_id_invalid_labels_and_bool_ids_rejected(self):
        for records in (
            [(1, "A", 0), (1, "B", 1)],
            [(True, "A", 0)],
            [(1, "A", True)],
            [(1, "A", 2)],
            [(0, "A", 0)],
        ):
            with self.subTest(records=records), self.assertRaises(ValueError):
                split_record_ids(records)
        with self.assertRaises(ValueError):
            split_record_ids(self.records, seed=True)

    def test_membership_extra_missing_duplicate_and_boolean_ids_rejected(self):
        good = split_record_ids(self.records)
        changes = []
        missing = copy.deepcopy(good)
        missing["train_ids"].pop()
        changes.append(missing)
        stranger = copy.deepcopy(good)
        stranger["train_ids"][0] = 999999
        changes.append(stranger)
        duplicate = copy.deepcopy(good)
        duplicate["train_ids"].append(duplicate["train_ids"][0])
        changes.append(duplicate)
        balanced_duplicate = copy.deepcopy(good)
        balanced_duplicate["train_ids"][0] = balanced_duplicate["train_ids"][1]
        changes.append(balanced_duplicate)
        boolean = copy.deepcopy(good)
        boolean["train_ids"][0] = True
        changes.append(boolean)
        for split in changes:
            with self.subTest(split=split), self.assertRaises(ValueError):
                validate_split(self.records, split)

    def test_exact_text_crossing_is_detected_even_when_ids_are_unique(self):
        good = split_record_ids(self.records)
        repeated_ids = next(ids for ids in group_rows_by_text(to_text_rows(self.records)).values() if len(ids) > 1)
        moved = repeated_ids[0]
        source_key = next(key for key, ids in good.items() if moved in ids)
        target_key = next(key for key in good if key != source_key)
        good[source_key].remove(moved)
        good[target_key].append(moved)
        with self.assertRaisesRegex(ValueError, "exact text overlaps"):
            validate_split(self.records, good)

    def test_summary_counts_records_and_groups_differently(self):
        records = [(11, "Meet", 0), (12, "Meet", 0), (13, "Prize", 1)]
        summary = summarize_split(records, {"train_ids": [11, 12], "val_ids": [13], "test_ids": []})
        self.assertEqual(summary["train"], {
            "n_records": 2, "ham_records": 2, "spam_records": 0,
            "n_groups": 1, "ham_groups": 1, "spam_groups": 0,
        })
        self.assertEqual(summary["val"], {
            "n_records": 1, "ham_records": 0, "spam_records": 1,
            "n_groups": 1, "ham_groups": 0, "spam_groups": 1,
        })
        self.assertEqual(set(summary["test"].values()), {0})

    def test_quality_report_counts_conflict_groups_not_incorrect_labels(self):
        records = [(1, "Same", 0), (2, "Same", 1), (3, "Same", 1), (4, "Other", 0)]
        result = summarize_quality({"n_lines": 5, "records": records, "issues": [(5, "blank line", "\n")]})
        self.assertEqual(result, {
            "n_lines": 5, "n_records": 4, "n_issues": 1,
            "ham_records": 2, "spam_records": 2,
            "duplicate_groups": 1, "duplicate_records": 3,
            "conflict_groups": 1, "conflict_records": 3,
        })
        self.assertEqual(find_label_conflicts(records), {"Same": [1, 2, 3]})


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.data_path = self.directory / "artificial.txt"
        write_artificial_file(self.data_path, with_bad_line=True)
        self.artifact = prepare_artifact(self.data_path)
        self.output = self.directory / "split.json"

    def test_save_load_preserves_ids_and_order_without_message_content(self):
        save_artifact(self.artifact, self.output)
        restored = load_artifact(self.output, self.data_path)
        self.assertEqual(restored, self.artifact)
        payload = self.output.read_text(encoding="utf-8")
        self.assertNotIn("ARTIFICIAL_", payload)
        self.assertNotIn("PRIVATE_SENTINEL", payload)
        self.assertEqual(restored["source_sha256"], file_sha256(self.data_path))
        self.assertEqual(restored["quality"]["n_issues"], 1)

    def test_existing_output_is_never_overwritten(self):
        self.output.write_text("keep this file", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            save_artifact(self.artifact, self.output)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "keep this file")

    def test_source_hash_change_is_rejected(self):
        save_artifact(self.artifact, self.output)
        with self.data_path.open("a", encoding="utf-8") as stream:
            stream.write("ham\tAdditional artificial group\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            load_artifact(self.output, self.data_path)

    def test_tampered_metadata_rules_counts_seed_and_id_order_rejected(self):
        mutations = [
            lambda item: item.update(schema_version=True),
            lambda item: item.update(schema_version=99),
            lambda item: item.update(rules_version="different"),
            lambda item: item.update(python_version="0.0.0"),
            lambda item: item.update(source_sha256="f" * 64),
            lambda item: item.update(seed=True),
            lambda item: item.update(seed=item["seed"] + 1),
            lambda item: item["rules"].update(class_order=[1, 0]),
            lambda item: item["rules"].update(minimum_groups_per_class=True),
            lambda item: item["summary"]["train"].update(n_records=0),
            lambda item: item["quality"].update(n_issues=True),
            lambda item: item["quality"].update(n_lines=99999),
            lambda item: item["split"]["train_ids"].reverse(),
            lambda item: item.update(unexpected="PRIVATE_SENTINEL"),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                changed = copy.deepcopy(self.artifact)
                mutate(changed)
                with self.assertRaises(ValueError):
                    verify_artifact(changed, self.data_path)

    def test_public_save_rejects_text_in_numeric_summary(self):
        self.artifact["summary"]["train"]["n_records"] = "PRIVATE_SENTINEL"
        with self.assertRaises(ValueError):
            save_artifact(self.artifact, self.output)
        self.assertFalse(self.output.exists())

    def test_duplicate_json_fields_and_nonfinite_json_rejected(self):
        for payload in ('{"schema_version":1,"schema_version":1}', '{"schema_version":NaN}'):
            self.output.write_text(payload, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_artifact(self.output, self.data_path)

    def test_rule_description_is_fresh_and_does_not_mutate_global_policy(self):
        rules = get_rules()
        rules["class_order"].reverse()
        self.assertEqual(get_rules()["class_order"], [0, 1])


class CLITests(unittest.TestCase):
    def test_new_process_prepare_check_refuse_overwrite_and_no_text_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "artificial input.txt"
            output = root / "new split.json"
            write_artificial_file(source, with_bad_line=True)
            environment = os.environ.copy()
            # Module lookup is explicit; data paths work from an unrelated cwd.
            environment["PYTHONPATH"] = str(REPO_ROOT)
            base = [sys.executable, "-B", "-m", "sms_classifier"]
            prepare = subprocess.run(
                base + ["prepare", "--data", str(source), "--output", str(output)],
                cwd=root, env=environment, text=True, capture_output=True, timeout=30,
            )
            self.assertEqual(prepare.returncode, 0, prepare.stderr)
            report = json.loads(prepare.stdout)
            self.assertEqual(report["status"], "prepared_and_verified")
            self.assertFalse(report["training_performed"])
            self.assertEqual(report["quality"]["n_issues"], 1)
            self.assertNotIn("PRIVATE_SENTINEL", prepare.stdout + prepare.stderr)
            self.assertNotIn("ARTIFICIAL_", prepare.stdout + prepare.stderr)
            before = output.read_bytes()
            check = subprocess.run(
                base + ["check", "--data", str(source), "--artifact", str(output)],
                cwd=root, env=environment, text=True, capture_output=True, timeout=30,
            )
            self.assertEqual(check.returncode, 0, check.stderr)
            self.assertEqual(json.loads(check.stdout)["status"], "verified")
            overwrite = subprocess.run(
                base + ["prepare", "--data", str(source), "--output", str(output)],
                cwd=root, env=environment, text=True, capture_output=True, timeout=30,
            )
            self.assertEqual(overwrite.returncode, 2)
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
