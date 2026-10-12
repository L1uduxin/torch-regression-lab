# Changes and contribution boundaries

## 2026-10-12 — SMS data preparation

The submitted reading, exact-text grouping and group-stratified splitting code is now a reusable `sms_classifier` package. No classifier or fitted text representation is included.

| Origin | Included work |
|---|---|
| Existing learner submissions, with the recorded teaching/help retained locally | First-TAB parsing, original physical line IDs, line validation and issue collection, duplicate/conflict definitions, ham/spam group splitting, ID coverage, pairwise text isolation, repeatability and input-preservation checks |
| AI-assisted extraction and engineering | Package-relative imports, callable boundaries, type/ID/seed checks, efficient conflict lookup, restored checks previously spread across submissions, public documentation and synthetic regression tests |
| Explicitly authorized AI completion | Per-split six-count summaries; source SHA256 and rule metadata; exclusive JSON creation; verified loading; command-line entry points and their failure handling |
| Existing local filename change integrated | `11/basic_test_01.py` is the parser's current name; `basic_text_01.py` remains as a compatibility import for existing callers |

AI completion and successful automated checks do not establish independent learner authorship or understanding. Summary denominators, JSON metadata, hashing, verified loading, and API checks remain subject to learner review. Original submissions are preserved locally; they are not republished as personal learning evidence.

The new package has no dependence on dated external Python folders. Existing historical regression and clustering scripts retain their prior scope and local asset requirements; they were not retrained or turned into a portable full training system in this change.

## Next project steps

1. Review the added summary and persistence contracts; confirm the data card and frozen split.
2. Specify training-only vocabulary/feature fitting, validation use, and final-test isolation before implementing a classifier.
3. Add a majority-class reference and the first classification baseline only after those prerequisites are understood and checked.
4. Later compare models under a shared split, preprocessing and declared budget. Do not infer model superiority from different initializations or update counts.

These are project dependencies, not claims that the steps are already implemented.
