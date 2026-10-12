"""AI-completed standard-library CLI; data preparation only, no training."""

import argparse
import json
import sys

from .artifacts import load_artifact, prepare_artifact, save_artifact
from .data import DEFAULT_SEED


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare and verify an exact-text SMS split; no training.")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="write a new verified split JSON")
    prepare.add_argument("--data", required=True, help="path to the original UTF-8 SMS file")
    prepare.add_argument("--output", required=True, help="new JSON path; existing files are refused")
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    check = commands.add_parser("check", help="verify JSON against the original data file")
    check.add_argument("--data", required=True)
    check.add_argument("--artifact", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            artifact = prepare_artifact(args.data, args.seed)
            save_artifact(artifact, args.output)
            restored = load_artifact(args.output, args.data)
            if restored != artifact:
                raise ValueError("artifact readback differs from the prepared payload")
            status = "prepared_and_verified"
        else:
            artifact = load_artifact(args.artifact, args.data)
            status = "verified"
    except UnicodeError:
        print("error: input decoding failed (expected UTF-8)", file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": status,
        "quality": artifact["quality"],
        "summary": artifact["summary"],
        "training_performed": False,
    }, indent=2))
    return 0
