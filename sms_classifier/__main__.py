"""AI-completed package entry point; importing the package runs no workflow."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
