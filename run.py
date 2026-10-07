"""Entry point: python run.py"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src import model  # noqa: E402


def main() -> int:
    model.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
