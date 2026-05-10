"""Entry point for ``python -m baremetal_agent``."""

import sys

from baremetal_agent.cli import main

if __name__ == "__main__":
    sys.exit(main())
