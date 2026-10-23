"""Console entry point. Lets the tool run as `python -m ai_model_picker`."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
