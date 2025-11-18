"""Main script entry point - delegates to test.py for all tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path if running directly
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import and run the main test suite from test.py
if __name__ == "__main__":
    import total_test

    total_test.main()
