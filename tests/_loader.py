"""Loads pacmap_animation_mnist.cli.py as a module for tests.

The file can't be imported with a normal `import` statement because its
name (dots before the extension) isn't a valid dotted module path, so we
load it directly from its file path instead.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_PATH = REPO_ROOT / "pacmap_animation_mnist.cli.py"

# The entry shim imports the `pacmap_cli` package by name (as it would when
# run directly, where Python puts the script's own directory on sys.path);
# loading it via importlib doesn't do that automatically, so add it here.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location("pacmap_cli_entry", CLI_PATH)
cli = importlib.util.module_from_spec(_spec)
sys.modules["pacmap_cli_entry"] = cli
_spec.loader.exec_module(cli)
