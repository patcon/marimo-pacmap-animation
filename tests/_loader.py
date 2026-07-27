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

_spec = importlib.util.spec_from_file_location("pacmap_cli", CLI_PATH)
cli = importlib.util.module_from_spec(_spec)
sys.modules["pacmap_cli"] = cli
_spec.loader.exec_module(cli)
