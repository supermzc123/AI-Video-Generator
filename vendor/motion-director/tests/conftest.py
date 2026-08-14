from __future__ import annotations

import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_minimax_h3_motion_director_testpkg"


# Load the repository as a synthetic package without executing the ComfyUI
# custom-node root __init__.py.  This keeps unit tests independent of a full
# ComfyUI install while preserving the plugin's relative imports.
if PACKAGE not in sys.modules:
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(ROOT)]
    package.__package__ = PACKAGE
    sys.modules[PACKAGE] = package


if "folder_paths" not in sys.modules:
    try:
        __import__("folder_paths")
    except ImportError:
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.get_output_directory = lambda: str(ROOT / ".test-output")
        folder_paths.get_input_directory = lambda: str(ROOT)
        folder_paths.get_annotated_filepath = lambda path: str(ROOT / str(path))
        folder_paths.get_folder_paths = lambda _name: []
        sys.modules["folder_paths"] = folder_paths


if "comfy" not in sys.modules:
    try:
        __import__("comfy.utils")
    except ImportError:
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        comfy_utils = types.ModuleType("comfy.utils")
        comfy_utils.common_upscale = lambda tensor, *_args, **_kwargs: tensor
        comfy.utils = comfy_utils
        sys.modules["comfy"] = comfy
        sys.modules["comfy.utils"] = comfy_utils
