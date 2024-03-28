import importlib
from pathlib import Path

import pytest

base_dir = Path(__file__).parents[1]
# Skip these directories and files
skip_dirs_files = {"__pycache__", "configs", "tests"}


def discover_modules(path, skip_dirs_files):
    """
    Recursively discovers Python modules in the given path, excluding specified directories and files.
    """
    modules = []
    for item in path.iterdir():
        if item.name in skip_dirs_files or item.name.startswith("_"):
            continue
        if item.is_dir():
            modules += discover_modules(item, skip_dirs_files)
        elif item.suffix == ".py":
            module_path = (
                item.relative_to(base_dir.parent)
                .with_suffix("")
                .as_posix()
                .replace("/", ".")
            )
            modules.append(module_path)
    return modules


# Perform module discovery directly
modules_to_test = discover_modules(base_dir, skip_dirs_files)


def import_module(module_path):
    """
    Attempts to import a module and returns an error message if it fails.
    """
    try:
        importlib.import_module(module_path)
        return None
    except Exception as e:
        return str(e)


@pytest.mark.parametrize("module_path", modules_to_test)
def test_module_import(module_path):
    """
    Test each module import, failing if any module cannot be imported.
    """
    error_message = import_module(module_path)
    assert error_message is None, f"Failed to import {module_path}: {error_message}"
