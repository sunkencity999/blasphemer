# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import os
from pathlib import Path


def discover_models_in_directory(search_path: str) -> dict:
    """
    Discover model directories (containing config.json) and GGUF files in the given path.
    
    Args:
        search_path: Directory path to search
        
    Returns:
        Dict with 'models' (list of directory Paths) and 'gguf_files' (list of file Paths)
    """
    search_path = os.path.expanduser(search_path)
    search_path = os.path.abspath(search_path)
    base_path = Path(search_path)
    
    if not base_path.exists():
        return {"models": [], "gguf_files": []}
    
    models = []
    gguf_files = []
    
    # Check if the path itself is a model directory
    if (base_path / "config.json").exists():
        models.append(base_path)
        # Also check for GGUF files in this directory
        try:
            gguf_files.extend(base_path.glob("*.gguf"))
        except PermissionError:
            pass
        return {"models": sorted(models, key=lambda p: p.name), 
                "gguf_files": sorted(gguf_files, key=lambda p: p.name)}
    
    # Search subdirectories and files (one level deep)
    try:
        for item in base_path.iterdir():
            if item.is_dir() and (item / "config.json").exists():
                models.append(item)
            elif item.is_file() and item.suffix == ".gguf":
                gguf_files.append(item)
    except PermissionError:
        pass
    
    return {
        "models": sorted(models, key=lambda p: p.name),
        "gguf_files": sorted(gguf_files, key=lambda p: p.name)
    }
