# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import os
from pathlib import Path
from typing import List, Dict

from huggingface_hub import HfApi
from heretic.utils import print


def discover_models_in_directory(search_paths: str | List[str]) -> dict:
    """
    Discover model directories (containing config.json) and GGUF files in the given path(s).
    
    Args:
        search_paths: Directory path or list of paths to search
        
    Returns:
        Dict with 'models' (list of directory Paths) and 'gguf_files' (list of file Paths)
    """
    if isinstance(search_paths, str):
        search_paths = [search_paths]
        
    all_models = []
    all_gguf_files = []
    
    for search_path in search_paths:
        search_path = os.path.expanduser(search_path)
        search_path = os.path.abspath(search_path)
        base_path = Path(search_path)
        
        if not base_path.exists():
            continue
        
        # Check if the path itself is a model directory
        if (base_path / "config.json").exists():
            all_models.append(base_path)
            # Also check for GGUF files in this directory
            try:
                all_gguf_files.extend(base_path.glob("*.gguf"))
            except PermissionError:
                pass
            continue # If it's a model dir, we don't search subdirs (usually) or maybe we do? 
            # If it's a model dir, it likely contains weights, not other model dirs.
            
        # Search subdirectories and files (one level deep)
        try:
            for item in base_path.iterdir():
                if item.is_dir() and (item / "config.json").exists():
                    all_models.append(item)
                elif item.is_file() and item.suffix == ".gguf":
                    all_gguf_files.append(item)
        except PermissionError:
            pass
            
    # Deduplicate based on absolute path
    unique_models = sorted(list(set(all_models)), key=lambda p: p.name)
    unique_gguf = sorted(list(set(all_gguf_files)), key=lambda p: p.name)
    
    return {
        "models": unique_models,
        "gguf_files": unique_gguf
    }


def search_huggingface_models(query: str, limit: int = 10) -> List[Dict]:
    """
    Search for models on Hugging Face Hub.
    
    Args:
        query: Search query string
        limit: Maximum number of results to return
        
    Returns:
        List of dictionaries containing model info (id, likes, downloads, tags)
    """
    from requests.exceptions import SSLError, ConnectionError
    
    api = HfApi()
    try:
        models = api.list_models(
            search=query,
            limit=limit,
            filter=["text-generation", "transformers"],
            sort="downloads",
            direction=-1,
        )
        
        results = []
        for model in models:
            results.append({
                "id": model.id,
                "likes": model.likes,
                "downloads": model.downloads,
                "tags": model.tags,
                "pipeline_tag": model.pipeline_tag,
            })
        
        return results

    except (SSLError, ConnectionError) as e:
        if "CERTIFICATE_VERIFY_FAILED" in str(e):
            print("[red]SSL Certificate Verification Failed.[/]")
            print("[yellow]SSL verification is disabled by default, but connection still failed.[/]")
            print("Check your network connection or proxy settings.")
        else:
             print(f"[red]Network Error: {e}[/]")
        return []
    except Exception as e:
        print(f"[red]Error searching models: {e}[/]")
        return []


def list_recent_trending_models(limit: int = 10) -> List[Dict]:
    """
    List trending text-generation models on Hugging Face Hub.
    
    Args:
        limit: Maximum number of results to return
        
    Returns:
        List of dictionaries containing model info
    """
    from requests.exceptions import SSLError, ConnectionError
    
    api = HfApi()
    try:
        models = api.list_models(
            limit=limit,
            filter=["text-generation", "transformers"],
            sort="trending_score",
            direction=-1,
        )
        
        results = []
        for model in models:
            results.append({
                "id": model.id,
                "likes": model.likes,
                "downloads": model.downloads,
                "tags": model.tags,
                "pipeline_tag": model.pipeline_tag,
            })
            
        return results
        
    except (SSLError, ConnectionError) as e:
        if "CERTIFICATE_VERIFY_FAILED" in str(e):
            print("[red]SSL Certificate Verification Failed.[/]")
            print("[yellow]SSL verification is disabled by default, but connection still failed.[/]")
            print("Check your network connection or proxy settings.")
        else:
             print(f"[red]Network Error: {e}[/]")
        return []
    except Exception as e:
        print(f"[red]Error fetching trending models: {e}[/]")
        return []
