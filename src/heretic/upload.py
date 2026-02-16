# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import time
from pathlib import Path

import huggingface_hub
import questionary
from huggingface_hub import ModelCard, HfApi, upload_folder
from questionary import Style
from transformers import AutoModelForCausalLM, AutoTokenizer

from heretic.discovery import discover_models_in_directory
from heretic.utils import print


def interactive_model_upload() -> None:
    """
    Interactive workflow to discover and upload models to HuggingFace.
    """
    print("\n[bold cyan]Upload Model to Hugging Face[/]")
    print("=" * 80)
    print("This will help you upload a model to HuggingFace Hub.")
    print()
    
    # Ask for search path
    search_path = questionary.path(
        "Enter directory to search for models (or direct model path):",
        only_directories=True,
    ).ask()
    
    if not search_path:
        print("[yellow]Upload cancelled[/]")
        return
    
    # Discover models and GGUF files
    print("\n[cyan]Searching for models and GGUF files...[/]")
    discovery = discover_models_in_directory(search_path)
    models = discovery["models"]
    gguf_files = discovery["gguf_files"]
    
    if not models and not gguf_files:
        print(f"[yellow]No models or GGUF files found in {search_path}[/]")
        print("[dim]Models must contain a config.json file, or be .gguf files.[/]")
        return
    
    # Build selection list
    all_choices = []
    choice_map = {}  # Map choice string to (type, path)
    
    # Add model directories
    if models:
        print(f"[green]Found {len(models)} model directory(ies):[/]")
        for model in models:
            choice_text = f"📁 {model.name} [Model Directory]"
            all_choices.append(choice_text)
            choice_map[choice_text] = ("model", model)
    
    # Add GGUF files
    if gguf_files:
        print(f"[green]Found {len(gguf_files)} GGUF file(s):[/]")
        for gguf in gguf_files:
            choice_text = f"📦 {gguf.name} [GGUF]"
            all_choices.append(choice_text)
            choice_map[choice_text] = ("gguf", gguf)
    
    print()
    
    # Let user select what to upload
    selected = questionary.select(
        "Select item to upload:",
        choices=all_choices,
        style=Style([("highlighted", "reverse")]),
    ).ask()
    
    if not selected:
        print("[yellow]Upload cancelled[/]")
        return
    
    # Get the selected item type and path
    item_type, item_path = choice_map[selected]
    
    # Upload based on type
    if item_type == "model":
        upload_model_to_huggingface(
            model_path=str(item_path),
            model_name=item_path.name,
        )
    elif item_type == "gguf":
        upload_gguf_to_huggingface(
            gguf_path=str(item_path),
        )


def upload_model_to_huggingface(
    model_path: str,
    model_name: str = None,
    token: str = None,
) -> None:
    """
    Upload a model directory to HuggingFace Hub.
    
    Args:
        model_path: Path to the model directory
        model_name: Optional model name (for default repo name)
        token: Optional HF token (will prompt if not provided)
    """
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"[red]Error: Model path does not exist: {model_path}[/]")
        return
    
    print("\n[bold cyan]Uploading Model to Hugging Face[/]")
    print("=" * 80)
    print(f"Model path: {model_path}")
    print()
    
    # Get token
    if not token:
        token = huggingface_hub.get_token()
    if not token:
        token = questionary.password("Hugging Face access token:").ask()
    if not token:
        print("[yellow]Upload cancelled[/]")
        return
    
    # Get user info
    try:
        user = huggingface_hub.whoami(token)
        print(f"Logged in as [bold]{user['fullname']} ({user['email']})[/]")
    except Exception as e:
        print(f"[red]Error: Invalid token or connection failed: {e}[/]")
        return
    
    # Get repo name
    default_name = model_name or model_path.name
    repo_id = questionary.text(
        "Name of repository:",
        default=f"{user['name']}/{default_name}",
    ).ask()
    
    if not repo_id:
        print("[yellow]Upload cancelled[/]")
        return
    
    # Get visibility
    visibility = questionary.select(
        "Should the repository be public or private?",
        choices=["Public", "Private"],
        style=Style([("highlighted", "reverse")]),
    ).ask()
    private = visibility == "Private"
    
    # Ask about model card
    create_card = questionary.confirm(
        "Would you like to create a model card?",
        default=True,
    ).ask()
    
    # Check for GGUF files
    gguf_files = list(model_path.glob("*.gguf"))
    
    # Load and upload model
    try:
        # If there are GGUF files, upload entire directory to preserve them
        if gguf_files:
            print(f"\n[cyan]Found {len(gguf_files)} GGUF file(s) - uploading entire directory...[/]")
            for gguf in gguf_files:
                print(f"  • {gguf.name}")
            print()
            
            # Create repository first
            api = HfApi()
            
            print("[cyan]Creating repository...[/]")
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                token=token,
                exist_ok=True,
            )
            
            # Upload entire folder to preserve all files
            print("[cyan]Uploading files (this may take several minutes for large GGUF files)...[/]")
            
            # List all files to be uploaded
            all_files = list(model_path.glob("*"))
            print(f"[dim]Uploading {len(all_files)} files total...[/]")
            
            upload_folder(
                folder_path=str(model_path),
                repo_id=repo_id,
                repo_type="model",
                token=token,
                ignore_patterns=[".*"],  # Only ignore hidden files
            )
            print("[green]✓ All files uploaded (including GGUFs)[/]")
        else:
            # Standard model upload (no GGUF files)
            print("\n[cyan]Loading model...[/]")
            model = AutoModelForCausalLM.from_pretrained(str(model_path))
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            
            print("[cyan]Uploading model...[/]")
            model.push_to_hub(repo_id, private=private, token=token)
            
            print("[cyan]Uploading tokenizer...[/]")
            tokenizer.push_to_hub(repo_id, private=private, token=token)
            print("[green]✓ Model uploaded[/]")
        
        # Create model card if requested
        if create_card:
            try:
                print("[cyan]Creating model card...[/]")
                card = ModelCard.load(repo_id, token=token)
                if not card.text or card.text.strip() == "":
                    # Build GGUF section if files exist
                    gguf_section = ""
                    if gguf_files:
                        gguf_list = "\n".join([f"- `{gguf.name}`" for gguf in gguf_files])
                        gguf_section = f"""
## GGUF Files

This repository includes pre-quantized GGUF files for use with llama.cpp and other GGUF-compatible inference engines:

{gguf_list}

### Using GGUF Files

```bash
# Download a specific GGUF file
huggingface-cli download {repo_id} {gguf_files[0].name}

# Use with llama.cpp
./llama.cpp/main -m {gguf_files[0].name} -p "Your prompt here"
```
"""
                    
                    card.text = f"""---
tags:
- text-generation
- transformers
- blasphemer{" " if not gguf_files else ""}
{"- gguf" if gguf_files else ""}
license: other
---

# {model_path.name}

This model was uploaded using [Blasphemer](https://github.com/sunkencity999/blasphemer).

## Model Details

- **Base Model**: {model_path.name}
- **Upload Date**: {time.strftime("%Y-%m-%d")}
- **Uploaded by**: {user['name']}
{f"- **GGUF Files**: {len(gguf_files)} quantized versions included" if gguf_files else ""}

## Usage

### Transformers (PyTorch)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

# Generate text
prompt = "Your prompt here"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```
{gguf_section}
## Citation

If you use this model, please cite:

```
@software{{blasphemer2025,
  author = {{Bradford, Christopher}},
  title = {{Blasphemer: Advanced Model Modification Toolkit}},
  year = {{2025}},
  url = {{https://github.com/sunkencity999/blasphemer}}
}}
```
"""
                    card.push_to_hub(repo_id, token=token)
                    print("[green]✓ Model card created[/]")
            except Exception as card_error:
                print(f"[yellow]Warning: Could not create model card: {card_error}[/]")
        
        print(f"\n[bold green]✓ Model uploaded to {repo_id}[/]")
        print(f"View at: [blue underline]https://huggingface.co/{repo_id}[/]")
        
    except Exception as e:
        print(f"[red]Error uploading model: {e}[/]")
        import traceback
        traceback.print_exc()


def upload_gguf_to_huggingface(
    gguf_path: str,
    token: str = None,
) -> None:
    """
    Upload a GGUF file to HuggingFace Hub.
    
    Args:
        gguf_path: Path to the GGUF file
        token: Optional HF token (will prompt if not provided)
    """
    gguf_path = Path(gguf_path)
    if not gguf_path.exists():
        print(f"[red]Error: GGUF file does not exist: {gguf_path}[/]")
        return
    
    print("\n[bold cyan]Uploading GGUF to Hugging Face[/]")
    print("=" * 80)
    print(f"GGUF file: {gguf_path.name}")
    print(f"Size: {gguf_path.stat().st_size / (1024**3):.2f} GB")
    print()
    
    # Get token
    if not token:
        token = huggingface_hub.get_token()
    if not token:
        token = questionary.password("Hugging Face access token:").ask()
    if not token:
        print("[yellow]Upload cancelled[/]")
        return
    
    # Get user info
    try:
        api = HfApi()
        user = huggingface_hub.whoami(token)
        print(f"Logged in as [bold]{user['fullname']} ({user['name']})[/]")
    except Exception as e:
        print(f"[red]Error: Invalid token or connection failed: {e}[/]")
        return
    
    # Get repo name
    default_name = gguf_path.stem.replace(".gguf", "")
    repo_name = questionary.text(
        "Repository name (without username):",
        default=default_name,
    ).ask()
    
    if not repo_name:
        print("[yellow]Upload cancelled[/]")
        return
    
    repo_id = f"{user['name']}/{repo_name}"
    
    # Ask if private
    private = questionary.confirm(
        "Make repository private?",
        default=False,
    ).ask()
    
    # Ask if this should create/update model card
    create_card = questionary.confirm(
        "Create or update model card?",
        default=True,
    ).ask()
    
    try:
        # Create or get repository
        print(f"\n[cyan]Creating repository {repo_id}...[/]")
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                exist_ok=True,
            )
            print("[green]✓ Repository ready[/]")
        except Exception as e:
            print(f"[yellow]Note: {e}[/]")
        
        # Upload GGUF file
        print(f"\n[cyan]Uploading {gguf_path.name} (this may take several minutes)...[/]")
        api.upload_file(
            path_or_fileobj=str(gguf_path),
            path_in_repo=gguf_path.name,
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
        print(f"[green]✓ {gguf_path.name} uploaded[/]")
        
        # Create model card if requested
        if create_card:
            try:
                print("[cyan]Creating/updating model card...[/]")
                
                # Try to load existing card
                try:
                    card = ModelCard.load(repo_id, token=token)
                    print("[dim]Updating existing model card...[/]")
                except Exception:
                    card = ModelCard("")
                    print("[dim]Creating new model card...[/]")
                
                # Only update if card is empty or very short
                if not card.text or len(card.text.strip()) < 100:
                    card.text = f"""---
tags:
- gguf
- quantized
- blasphemer
license: other
---

# {repo_name}

This GGUF model was uploaded using [Blasphemer](https://github.com/sunkencity999/blasphemer).

## File Information

- **Filename**: `{gguf_path.name}`
- **Size**: {gguf_path.stat().st_size / (1024**3):.2f} GB
- **Upload Date**: {time.strftime("%Y-%m-%d")}

## Usage

### With llama.cpp

```bash
# Download the model
huggingface-cli download {repo_id} {gguf_path.name}

# Run with llama.cpp
./llama.cpp/main -m {gguf_path.name} -p "Your prompt here"
```

### With LM Studio

1. Open LM Studio
2. Go to "Download" or search for `{repo_id}`
3. Download and load the model
4. Start chatting!

### With Python (llama-cpp-python)

```python
from llama_cpp import Llama

llm = Llama(model_path="{gguf_path.name}")
output = llm("Your prompt here", max_tokens=100)
print(output['choices'][0]['text'])
```

## Citation

If you use this model, please cite:

```
@software{{blasphemer2025,
  author = {{Bradford, Christopher}},
  title = {{Blasphemer: Advanced Model Modification Toolkit}},
  year = {{2025}},
  url = {{https://github.com/sunkencity999/blasphemer}}
}}
```
"""
                card.push_to_hub(repo_id, token=token)
                print("[green]✓ Model card created[/]")
            except Exception as card_error:
                print(f"[yellow]Warning: Could not create/update model card: {card_error}[/]")
        
        print(f"\n[bold green]✓ GGUF file uploaded to {repo_id}[/]")
        print(f"View at: [blue underline]https://huggingface.co/{repo_id}[/]")
        
    except Exception as e:
        print(f"[red]Error uploading GGUF: {e}[/]")
        import traceback
        traceback.print_exc()
