# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import hashlib
import math
import sys
import time
import warnings
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import huggingface_hub
import optuna
import questionary
import torch
import torch.nn.functional as F
import transformers
from accelerate.utils import (
    is_mlu_available,
    is_musa_available,
    is_npu_available,
    is_sdaa_available,
    is_xpu_available,
)
from huggingface_hub import ModelCard, ModelCardData
from optuna import Trial
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler
from optuna.study import StudyDirection
from pydantic import ValidationError
from questionary import Choice, Style
from rich.traceback import install

from heretic.config import Settings
from heretic.evaluator import Evaluator
from heretic.model import AbliterationParameters, Model
from heretic.progress import ProgressTracker
from heretic.utils import (
    format_duration,
    get_readme_intro,
    get_trial_parameters,
    load_prompts,
    print,
)


def discover_models_in_directory(search_path: str) -> list:
    """
    Discover model directories (containing config.json) in the given path.
    
    Args:
        search_path: Directory path to search
        
    Returns:
        List of Path objects for discovered models
    """
    import os
    search_path = os.path.expanduser(search_path)
    search_path = os.path.abspath(search_path)
    base_path = Path(search_path)
    
    if not base_path.exists():
        return []
    
    models = []
    
    # Check if the path itself is a model
    if (base_path / "config.json").exists():
        models.append(base_path)
        return models
    
    # Search subdirectories (one level deep)
    try:
        for item in base_path.iterdir():
            if item.is_dir() and (item / "config.json").exists():
                models.append(item)
    except PermissionError:
        pass
    
    return sorted(models, key=lambda p: p.name)


def interactive_model_upload() -> None:
    """
    Interactive workflow to discover and upload models to HuggingFace.
    """
    print(f"\n[bold cyan]Upload Model to Hugging Face[/]")
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
    
    # Discover models
    print("\n[cyan]Searching for models...[/]")
    models = discover_models_in_directory(search_path)
    
    if not models:
        print(f"[yellow]No models found in {search_path}[/]")
        print("[dim]Models must contain a config.json file.[/]")
        return
    
    # Show discovered models
    print(f"[green]Found {len(models)} model(s):[/]")
    model_choices = [
        f"{model.name} ({model})" for model in models
    ]
    
    # Let user select model
    selected = questionary.select(
        "Select model to upload:",
        choices=model_choices,
        style=Style([("highlighted", "reverse")]),
    ).ask()
    
    if not selected:
        print("[yellow]Upload cancelled[/]")
        return
    
    # Extract model path from selection
    model_index = model_choices.index(selected)
    model_path = models[model_index]
    
    # Upload the model
    upload_model_to_huggingface(
        model_path=str(model_path),
        model_name=model_path.name,
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
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"[red]Error: Model path does not exist: {model_path}[/]")
        return
    
    print(f"\n[bold cyan]Uploading Model to Hugging Face[/]")
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
            from huggingface_hub import HfApi, upload_folder
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


def finetune_model(settings: Settings) -> None:
    """
    Run fine-tuning on an existing model (without abliteration).
    
    Args:
        settings: Application settings
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .finetuner import FineTuner
    
    print(f"\n[bold cyan]Fine-Tuning Mode[/]")
    print("=" * 80)
    print(f"Model: {settings.model}")
    print(f"Dataset: {settings.fine_tune_dataset}")
    print()
    
    # Ask for output directory
    default_output = str(Path(settings.finetuning_output_dir) / Path(settings.model).name)
    output_dir = questionary.text(
        "Output directory for fine-tuned model:",
        default=default_output,
    ).ask()
    
    if not output_dir:
        print("[yellow]Fine-tuning cancelled[/]")
        return
    
    # Update settings with user's output choice
    settings.finetuning_output_dir = output_dir
    
    # Load model
    print("[cyan]Loading model...[/]")
    model = AutoModelForCausalLM.from_pretrained(
        settings.model,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(settings.model)
    print("[green]✓ Model loaded[/]")
    
    # Create fine-tuner
    finetuner = FineTuner(
        model=model,
        tokenizer=tokenizer,
        settings=settings,
    )
    
    # Run fine-tuning
    result_path = finetuner.run(
        dataset_source=settings.fine_tune_dataset,
        preview_data=True,
    )
    
    if result_path:
        print(f"\n[bold green]✓ Fine-tuning complete![/]")
        print(f"  Output: {result_path}")
        
        # Ask if user wants to upload
        upload = questionary.confirm(
            "Would you like to upload the fine-tuned model to Hugging Face?",
            default=False,
        ).ask()
        
        if upload:
            upload_model_to_huggingface(
                model_path=result_path,
                model_name=Path(settings.model).name,
            )


def run():
    # Modified "Pagga" font from https://budavariam.github.io/asciiart-text/
    print(f"[cyan]█▀▄░█░░░█▀█░█▀▀░█▀█░█░█░█▀▀░█▄█░█▀▀░█▀▄[/]  v{version('blasphemer')}")
    print("[cyan]█▀▄░█░░░█▀█░▀▀█░█▀▀░█▀█░█▀▀░█░█░█▀▀░█▀▄[/]")
    print(
        "[cyan]▀▀░░▀▀▀░▀░▀░▀▀▀░▀░░░▀░▀░▀▀▀░▀░▀░▀▀▀░▀░▀[/]  [blue underline]https://github.com/sunkencity999/blasphemer[/]"
    )
    print()
    print("[dim]Developed by Christopher Bradford (@sunkencity999)[/]")
    print("[dim]Enhanced fork of Heretic - optimized for macOS (Apple Silicon)[/]")
    print()

    if (
        # An odd number of arguments have been passed (argv[0] is the program name),
        # so that after accounting for "--param VALUE" pairs, there is one left over.
        len(sys.argv) % 2 == 0
        # The leftover argument is a parameter value rather than a flag (such as "--help").
        and not sys.argv[-1].startswith("-")
    ):
        # Assume the last argument is the model.
        sys.argv.insert(-1, "--model")

    try:
        settings = Settings()
    except ValidationError as error:
        print(f"[red]Configuration contains [bold]{error.error_count()}[/] errors:[/]")

        for error in error.errors():
            print(f"[bold]{error['loc'][0]}[/]: [yellow]{error['msg']}[/]")

        print()
        print(
            "Run [bold]blasphemer --help[/] or see [bold]config.default.toml[/] for details about configuration parameters."
        )
        return

    # Adapted from https://github.com/huggingface/accelerate/blob/main/src/accelerate/commands/env.py
    if torch.cuda.is_available():
        print(f"GPU type: [bold]{torch.cuda.get_device_name()}[/]")
    elif torch.backends.mps.is_available():
        print(f"GPU type: [bold]Apple Silicon (MPS)[/]")
    elif is_xpu_available():
        print(f"XPU type: [bold]{torch.xpu.get_device_name()}[/]")
    elif is_mlu_available():
        print(f"MLU type: [bold]{torch.mlu.get_device_name()}[/]")
    elif is_sdaa_available():
        print(f"SDAA type: [bold]{torch.sdaa.get_device_name()}[/]")
    elif is_musa_available():
        print(f"MUSA type: [bold]{torch.musa.get_device_name()}[/]")
    elif is_npu_available():
        print(f"CANN version: [bold]{torch.version.cann}[/]")
    else:
        print(
            "[bold yellow]No GPU or other accelerator detected. Operations will be slow.[/]"
        )

    # We don't need gradients as we only do inference.
    torch.set_grad_enabled(False)

    # While determining the optimal batch size, we will try many different batch sizes,
    # resulting in many computation graphs being compiled. Raising the limit (default = 8)
    # avoids errors from TorchDynamo assuming that something is wrong because we
    # recompile too often.
    torch._dynamo.config.cache_size_limit = 64

    # Silence warning spam from Transformers.
    # In my entire career I've never seen a useful warning from that library.
    transformers.logging.set_verbosity_error()

    # We do our own trial logging, so we don't need the INFO messages
    # about parameters and results.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Silence the warning about multivariate TPE being experimental.
    warnings.filterwarnings("ignore", category=ExperimentalWarning)

    # Offer standalone upload option if no specific mode requested
    if not settings.fine_tune_only and not settings.fine_tune_dataset:
        # Check if user just wants to upload a model
        action = questionary.select(
            "What would you like to do?",
            choices=[
                "Process a model (abliteration/fine-tuning)",
                "Upload a model to HuggingFace",
                "Exit",
            ],
            style=Style([("highlighted", "reverse")]),
        ).ask()
        
        if action == "Upload a model to HuggingFace":
            interactive_model_upload()
            return
        elif action == "Exit":
            print("[cyan]Goodbye![/]")
            return
        # If "Process a model" selected, continue to normal flow

    # Check for fine-tune only mode
    if settings.fine_tune_only:
        if not settings.fine_tune_dataset:
            print("[red]Error: --fine-tune-dataset is required when using --fine-tune-only[/]")
            return
        finetune_model(settings)
        return
    
    # Check if model is a local path (expand and validate first)
    if settings.model:
        # Expand user paths and make absolute
        import os
        expanded_path = os.path.expanduser(settings.model)
        expanded_path = os.path.abspath(expanded_path)
        model_path = Path(expanded_path)
        
        # Check if it's a local model directory
        if model_path.exists() and (model_path / "config.json").exists():
            # Update settings to use absolute path
            settings.model = str(model_path)
            
            # Only show menu if not in fine-tune-only mode
            if not settings.fine_tune_only:
                action = questionary.select(
                    f"What would you like to do with {model_path.name}?",
                    choices=[
                        "Abliterate (remove censorship)",
                        "Fine-tune with LoRA",
                        "Upload to Hugging Face",
                    ],
                    style=Style([("highlighted", "reverse")]),
                ).ask()
                
                if action == "Fine-tune with LoRA":
                    # Get dataset
                    dataset_path = questionary.text(
                        "Path to fine-tuning dataset (directory, PDF, or HF dataset name):",
                    ).ask()
                    if dataset_path:
                        settings.fine_tune_dataset = dataset_path
                        settings.fine_tune_only = True
                        finetune_model(settings)
                    return
                elif action == "Upload to Hugging Face":
                    upload_model_to_huggingface(
                        model_path=str(model_path),
                        model_name=model_path.name,
                    )
                    return
                # If "Abliterate" selected, continue to normal flow
    
    model = Model(settings)

    print()
    print(f"Loading good prompts from [bold]{settings.good_prompts.dataset}[/]...")
    good_prompts = load_prompts(settings.good_prompts)
    print(f"* [bold]{len(good_prompts)}[/] prompts loaded")

    print()
    print(f"Loading bad prompts from [bold]{settings.bad_prompts.dataset}[/]...")
    bad_prompts = load_prompts(settings.bad_prompts)
    print(f"* [bold]{len(bad_prompts)}[/] prompts loaded")

    if settings.batch_size == 0:
        print()
        print("Determining optimal batch size...")

        batch_size = 1
        best_batch_size = -1
        best_performance = -1

        while batch_size <= settings.max_batch_size:
            print(f"* Trying batch size [bold]{batch_size}[/]... ", end="")

            prompts = good_prompts * math.ceil(batch_size / len(good_prompts))
            prompts = prompts[:batch_size]

            try:
                # Warmup run to build the computation graph so that part isn't benchmarked.
                model.get_responses(prompts)

                start_time = time.perf_counter()
                responses = model.get_responses(prompts)
                end_time = time.perf_counter()
            except Exception as error:
                if batch_size == 1:
                    # Even a batch size of 1 already fails.
                    # We cannot recover from this.
                    raise

                print(f"[red]Failed[/] ({error})")
                break

            response_lengths = [
                len(model.tokenizer.encode(response)) for response in responses
            ]
            performance = sum(response_lengths) / (end_time - start_time)

            print(f"[green]Ok[/] ([bold]{performance:.0f}[/] tokens/s)")

            if performance > best_performance:
                best_batch_size = batch_size
                best_performance = performance

            batch_size *= 2

        settings.batch_size = best_batch_size
        print(f"* Chosen batch size: [bold]{settings.batch_size}[/]")

    evaluator = Evaluator(settings, model)

    if settings.evaluate_model is not None:
        print()
        print(f"Loading model [bold]{settings.evaluate_model}[/]...")
        settings.model = settings.evaluate_model
        model.reload_model()
        print("* Evaluating...")
        evaluator.get_score()
        return

    # Set up checkpoint directory and study storage
    checkpoint_dir = Path(settings.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a unique study name based on the model ID
    model_hash = hashlib.md5(settings.model.encode()).hexdigest()[:8]
    study_name = f"blasphemer_{Path(settings.model).name}_{model_hash}"
    storage_path = checkpoint_dir / f"{study_name}.db"
    storage_url = f"sqlite:///{storage_path}"
    
    # Check if we're resuming from a checkpoint
    existing_study = None
    if storage_path.exists():
        if settings.resume:
            print()
            print(f"[bold green]Found existing checkpoint:[/] {storage_path}")
            try:
                existing_study = optuna.load_study(
                    study_name=study_name,
                    storage=storage_url,
                )
                completed_trials = len([t for t in existing_study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                print(f"* Completed trials: [bold]{completed_trials}[/]/{settings.n_trials}")
                if completed_trials >= settings.n_trials:
                    print("[yellow]Study already completed! Using existing results.[/]")
                else:
                    remaining = settings.n_trials - completed_trials
                    print(f"[bold cyan]Resuming optimization[/] - {remaining} trials remaining")
            except Exception as error:
                print(f"[yellow]Warning: Could not load checkpoint ({error}). Starting fresh.[/]")
                existing_study = None
        else:
            print()
            print(f"[yellow]Found existing checkpoint but --resume not specified.[/]")
            print(f"* Checkpoint location: {storage_path}")
            print("* Use [bold]--resume[/] to continue from checkpoint")
            print("* Starting fresh optimization (checkpoint will be overwritten)")
    else:
        print()
        print(f"Checkpoint will be saved to: [bold]{storage_path}[/]")
        print("* Use [bold]--resume[/] to continue if interrupted")

    print()
    print("Calculating per-layer refusal directions...")
    print("* Obtaining residuals for good prompts...")
    good_residuals = model.get_residuals_batched(good_prompts)
    print("* Obtaining residuals for bad prompts...")
    bad_residuals = model.get_residuals_batched(bad_prompts)
    refusal_directions = F.normalize(
        bad_residuals.mean(dim=0) - good_residuals.mean(dim=0),
        p=2,
        dim=1,
    )

    trial_index = 0
    start_time = time.perf_counter()
    saved_weights = None  # Cache for clean model weights
    
    # Initialize progress tracker for enhanced observability
    progress_tracker = ProgressTracker(
        total_trials=settings.n_trials,
        model_name=settings.model
    )

    def objective(trial: Trial) -> tuple[float, float]:
        nonlocal trial_index, saved_weights
        trial_index += 1
        trial.set_user_attr("index", trial_index)

        direction_scope = trial.suggest_categorical(
            "direction_scope",
            [
                "global",
                "per layer",
            ],
        )

        # Discrimination between "harmful" and "harmless" inputs is usually strongest
        # in layers slightly past the midpoint of the layer stack. See the original
        # abliteration paper (https://arxiv.org/abs/2406.11717) for a deeper analysis.
        #
        # Note that we always sample this parameter even though we only need it for
        # the "global" direction scope. The reason is that multivariate TPE doesn't
        # work with conditional or variable-range parameters.
        direction_index = trial.suggest_float(
            "direction_index",
            0.4 * (len(model.get_layers()) - 1),
            0.9 * (len(model.get_layers()) - 1),
        )

        if direction_scope == "per layer":
            direction_index = None

        parameters = {}

        for component in model.get_abliterable_components():
            # The parameter ranges are based on experiments with various models
            # and much wider ranges. They are not set in stone and might have to be
            # adjusted for future models.
            max_weight = trial.suggest_float(
                f"{component}.max_weight",
                0.8,
                1.5,
            )
            max_weight_position = trial.suggest_float(
                f"{component}.max_weight_position",
                0.6 * (len(model.get_layers()) - 1),
                len(model.get_layers()) - 1,
            )
            # For sampling purposes, min_weight is expressed as a fraction of max_weight,
            # again because multivariate TPE doesn't support variable-range parameters.
            # The value is transformed into the actual min_weight value below.
            min_weight = trial.suggest_float(
                f"{component}.min_weight",
                0.0,
                1.0,
            )
            min_weight_distance = trial.suggest_float(
                f"{component}.min_weight_distance",
                1.0,
                0.6 * (len(model.get_layers()) - 1),
            )

            parameters[component] = AbliterationParameters(
                max_weight=max_weight,
                max_weight_position=max_weight_position,
                min_weight=(min_weight * max_weight),
                min_weight_distance=min_weight_distance,
            )

        trial.set_user_attr("direction_index", direction_index)
        # Convert AbliterationParameters objects to dicts for JSON serialization
        trial.set_user_attr("parameters", {k: asdict(v) for k, v in parameters.items()})

        # Display progress with current parameters
        progress_tracker.display_progress(
            current_trial=trial_index,
            current_params=get_trial_parameters(trial)
        )
        
        print()
        
        # Performance optimization: Save clean weights on first trial,
        # then restore from cache instead of reloading entire model.
        # This reduces trial overhead from ~30s to ~2s (10-15x speedup).
        if trial_index == 1:
            print("* Saving clean model weights...")
            saved_weights = model.save_abliterable_weights()
            print("* Abliterating...")
        else:
            print("* Restoring clean weights (fast)...")
            model.restore_abliterable_weights(saved_weights)
            print("* Abliterating...")
        
        model.abliterate(refusal_directions, direction_index, parameters)
        print("* Evaluating...")
        score, kl_divergence, refusals = evaluator.get_score()

        # Add trial results to progress tracker
        progress_tracker.add_trial(
            trial_number=trial_index,
            kl_divergence=kl_divergence,
            refusals=refusals,
            total_prompts=len(evaluator.bad_prompts),
            parameters=get_trial_parameters(trial)
        )
        
        # Display updated progress with results
        progress_tracker.display_progress(
            current_trial=trial_index,
            current_kl=kl_divergence,
            current_refusals=refusals,
            current_params=get_trial_parameters(trial)
        )

        trial.set_user_attr("kl_divergence", kl_divergence)
        trial.set_user_attr("refusals", refusals)

        return score

    # Create or load study with persistent storage
    if existing_study is not None:
        study = existing_study
        # Calculate how many trials are left to run
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_trials_to_run = settings.n_trials - completed_trials
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            sampler=TPESampler(
                n_startup_trials=settings.n_startup_trials,
                n_ei_candidates=128,
                multivariate=True,
            ),
            directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE],
            load_if_exists=False,  # We already checked above
        )
        n_trials_to_run = settings.n_trials

    if n_trials_to_run > 0:
        print()
        print(f"[bold cyan]Starting optimization:[/] {n_trials_to_run} trials")
        print(f"* Checkpoint: [bold]{storage_path}[/]")
        study.optimize(objective, n_trials=n_trials_to_run)
        
        # Display completion summary with quality analysis
        progress_tracker.display_completion_summary()
        print()
        print(f"[bold green]✓ Checkpoint saved:[/] {storage_path}")

    best_trials = sorted(
        study.best_trials,
        key=lambda trial: trial.user_attrs["refusals"],
    )

    choices = [
        Choice(
            title=(
                f"[Trial {trial.user_attrs['index']:>3}] "
                f"Refusals: {trial.user_attrs['refusals']:>2}/{len(evaluator.bad_prompts)}, "
                f"KL divergence: {trial.user_attrs['kl_divergence']:.2f}"
            ),
            value=trial,
        )
        for trial in best_trials
    ]

    choices.append(
        Choice(
            title="None (exit program)",
            value="",
        )
    )

    print()
    print("[bold green]Optimization finished![/]")
    print()
    print(
        (
            "The following trials resulted in Pareto optimal combinations of refusals and KL divergence. "
            "After selecting a trial, you will be able to save the model, upload it to Hugging Face, "
            "or chat with it to test how well it works. You can return to this menu later to select a different trial. "
            "[yellow]Note that KL divergence values above 1 usually indicate significant damage to the original model's capabilities.[/]"
        )
    )

    while True:
        print()
        trial = questionary.select(
            "Which trial do you want to use?",
            choices=choices,
            style=Style([("highlighted", "reverse")]),
        ).ask()

        if trial is None or trial == "":
            break

        print()
        print(f"Restoring model from trial [bold]{trial.user_attrs['index']}[/]...")
        print("* Reloading model...")
        model.reload_model()
        print("* Abliterating...")
        
        # Convert parameter dicts back to AbliterationParameters objects
        parameters_dict = trial.user_attrs["parameters"]
        parameters = {
            k: AbliterationParameters(**v) for k, v in parameters_dict.items()
        }
        
        model.abliterate(
            refusal_directions,
            trial.user_attrs["direction_index"],
            parameters,
        )

        while True:
            print()
            
            # Build menu choices
            menu_choices = [
                "Save the model to a local folder",
                "Upload the model to Hugging Face",
                "Upload a different model directory to Hugging Face",
                "Upload any model (browse and discover)",
            ]
            
            # Add fine-tuning option if dataset is configured
            if settings.fine_tune_dataset:
                menu_choices.insert(0, "Fine-tune with LoRA (knowledge injection)")
            
            menu_choices.extend([
                "Chat with the model",
                "Nothing (return to trial selection menu)",
            ])
            
            action = questionary.select(
                "What do you want to do with the decensored model?",
                choices=menu_choices,
                style=Style([("highlighted", "reverse")]),
            ).ask()

            if action is None or action == "Nothing (return to trial selection menu)":
                break

            # All actions are wrapped in a try/except block so that if an error occurs,
            # another action can be tried, instead of the program crashing and losing
            # the optimized model.
            try:
                match action:
                    case "Fine-tune with LoRA (knowledge injection)":
                        from .finetuner import FineTuner
                        
                        print("\n[bold cyan]Starting Fine-Tuning Process[/]")
                        print("=" * 80)
                        
                        # Ask for output directory
                        default_output = str(Path(settings.finetuning_output_dir) / f"{Path(settings.model).name}-finetuned")
                        output_dir = questionary.text(
                            "Output directory for fine-tuned model:",
                            default=default_output,
                        ).ask()
                        
                        if not output_dir:
                            print("[yellow]Fine-tuning cancelled[/]")
                            continue
                        
                        # Update settings with user's output choice
                        original_output_dir = settings.finetuning_output_dir
                        settings.finetuning_output_dir = output_dir
                        
                        # Create fine-tuner
                        finetuner = FineTuner(
                            model=model.model,
                            tokenizer=model.tokenizer,
                            settings=settings,
                        )
                        
                        # Run fine-tuning
                        result_path = finetuner.run(
                            dataset_source=settings.fine_tune_dataset,
                            preview_data=True,
                        )
                        
                        # Restore original setting
                        settings.finetuning_output_dir = original_output_dir
                        
                        if result_path:
                            print(f"\n[bold green]✓ Fine-tuning complete![/]")
                            print(f"  Output: {result_path}")
                            
                            # Ask if user wants to upload
                            upload = questionary.confirm(
                                "Would you like to upload the fine-tuned model to Hugging Face?",
                                default=False,
                            ).ask()
                            
                            if upload:
                                upload_model_to_huggingface(
                                    model_path=result_path,
                                    model_name=Path(settings.model).name,
                                )
                            
                            # Update model reference if merged
                            if settings.merge_lora:
                                # Reload merged model
                                from transformers import AutoModelForCausalLM, AutoTokenizer
                                
                                print("\n[cyan]Reloading merged model...[/]")
                                model.model = AutoModelForCausalLM.from_pretrained(result_path)
                                model.tokenizer = AutoTokenizer.from_pretrained(result_path)
                                print("[green]✓ Merged model loaded[/]")
                    
                    case "Save the model to a local folder":
                        save_directory = questionary.path("Path to the folder:").ask()
                        if not save_directory:
                            continue

                        # Expand ~ and environment variables in path
                        import os
                        save_directory = os.path.expanduser(save_directory)
                        save_directory = os.path.abspath(save_directory)

                        print("Saving model...")
                        model.model.save_pretrained(save_directory)
                        model.tokenizer.save_pretrained(save_directory)
                        print(f"Model saved to [bold]{save_directory}[/].")

                    case "Upload the model to Hugging Face":
                        # We don't use huggingface_hub.login() because that stores the token on disk,
                        # and since this program will often be run on rented or shared GPU servers,
                        # it's better to not persist credentials.
                        token = huggingface_hub.get_token()
                        if not token:
                            token = questionary.password(
                                "Hugging Face access token:"
                            ).ask()
                        if not token:
                            continue

                        user = huggingface_hub.whoami(token)
                        print(
                            f"Logged in as [bold]{user['fullname']} ({user['email']})[/]"
                        )

                        repo_id = questionary.text(
                            "Name of repository:",
                            default=f"{user['name']}/{Path(settings.model).name}-blasphemer",
                        ).ask()

                        visibility = questionary.select(
                            "Should the repository be public or private?",
                            choices=[
                                "Public",
                                "Private",
                            ],
                            style=Style([("highlighted", "reverse")]),
                        ).ask()
                        private = visibility == "Private"

                        print("Uploading model...")

                        model.model.push_to_hub(
                            repo_id,
                            private=private,
                            token=token,
                        )
                        model.tokenizer.push_to_hub(
                            repo_id,
                            private=private,
                            token=token,
                        )

                        # If the model path doesn't exist locally, it can be assumed
                        # to be a model hosted on the Hugging Face Hub, in which case
                        # we can retrieve the model card.
                        if not Path(settings.model).exists():
                            card = ModelCard.load(settings.model)
                            if card.data is None:
                                card.data = ModelCardData()
                            if card.data.tags is None:
                                card.data.tags = []
                            card.data.tags.append("heretic")
                            card.data.tags.append("uncensored")
                            card.data.tags.append("decensored")
                            card.data.tags.append("abliterated")
                            card.text = (
                                get_readme_intro(
                                    settings,
                                    trial,
                                    evaluator.base_refusals,
                                    evaluator.bad_prompts,
                                )
                                + card.text
                            )
                            card.push_to_hub(repo_id, token=token)

                        print(f"Model uploaded to [bold]{repo_id}[/].")

                    case "Upload a different model directory to Hugging Face":
                        # Ask for model directory path
                        model_dir = questionary.path(
                            "Path to the model directory:",
                            only_directories=True,
                        ).ask()
                        
                        if not model_dir:
                            print("[yellow]Upload cancelled[/]")
                            continue
                        
                        # Expand path
                        import os
                        model_dir = os.path.expanduser(model_dir)
                        model_dir = os.path.abspath(model_dir)
                        
                        # Upload the model
                        upload_model_to_huggingface(
                            model_path=model_dir,
                            model_name=Path(model_dir).name,
                        )

                    case "Upload any model (browse and discover)":
                        # Use interactive model discovery and upload
                        interactive_model_upload()

                    case "Chat with the model":
                        print()
                        print(
                            "[cyan]Press Ctrl+C at any time to return to the menu.[/]"
                        )

                        chat = [
                            {"role": "system", "content": settings.system_prompt},
                        ]

                        while True:
                            try:
                                message = questionary.text(
                                    "User:",
                                    qmark=">",
                                ).unsafe_ask()
                                if not message:
                                    break
                                chat.append({"role": "user", "content": message})

                                print("[bold]Assistant:[/] ", end="")
                                response = model.stream_chat_response(chat)
                                chat.append({"role": "assistant", "content": response})
                            except (KeyboardInterrupt, EOFError):
                                # Ctrl+C/Ctrl+D
                                break

            except Exception as error:
                print(f"[red]Error: {error}[/]")


def main():
    # Install Rich traceback handler.
    install()

    try:
        run()
    except BaseException as error:
        # Transformers appears to handle KeyboardInterrupt (or BaseException)
        # internally in some places, which can re-raise a different error in the handler,
        # masking the root cause. We therefore check both the error itself and its context.
        if isinstance(error, KeyboardInterrupt) or isinstance(
            error.__context__, KeyboardInterrupt
        ):
            print()
            print("[red]Shutting down...[/]")
        else:
            raise
