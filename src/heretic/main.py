# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import math
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path
import os

# SSL/TLS verification for Hugging Face traffic is ON by default. Environments
# behind a TLS-intercepting corporate proxy or VPN can opt out by setting
# BLASPHEMER_DISABLE_SSL_VERIFY=1. This MUST be resolved before huggingface_hub
# is imported, since it reads HF_HUB_DISABLE_SSL_VERIFY at import time.
if os.environ.get("BLASPHEMER_DISABLE_SSL_VERIFY", "").lower() in ("1", "true", "yes"):
    os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

import huggingface_hub
import optuna
import questionary
import torch
import transformers
from accelerate.utils import (
    is_mlu_available,
    is_musa_available,
    is_npu_available,
    is_sdaa_available,
    is_xpu_available,
)
from huggingface_hub import ModelCard, ModelCardData
from optuna.exceptions import ExperimentalWarning
from pydantic import ValidationError
from questionary import Choice, Style
from rich.traceback import install

from heretic.config import Settings
from heretic.evaluator import Evaluator
from heretic.model import AbliterationParameters, Model
from heretic.utils import (
    get_readme_intro,
    load_prompts,
    print,
)
from heretic.upload import (
    interactive_model_upload,
    upload_model_to_huggingface,
)
from heretic.optimizer import Optimizer
from heretic.preflight import PreflightCheck
from heretic.discovery import search_huggingface_models, list_recent_trending_models
from heretic.deploy import Deployer





def finetune_model(settings: Settings) -> None:
    """
    Run fine-tuning on an existing model (without abliteration).
    
    Args:
        settings: Application settings
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .finetuner import FineTuner
    
    print("\n[bold cyan]Fine-Tuning Mode[/]")
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
        print("\n[bold green]✓ Fine-tuning complete![/]")
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
        print("GPU type: [bold]Apple Silicon (MPS)[/]")
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

    # Show the interactive top-level menu only when no model was specified on
    # the command line / config (pure interactive launch). When a model is
    # given, honor the documented CLI behavior and proceed straight to
    # processing (respecting any fine-tuning flags).
    while not settings.model:
        # Interactive top-level menu, shown only when no model was given.
        # Each branch either sets settings.model and breaks (to proceed to
        # processing), or loops back to this menu via continue.
        action = questionary.select(
            "What would you like to do?",
            choices=[
                "Process a model (abliteration/fine-tuning)",
                "Search HuggingFace Models",
                "List Recent Trending Models",
                "Quantize & Upload Model",
                "Exit",
            ],
            style=Style([("highlighted", "reverse")]),
        ).ask()

        if action is None or action == "Exit":
            print("[cyan]Goodbye![/]")
            return

        if action == "Process a model (abliteration/fine-tuning)":
            # Proceed to model selection below.
            break

        if action == "Quantize & Upload Model":
             print("\n[bold cyan]Quantization & Deployment[/]")
             print("=" * 80)

             deployer = Deployer()
             if not deployer.ensure_llama_cpp_build():
                 continue

             # Ask what to do
             deploy_action = questionary.select(
                 "Choose action:",
                 choices=[
                     "Convert HF model to GGUF",
                     "Quantize existing GGUF",
                     "Upload model/GGUF to HuggingFace",
                     "Back",
                 ]
             ).ask()
             
             if deploy_action == "Convert HF model to GGUF":
                 from heretic.discovery import discover_models_in_directory
                 import os
                 default_models_dir = os.path.expanduser("~/blasphemer-models")
                 search_paths = [".", default_models_dir]
                 
                 print(f"[cyan]Scanning for models in current directory and {default_models_dir}...[/]")
                 discovered = discover_models_in_directory(search_paths)
                 choices = [str(m) for m in discovered.get("models", [])]
                 choices.append("Other (enter path)")
                 
                 model_path = questionary.select(
                     "Select model directory:",
                     choices=choices
                 ).ask()
                 
                 if model_path == "Other (enter path)":
                     model_path = questionary.path("Path to HF model directory:", only_directories=True).ask()
                 
                 if model_path:
                     out_type = questionary.select("Output type:", choices=["f16", "f32", "q8_0"]).ask()
                     deployer.convert_to_gguf(Path(model_path), out_type=out_type)
                     
             elif deploy_action == "Quantize existing GGUF":
                 from heretic.discovery import discover_models_in_directory
                 import os
                 default_models_dir = os.path.expanduser("~/blasphemer-models")
                 search_paths = [".", default_models_dir]
                 
                 print(f"[cyan]Scanning for GGUF files in current directory and {default_models_dir}...[/]")
                 discovered = discover_models_in_directory(search_paths)
                 choices = [str(g) for g in discovered.get("gguf_files", [])]
                 choices.append("Other (enter path)")
                 
                 gguf_path = questionary.select(
                     "Select GGUF file:",
                     choices=choices
                 ).ask()
                 
                 if gguf_path == "Other (enter path)":
                     gguf_path = questionary.path("Path to GGUF file:").ask()
                 
                 if gguf_path:
                     methods = questionary.checkbox(
                         "Select quantization methods:",
                         choices=["Q4_K_M", "Q5_K_M", "Q8_0", "Q6_K", "Q3_K_M"],
                         default="Q4_K_M"
                     ).ask()
                     if methods:
                         deployer.quantize_model(Path(gguf_path), methods)
             
             elif deploy_action == "Upload model/GGUF to HuggingFace":
                 interactive_model_upload()

             # Return to the top-level menu after the deploy action.
             continue

        if action in ["Search HuggingFace Models", "List Recent Trending Models"]:
            if action == "Search HuggingFace Models":
                query = questionary.text("Enter search query:").ask()
                if not query:
                    continue
                print("[cyan]Searching...[/]")
                models = search_huggingface_models(query, limit=20)
            else:
                print("[cyan]Fetching trending models...[/]")
                models = list_recent_trending_models(limit=20)

            if not models:
                print("[yellow]No models found.[/]")
                continue

            # Create choices
            model_choices = []
            for m in models:
                info = f"{m['id']} (⬇ {m['downloads']} ❤ {m['likes']})"
                model_choices.append(Choice(title=info, value=m['id']))

            model_choices.append(Choice(title="Back", value="back"))

            selected_model_id = questionary.select(
                "Select a model:",
                choices=model_choices,
                style=Style([("highlighted", "reverse")]),
            ).ask()

            if not selected_model_id or selected_model_id == "back":
                continue

            # Ask what to do with the selected model
            next_step = questionary.select(
                f"Action for {selected_model_id}:",
                choices=[
                    "Process this model (Download & Abliterate)",
                    "View on HuggingFace",
                    "Back",
                ],
                style=Style([("highlighted", "reverse")]),
            ).ask()

            if next_step == "Process this model (Download & Abliterate)":
                # Transformers will download this model ID automatically.
                settings.model = selected_model_id
                break
            elif next_step == "View on HuggingFace":
                import webbrowser
                webbrowser.open(f"https://huggingface.co/{selected_model_id}")
                continue
            else:
                continue


    # Check if model is set. If we reached here (Process a model selected or fell through),
    # and settings.model is None, we MUST ask for it.
    if not settings.model:
         from heretic.discovery import discover_models_in_directory
         import os
         
         print("\n[cyan]No model specified. Please select a model to process.[/]")
         
         # Scan for models
         default_models_dir = os.path.expanduser("~/blasphemer-models")
         search_paths = [".", default_models_dir]
         discovered = discover_models_in_directory(search_paths)
         choices = [str(m) for m in discovered.get("models", [])]
         choices.append("Other (enter path or HuggingFace ID)")
         
         model_input = questionary.select(
             "Select model:",
             choices=choices
         ).ask()
         
         if model_input == "Other (enter path or HuggingFace ID)":
             settings.model = questionary.text("Enter path or HuggingFace ID:").ask()
         else:
             settings.model = model_input
             
         if not settings.model:
             print("[yellow]No model selected. Exiting.[/]")
             return

    # Check if for fine-tune only mode
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
    
    # Run pre-flight checks
    preflight = PreflightCheck(settings.model, settings.device_map)
    if not preflight.run_checks():
        # Ask user if they want to continue despite errors/warnings
        if not questionary.confirm("Checks failed or emitted warnings. Continue anyway?", default=False).ask():
            print("[yellow]Aborted by user.[/]")
            return

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

    # Run optimization
    optimizer = Optimizer(settings, model, evaluator)
    opt_start = time.perf_counter()
    study = optimizer.optimize()
    opt_elapsed = time.perf_counter() - opt_start

    if not study:
         print("[yellow]Optimization skipped or no trials run.[/]")
         return

    # Export a durable optimization report (CSV of all trials + Markdown
    # Pareto-front summary) so the run's results survive after the program exits.
    try:
        from heretic.reporting import build_records_from_study, export_report

        report_dir = (
            Path(settings.checkpoint_dir) / "reports" / Path(settings.model).name
        )
        export_report(
            build_records_from_study(study),
            model_name=settings.model,
            output_dir=report_dir,
            base_refusals=evaluator.base_refusals,
            total_prompts=len(evaluator.bad_prompts),
            elapsed_seconds=opt_elapsed,
        )
        print(f"[green]Optimization report saved to[/] [bold]{report_dir}[/]")
    except Exception as error:
        print(f"[yellow]Could not write optimization report: {error}[/]")

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
            optimizer.refusal_directions,
            trial.user_attrs["direction_index"],
            parameters,
        )

        while True:
            print()
            
            # Build menu choices
            menu_choices = [
                "Save the model to a local folder",
                "Upload to Hugging Face",
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
                            print("\n[bold green]✓ Fine-tuning complete![/]")
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

                    case "Upload to Hugging Face":
                        upload_source = questionary.select(
                            "What would you like to upload?",
                            choices=[
                                "This decensored model",
                                "Another model directory",
                                "Browse and discover models/GGUFs",
                                "Back",
                            ],
                            style=Style([("highlighted", "reverse")]),
                        ).ask()

                        if upload_source is None or upload_source == "Back":
                            continue

                        if upload_source == "Another model directory":
                            model_dir = questionary.path(
                                "Path to the model directory:",
                                only_directories=True,
                            ).ask()
                            if not model_dir:
                                print("[yellow]Upload cancelled[/]")
                                continue
                            import os
                            model_dir = os.path.expanduser(model_dir)
                            model_dir = os.path.abspath(model_dir)
                            upload_model_to_huggingface(
                                model_path=model_dir,
                                model_name=Path(model_dir).name,
                            )
                            continue

                        if upload_source == "Browse and discover models/GGUFs":
                            interactive_model_upload()
                            continue

                        # upload_source == "This decensored model"
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

if __name__ == "__main__":
    main()
