import hashlib
import optuna
import torch.nn.functional as F
from dataclasses import asdict
from pathlib import Path
from typing import Optional
from optuna.samplers import TPESampler
from optuna.study import StudyDirection
from optuna import Trial

from heretic.model import Model, AbliterationParameters
from heretic.config import Settings
from heretic.evaluator import Evaluator
from heretic.progress import ProgressTracker
from heretic.utils import get_trial_parameters, print

class Optimizer:
    def __init__(self, settings: Settings, model: Model, evaluator: Evaluator):
        self.settings = settings
        self.model = model
        self.evaluator = evaluator
        self.trial_index = 0
        self.saved_weights = None
        self.refusal_directions = None

    def prepare(self):
        """Calculate necessary vectors before optimization."""
        print()
        print("Calculating per-layer refusal directions...")
        print("* Obtaining residuals for good prompts...")
        good_prompts = self.evaluator.good_prompts
        good_residuals = self.model.get_residuals_batched(good_prompts)
        
        print("* Obtaining residuals for bad prompts...")
        bad_prompts = self.evaluator.bad_prompts
        bad_residuals = self.model.get_residuals_batched(bad_prompts)
        
        self.refusal_directions = F.normalize(
            bad_residuals.mean(dim=0) - good_residuals.mean(dim=0),
            p=2,
            dim=1,
        )

    def optimize(self) -> Optional[optuna.Study]:
        """Run the optimization process."""
        if self.refusal_directions is None:
            self.prepare()

        # Set up checkpoint directory and study storage
        checkpoint_dir = Path(self.settings.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a unique study name based on the model ID
        model_hash = hashlib.md5(self.settings.model.encode()).hexdigest()[:8]
        study_name = f"blasphemer_{Path(self.settings.model).name}_{model_hash}"
        storage_path = checkpoint_dir / f"{study_name}.db"
        storage_url = f"sqlite:///{storage_path}"

        # Record the exact model identifier next to the checkpoint so the
        # launcher can offer resume-by-selection without the user having to
        # retype it (the study name only preserves the model's basename + hash).
        try:
            (checkpoint_dir / f"{study_name}.model").write_text(self.settings.model)
        except OSError:
            pass
        
        # Check for existing study
        existing_study = None
        if storage_path.exists():
            if self.settings.resume:
                print(f"[bold green]Found existing checkpoint:[/] {storage_path}")
                try:
                    existing_study = optuna.load_study(
                        study_name=study_name,
                        storage=storage_url,
                    )
                    completed_trials = len([t for t in existing_study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                    if completed_trials >= self.settings.n_trials:
                        print("[yellow]Study already completed! Using existing results.[/]")
                        return existing_study
                    print(f"Resuming... {self.settings.n_trials - completed_trials} trials remaining")
                except Exception as error:
                    print(f"[yellow]Warning: Could not load checkpoint ({error}). Starting fresh.[/]")
            else:
                print(f"[yellow]Found existing checkpoint at {storage_path} but --resume not specified. Starting fresh.[/]")
        
        # Create study if not loaded
        if existing_study:
            study = existing_study
            n_trials_to_run = self.settings.n_trials - len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        else:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                sampler=TPESampler(
                    n_startup_trials=self.settings.n_startup_trials,
                    n_ei_candidates=128,
                    multivariate=True,
                ),
                directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE],
                load_if_exists=False,
            )
            n_trials_to_run = self.settings.n_trials

        if n_trials_to_run <= 0:
            return study

        print(f"[bold cyan]Starting optimization:[/] {n_trials_to_run} trials")
        
        # Initialize progress tracker with context manager for Live display
        with ProgressTracker(self.settings.n_trials, self.settings.model) as progress:
            # Sync progress tracker with existing trials if resuming
            if existing_study:
                for t in existing_study.trials:
                    if t.state == optuna.trial.TrialState.COMPLETE:
                         progress.add_trial(
                            trial_number=t.number,
                            kl_divergence=t.user_attrs.get("kl_divergence", 0.0),
                            refusals=t.user_attrs.get("refusals", 0),
                            total_prompts=len(self.evaluator.bad_prompts),
                            parameters=t.user_attrs.get("parameters", {})
                        )
                # Update trial index for the objective function
                self.trial_index = len(study.trials)

            def objective(trial: Trial) -> tuple[float, float]:
                self.trial_index += 1
                trial.set_user_attr("index", self.trial_index)
                
                progress.update_status(f"Running Trial {self.trial_index}/{self.settings.n_trials}")

                # Suggest parameters
                direction_scope = trial.suggest_categorical("direction_scope", ["global", "per layer"])
                direction_index = trial.suggest_float(
                    "direction_index",
                    0.4 * (len(self.model.get_layers()) - 1),
                    0.9 * (len(self.model.get_layers()) - 1),
                )
                if direction_scope == "per layer":
                    direction_index = None

                parameters = {}
                for component in self.model.get_abliterable_components():
                    max_weight = trial.suggest_float(f"{component}.max_weight", 0.8, 1.5)
                    max_weight_position = trial.suggest_float(
                        f"{component}.max_weight_position",
                        0.6 * (len(self.model.get_layers()) - 1),
                        len(self.model.get_layers()) - 1,
                    )
                    min_weight = trial.suggest_float(f"{component}.min_weight", 0.0, 1.0)
                    min_weight_distance = trial.suggest_float(
                        f"{component}.min_weight_distance", 
                        1.0, 
                        0.6 * (len(self.model.get_layers()) - 1)
                    )
                    
                    parameters[component] = AbliterationParameters(
                        max_weight=max_weight,
                        max_weight_position=max_weight_position,
                        min_weight=(min_weight * max_weight),
                        min_weight_distance=min_weight_distance,
                    )

                trial.set_user_attr("direction_index", direction_index)
                trial.set_user_attr("parameters", {k: asdict(v) for k, v in parameters.items()})

                # Optimization: Cache weights
                if self.trial_index == 1 or self.saved_weights is None:
                    progress.update_status("Saving clean model weights...")
                    self.saved_weights = self.model.save_abliterable_weights()
                else:
                    progress.update_status("Restoring weights...")
                    self.model.restore_abliterable_weights(self.saved_weights)
                
                progress.update_status("Abliterating...")
                self.model.abliterate(self.refusal_directions, direction_index, parameters)
                
                progress.update_status("Evaluating model...")
                score, kl_divergence, refusals = self.evaluator.get_score()

                progress.add_trial(
                    trial_number=self.trial_index,
                    kl_divergence=kl_divergence,
                    refusals=refusals,
                    total_prompts=len(self.evaluator.bad_prompts),
                    parameters=get_trial_parameters(trial)
                )

                trial.set_user_attr("kl_divergence", kl_divergence)
                trial.set_user_attr("refusals", refusals)
                trial.set_user_attr("total_prompts", len(self.evaluator.bad_prompts))

                return score

            study.optimize(objective, n_trials=n_trials_to_run)
            progress.display_completion_summary()

        return study
