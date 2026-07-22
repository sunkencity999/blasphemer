# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Capability healing via DPO.

Abliteration removes refusals by projecting a direction out of the model's
weights, which also nicks general capabilities (mlabonne measured ~0.8 avg
benchmark points on Llama-3-8B, worst on TruthfulQA). A short Direct Preference
Optimization (DPO) pass on a general preference dataset recovers most of that
loss — the NeuralDaredevil recipe: one epoch of LoRA DPO on orpo-dpo-mix-40k.

We train a LoRA adapter with the abliterated model itself as the (implicit)
DPO reference — TRL disables the adapter to obtain reference log-probs — so
healing nudges the model back toward capable, preferred responses without
undoing the abliteration or drifting far from it.

Note: reported benchmark recovery is self-reported on a narrow suite, so treat
this as "recovers most residual degradation," not a guarantee across all tasks.
"""

import time
from pathlib import Path
from typing import Optional

import torch
from rich import print

from .config import Settings

# The columns TRL's DPOTrainer understands for a preference dataset.
_DPO_COLUMNS = ("prompt", "chosen", "rejected")


def preference_columns_to_keep(available: list[str]) -> list[str]:
    """
    Given a dataset's columns, return the subset relevant to DPO.

    TRL fails or wastes work on unrelated columns (source, id, score, ...), so
    we keep only prompt/chosen/rejected. ``chosen``/``rejected`` are required;
    ``prompt`` is kept when present (explicit-prompt format) and omitted
    otherwise (TRL extracts the shared prefix as an implicit prompt).
    """
    return [c for c in _DPO_COLUMNS if c in available]


class Healer:
    """Recover post-abliteration capabilities with a LoRA DPO pass."""

    def __init__(self, model, tokenizer, settings: Settings):
        # Unwrap a torch.compile()'d module so PEFT sees the real model.
        self.model = getattr(model, "_orig_mod", model)
        self.tokenizer = tokenizer
        self.settings = settings

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_dataset(self):
        from datasets import load_dataset

        settings = self.settings
        print(f"[cyan]Loading preference dataset: {settings.heal_dataset}[/]")
        dataset = load_dataset(settings.heal_dataset, split="train")

        keep = preference_columns_to_keep(dataset.column_names)
        if "chosen" not in keep or "rejected" not in keep:
            raise ValueError(
                f"Dataset '{settings.heal_dataset}' lacks 'chosen'/'rejected' columns "
                f"(found {dataset.column_names}); it is not a usable preference dataset."
            )

        drop = [c for c in dataset.column_names if c not in keep]
        if drop:
            dataset = dataset.remove_columns(drop)

        if settings.heal_samples and settings.heal_samples > 0:
            n = min(settings.heal_samples, len(dataset))
            dataset = dataset.select(range(n))

        print(f"[green]✓ Using {len(dataset)} preference pairs[/]")
        return dataset

    def run(self, output_dir: Optional[str] = None) -> Optional[str]:
        """
        Run the DPO healing pass and return the path to the healed model.

        Returns the merged-model path when ``merge_lora`` is set, otherwise the
        adapter path. Returns ``None`` if healing is cancelled or fails.
        """
        from peft import LoraConfig, TaskType
        from trl import DPOConfig, DPOTrainer

        settings = self.settings
        output_dir = output_dir or settings.heal_output_dir
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        print("\n[bold cyan]═══════════════════════════════════════════[/]")
        print("[bold cyan]        DPO Capability Healing             [/]")
        print("[bold cyan]═══════════════════════════════════════════[/]\n")

        try:
            train_dataset = self._load_dataset()
        except Exception as error:
            print(f"[red]Could not load healing dataset: {error}[/]")
            return None

        lora_config = LoraConfig(
            r=settings.lora_rank,
            lora_alpha=settings.lora_alpha,
            target_modules=settings.lora_target_modules,
            lora_dropout=settings.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # fp16/bf16 mixed precision is disabled for MPS stability, mirroring the
        # LoRA knowledge-injection trainer.
        on_mps = torch.backends.mps.is_available()

        dpo_config = DPOConfig(
            output_dir=str(out),
            num_train_epochs=settings.heal_epochs,
            per_device_train_batch_size=settings.per_device_train_batch_size,
            gradient_accumulation_steps=settings.gradient_accumulation_steps,
            learning_rate=settings.heal_learning_rate,
            beta=settings.heal_beta,
            max_length=settings.heal_max_length,
            max_prompt_length=settings.heal_max_prompt_length,
            warmup_ratio=settings.warmup_ratio,
            logging_steps=10,
            save_strategy="no",
            lr_scheduler_type="linear",
            optim="adamw_torch",
            report_to="none",
            bf16=False,
            fp16=False,
            gradient_checkpointing=False,
            remove_unused_columns=False,
        )

        print("[cyan]Healing configuration:[/]")
        print(f"  Dataset: {settings.heal_dataset}")
        print(f"  Pairs: {len(train_dataset)}")
        print(f"  Epochs: {settings.heal_epochs}")
        print(f"  Learning rate: {settings.heal_learning_rate}")
        print(f"  DPO beta: {settings.heal_beta}")
        print(f"  Device: {'MPS' if on_mps else 'CPU/CUDA'}")

        # ref_model=None + peft_config: TRL uses the adapter-disabled base model
        # (i.e. the abliterated model) as the DPO reference.
        trainer = DPOTrainer(
            model=self.model,
            ref_model=None,
            args=dpo_config,
            train_dataset=train_dataset,
            processing_class=self.tokenizer,
            peft_config=lora_config,
        )

        print("\n[bold green]Healing in progress...[/]")
        start = time.time()
        try:
            result = trainer.train()
            print("\n[bold green]✓ Healing complete![/]")
            print(f"  Time: {(time.time() - start) / 60:.1f} minutes")
            print(f"  Final loss: {result.training_loss:.4f}")
        except KeyboardInterrupt:
            print("\n[yellow]Healing interrupted by user; saving current state...[/]")

        adapter_path = out / "adapter"
        trainer.save_model(str(adapter_path))
        self.tokenizer.save_pretrained(str(adapter_path))
        print(f"[green]✓ Adapter saved to {adapter_path}[/]")

        if not settings.merge_lora:
            return str(adapter_path)

        print("\n[cyan]Merging healed adapter into the model...[/]")
        merged_path = out / "merged"
        merged_path.mkdir(parents=True, exist_ok=True)
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(str(merged_path))
        self.tokenizer.save_pretrained(str(merged_path))
        print(f"[green]✓ Healed model saved to {merged_path}[/]")

        return str(merged_path)
