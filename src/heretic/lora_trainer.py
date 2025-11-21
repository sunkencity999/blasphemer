"""
LoRA training module using PEFT.
Handles the core training loop with progress tracking.
"""

import os
import time
from typing import Dict, Optional
from pathlib import Path

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from .config import Settings


class LoRATrainer:
    """Train LoRA adapters for knowledge injection."""
    
    def __init__(
        self,
        model,
        tokenizer,
        settings: Settings,
    ):
        """
        Initialize LoRA trainer.
        
        Args:
            model: Base model to train
            tokenizer: Model tokenizer
            settings: Configuration settings
        """
        self.model = model
        self.tokenizer = tokenizer
        self.settings = settings
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_model(self) -> None:
        """Prepare model for LoRA training."""
        print("[cyan]Preparing model for LoRA training...[/]")
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=self.settings.lora_rank,
            lora_alpha=self.settings.lora_alpha,
            target_modules=self.settings.lora_target_modules,
            lora_dropout=self.settings.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Disable KV cache for training (avoids compatibility issues)
        self.model.config.use_cache = False
        
        # Enable gradient checkpointing if available
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        
        # Ensure model is in training mode
        self.model.train()
        
        # Print trainable parameters
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"[green]✓ LoRA adapters applied[/]")
        print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        print(f"  Total parameters: {total_params:,}")
    
    def train(
        self,
        train_dataset,
        eval_dataset,
        output_dir: str,
    ) -> str:
        """
        Train LoRA adapter.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Validation dataset
            output_dir: Directory to save checkpoints
        
        Returns:
            Path to best model checkpoint
        """
        print("\n[bold cyan]Starting LoRA Training[/]")
        print("=" * 80)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Tokenize datasets
        print("[cyan]Tokenizing datasets...[/]")
        
        def tokenize_function(examples):
            # Combine instruction and response
            prompts = [
                f"### Instruction:\n{inst}\n\n### Response:\n{resp}"
                for inst, resp in zip(examples['instruction'], examples['response'])
            ]
            
            # Tokenize
            tokenized = self.tokenizer(
                prompts,
                truncation=True,
                max_length=self.settings.max_seq_length,
                padding=False,
            )
            
            # Labels are same as input_ids for causal LM
            tokenized['labels'] = tokenized['input_ids'].copy()
            
            return tokenized
        
        tokenized_train = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing training data",
        )
        
        tokenized_eval = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing validation data",
        )
        
        print(f"[green]✓ Tokenization complete[/]")
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Calculate training steps
        total_steps = (
            len(tokenized_train)
            // self.settings.per_device_train_batch_size
            // self.settings.gradient_accumulation_steps
            * self.settings.num_train_epochs
        )
        
        print(f"[cyan]Training configuration:[/]")
        print(f"  Epochs: {self.settings.num_train_epochs}")
        print(f"  Batch size: {self.settings.per_device_train_batch_size}")
        print(f"  Gradient accumulation: {self.settings.gradient_accumulation_steps}")
        print(f"  Effective batch size: {self.settings.per_device_train_batch_size * self.settings.gradient_accumulation_steps}")
        print(f"  Learning rate: {self.settings.learning_rate}")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {int(total_steps * self.settings.warmup_ratio)}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=self.settings.num_train_epochs,
            per_device_train_batch_size=self.settings.per_device_train_batch_size,
            per_device_eval_batch_size=self.settings.per_device_eval_batch_size,
            gradient_accumulation_steps=self.settings.gradient_accumulation_steps,
            learning_rate=self.settings.learning_rate,
            warmup_ratio=self.settings.warmup_ratio,
            logging_steps=10,
            save_steps=self.settings.save_steps,
            save_strategy="steps",
            eval_strategy="steps",  # Must match save_strategy when load_best_model_at_end=True
            eval_steps=self.settings.save_steps,
            save_total_limit=self.settings.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=False,  # Disabled for MPS compatibility - can cause gradient issues
            gradient_checkpointing=False,  # Disabled for model compatibility (e.g., Phi-3)
            optim="adamw_torch",
            lr_scheduler_type="linear",
            report_to="none",  # Disable wandb/tensorboard
            remove_unused_columns=False,
            save_safetensors=True,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
        )
        
        # Train
        print("\n[bold green]Training in progress...[/]")
        start_time = time.time()
        
        try:
            train_result = trainer.train()
            
            training_time = time.time() - start_time
            print(f"\n[bold green]✓ Training complete![/]")
            print(f"  Time: {training_time / 60:.1f} minutes")
            print(f"  Final loss: {train_result.training_loss:.4f}")
            
        except KeyboardInterrupt:
            print("\n[yellow]Training interrupted by user[/]")
            print("[cyan]Saving current checkpoint...[/]")
            trainer.save_model()
        
        # Save final model
        final_path = output_path / "final"
        trainer.save_model(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))
        
        print(f"[green]✓ Model saved to {final_path}[/]")
        
        # Evaluate on validation set
        print("\n[cyan]Running final evaluation...[/]")
        eval_results = trainer.evaluate()
        
        print(f"[green]✓ Evaluation complete[/]")
        print(f"  Validation loss: {eval_results['eval_loss']:.4f}")
        print(f"  Validation perplexity: {torch.exp(torch.tensor(eval_results['eval_loss'])):.2f}")
        
        return str(final_path)
    
    def merge_adapter(
        self,
        adapter_path: str,
        output_path: str,
    ) -> None:
        """
        Merge LoRA adapter into base model.
        
        Args:
            adapter_path: Path to LoRA adapter
            output_path: Path to save merged model
        """
        print(f"\n[cyan]Merging LoRA adapter...[/]")
        
        from peft import PeftModel
        
        # Load adapter
        model_with_adapter = PeftModel.from_pretrained(
            self.model,
            adapter_path,
        )
        
        # Merge
        merged_model = model_with_adapter.merge_and_unload()
        
        # Save
        output = Path(output_path)
        output.mkdir(parents=True, exist_ok=True)
        
        merged_model.save_pretrained(str(output))
        self.tokenizer.save_pretrained(str(output))
        
        print(f"[green]✓ Merged model saved to {output}[/]")
    
    def test_knowledge(
        self,
        test_prompts: list[str],
    ) -> None:
        """
        Test model's knowledge with sample prompts.
        
        Args:
            test_prompts: List of prompts to test
        """
        print("\n[bold cyan]Testing Knowledge Injection:[/]")
        print("=" * 80)
        
        self.model.eval()
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n[bold]Test {i}:[/]")
            print(f"[yellow]Prompt:[/] {prompt}")
            
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove prompt from response
            response = response[len(prompt):].strip()
            
            print(f"[green]Response:[/] {response[:500]}...")
            print("-" * 80)
