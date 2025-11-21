"""
Fine-tuning orchestration module.
Ties together dataset processing and LoRA training.
"""

from pathlib import Path
from typing import Optional

from rich import print

from .config import Settings
from .dataset_processor import DatasetProcessor
from .lora_trainer import LoRATrainer


class FineTuner:
    """Orchestrate the complete fine-tuning process."""
    
    def __init__(
        self,
        model,
        tokenizer,
        settings: Settings,
    ):
        """
        Initialize fine-tuner.
        
        Args:
            model: Model to fine-tune (base or abliterated)
            tokenizer: Model tokenizer
            settings: Configuration settings
        """
        self.model = model
        self.tokenizer = tokenizer
        self.settings = settings
        
        self.dataset_processor = DatasetProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            train_test_split=0.9,
        )
        
        self.lora_trainer = LoRATrainer(
            model=model,
            tokenizer=tokenizer,
            settings=settings,
        )
    
    def run(
        self,
        dataset_source: str,
        preview_data: bool = True,
    ) -> str:
        """
        Run complete fine-tuning pipeline.
        
        Args:
            dataset_source: Path to data or HuggingFace dataset name
            preview_data: Show data preview before training
        
        Returns:
            Path to trained adapter
        """
        print("\n[bold cyan]═══════════════════════════════════════════[/]")
        print("[bold cyan]        LoRA Fine-Tuning Pipeline          [/]")
        print("[bold cyan]═══════════════════════════════════════════[/]\n")
        
        # Step 1: Process dataset
        print("[bold]Step 1: Processing Dataset[/]")
        datasets = self.dataset_processor.process(
            source=dataset_source,
            tokenizer=self.tokenizer,
        )
        
        # Preview data
        if preview_data:
            self.dataset_processor.preview_examples(datasets, num_examples=2)
            
            # Ask for confirmation
            print("\n[yellow]Review the examples above.[/]")
            response = input("Continue with training? [y/N]: ").strip().lower()
            if response != 'y':
                print("[red]Training cancelled by user.[/]")
                return None
        
        # Step 2: Prepare model with LoRA
        print("\n[bold]Step 2: Preparing Model[/]")
        self.lora_trainer.prepare_model()
        
        # Step 3: Train
        print("\n[bold]Step 3: Training LoRA Adapter[/]")
        adapter_path = self.lora_trainer.train(
            train_dataset=datasets['train'],
            eval_dataset=datasets['validation'],
            output_dir=self.settings.finetuning_output_dir,
        )
        
        # Step 4: Test knowledge (optional)
        if self.settings.test_after_training:
            print("\n[bold]Step 4: Testing Knowledge Injection[/]")
            
            # Generate test prompts from first few examples
            test_prompts = [
                datasets['train'][i]['instruction']
                for i in range(min(3, len(datasets['train'])))
            ]
            
            self.lora_trainer.test_knowledge(test_prompts)
        
        # Step 5: Merge if requested
        if self.settings.merge_lora:
            print("\n[bold]Step 5: Merging LoRA Adapter[/]")
            
            merged_path = str(Path(self.settings.finetuning_output_dir) / "merged")
            self.lora_trainer.merge_adapter(
                adapter_path=adapter_path,
                output_path=merged_path,
            )
            
            print(f"\n[bold green]✓ Fine-tuning complete![/]")
            print(f"  Adapter: {adapter_path}")
            print(f"  Merged model: {merged_path}")
            
            return merged_path
        else:
            print(f"\n[bold green]✓ Fine-tuning complete![/]")
            print(f"  Adapter: {adapter_path}")
            print(f"  [dim]To merge later, use --merge-lora flag[/]")
            
            return adapter_path
    
    def merge_existing_adapter(
        self,
        adapter_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Merge an existing LoRA adapter.
        
        Args:
            adapter_path: Path to adapter to merge
            output_path: Where to save merged model
        
        Returns:
            Path to merged model
        """
        if output_path is None:
            output_path = str(Path(adapter_path).parent / "merged")
        
        print(f"\n[cyan]Merging adapter: {adapter_path}[/]")
        
        self.lora_trainer.merge_adapter(
            adapter_path=adapter_path,
            output_path=output_path,
        )
        
        return output_path
