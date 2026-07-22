import shutil
import psutil
import torch
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

class PreflightCheck:
    def __init__(self, model_path: str, device_map: str = "auto"):
        self.model_path = Path(model_path)
        self.device_map = device_map
        self.errors = []
        self.warnings = []

    def check_disk_space(self, required_gb: float = 10.0) -> bool:
        """Check if there is enough free disk space."""
        total, used, free = shutil.disk_usage(".")
        free_gb = free // (2**30)
        
        if free_gb < required_gb:
            self.warnings.append(f"Low disk space: {free_gb}GB free. Recommended: {required_gb}GB+")
            return False
        return True

    def check_memory(self) -> bool:
        """Check available RAM and VRAM."""
        # System RAM
        vm = psutil.virtual_memory()
        available_gb = vm.available / (1024**3)
        total_gb = vm.total / (1024**3)
        
        if available_gb < 8.0:
            self.warnings.append(f"Low system RAM: {available_gb:.1f}GB available (Total: {total_gb:.1f}GB)")

        # GPU VRAM
        # GPU VRAM
        if torch.cuda.is_available():
            pass
            # for i in range(torch.cuda.device_count()):
            #     props = torch.cuda.get_device_properties(i)
            #     total_vram = props.total_memory / (1024**3)
            #     # We can't easily check 'free' VRAM without pynvml, but total is useful context
            #     # print(f"GPU {i}: {props.name} ({total_vram:.1f}GB VRAM)")
        elif torch.backends.mps.is_available():
            # MPS shares unified memory, roughly checked by system RAM check
            pass
        else:
            self.warnings.append("No GPU detected (CUDA/MPS). Inference will be very slow.")

        return True

    def check_model_config(self) -> bool:
        """Validate model configuration presence."""
        # If it's a local path, check for config.json
        if self.model_path.exists():
             if not (self.model_path / "config.json").exists():
                 self.errors.append(f"Model directory '{self.model_path}' missing 'config.json'")
                 return False
        # If it's a HF ID, we assume it's valid or will fail at download time
        return True

    def run_checks(self) -> bool:
        """Run all checks and print report."""
        console.print("[bold cyan]Running Pre-flight Checks...[/]")
        
        self.check_disk_space()
        self.check_memory()
        self.check_model_config()

        passed = len(self.errors) == 0
        
        if passed and not self.warnings:
            console.print("[green]✓ All checks passed[/]")
            return True

        if self.warnings:
            console.print(Panel("\n".join(self.warnings), title="[yellow]Warnings[/]", border_style="yellow"))
        
        if self.errors:
            console.print(Panel("\n".join(self.errors), title="[red]Errors[/]", border_style="red"))
            return False
            
        return True
