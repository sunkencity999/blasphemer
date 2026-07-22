"""
Enhanced progress display and observability for Blasphemer optimization runs.

Provides real-time quality metrics, trend analysis, and visual feedback during
long-running abliteration processes.
"""

import time
import psutil
import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.align import Align
from rich import box


@dataclass
class TrialMetrics:
    """Metrics for a single trial."""
    trial_number: int
    kl_divergence: float
    refusals: int
    total_prompts: int
    parameters: dict
    timestamp: float
    
    @property
    def refusal_rate(self) -> float:
        """Refusal rate as a percentage."""
        return (self.refusals / self.total_prompts * 100) if self.total_prompts > 0 else 0.0


class ProgressTracker:
    """Tracks and displays optimization progress with quality metrics using Live display."""
    
    def __init__(self, total_trials: int, model_name: str):
        self.total_trials = total_trials
        self.model_name = model_name
        self.console = Console()
        self.trial_history: List[TrialMetrics] = []
        self.start_time = time.time()
        self.best_trial: Optional[TrialMetrics] = None
        self.current_status = "Initializing..."
        self.live = None
        
    def __enter__(self):
        self.live = Live(self._generate_layout(), refresh_per_second=4, console=self.console)
        self.live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.live:
            self.live.stop()

    def update_status(self, status: str):
        """Update the current status message."""
        self.current_status = status
        # Explicit refresh not needed as Live handles it, but ensures immediate update if outside loop
        if self.live:
            self.live.update(self._generate_layout())

    def add_trial(
        self,
        trial_number: int,
        kl_divergence: float,
        refusals: int,
        total_prompts: int,
        parameters: dict
    ):
        """Add a completed trial to the history."""
        metrics = TrialMetrics(
            trial_number=trial_number,
            kl_divergence=kl_divergence,
            refusals=refusals,
            total_prompts=total_prompts,
            parameters=parameters,
            timestamp=time.time()
        )
        
        self.trial_history.append(metrics)
        
        # Update best trial
        if self.best_trial is None or self._is_better(metrics, self.best_trial):
            self.best_trial = metrics
            
        self.update_status(f"Completed trial {trial_number}/{self.total_trials}")
    
    def _is_better(self, trial1: TrialMetrics, trial2: TrialMetrics) -> bool:
        """Determine if trial1 is better than trial2."""
        # Lower KL divergence and lower refusals is better
        # Weight KL divergence more heavily (60/40 split)
        score1 = (0.6 * trial1.kl_divergence) + (0.4 * trial1.refusal_rate / 100)
        score2 = (0.6 * trial2.kl_divergence) + (0.4 * trial2.refusal_rate / 100)
        return score1 < score2

    def get_trend(self, window: int = 10) -> Tuple[str, str]:
        """
        Analyze recent trend in quality metrics.
        
        Returns:
            Tuple of (trend_direction, trend_symbol)
            - trend_direction: "improving", "degrading", "stable"
            - trend_symbol: "▲", "▼", "▬"
        """
        if len(self.trial_history) < window:
            return "insufficient_data", "?"
        
        recent_trials = self.trial_history[-window:]
        first_half = recent_trials[:window // 2]
        second_half = recent_trials[window // 2:]
        
        # Average score for each half
        first_avg = sum(
            (0.6 * t.kl_divergence) + (0.4 * t.refusal_rate / 100)
            for t in first_half
        ) / len(first_half)
        
        second_avg = sum(
            (0.6 * t.kl_divergence) + (0.4 * t.refusal_rate / 100)
            for t in second_half
        ) / len(second_half)
        
        improvement = first_avg - second_avg
        
        if improvement > 0.05:  # Significant improvement
            return "improving", "▼"
        elif improvement < -0.05:  # Significant degradation
            return "degrading", "▲"
        else:
            return "stable", "▬"

    def get_quality_bar(self, value: float, max_value: float = 1.0) -> str:
        """Generate a visual quality bar."""
        filled = int((1 - (value / max_value)) * 10)
        filled = max(0, min(10, filled))
        empty = 10 - filled
        return "█" * filled + "▓" * (empty // 2) + "░" * (empty - empty // 2)

    def predict_outcome(self) -> Tuple[str, str]:
        """
        Predict expected outcome quality based on trends.
        
        Returns:
            Tuple of (quality_level, description)
        """
        if not self.best_trial:
            return "unknown", "Insufficient data"
        
        kl = self.best_trial.kl_divergence
        refusal_rate = self.best_trial.refusal_rate
        
        # Quality thresholds based on research and community standards
        if kl < 0.15 and refusal_rate < 1.0:
            return "excellent", "Excellent - High quality with minimal refusals"
        elif kl < 0.25 and refusal_rate < 2.5:
            return "very_good", "Very Good - Good balance of quality and safety removal"
        elif kl < 0.40 and refusal_rate < 5.0:
            return "good", "Good - Acceptable quality trade-off"
        elif kl < 0.60 and refusal_rate < 10.0:
            return "acceptable", "Acceptable - Noticeable quality impact"
        else:
            return "poor", "Poor - Significant quality degradation"

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        seconds = int(seconds)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def _get_resource_usage(self) -> Tuple[float, float, str]:
        """Get RAM and VRAM usage."""
        # System RAM
        vm = psutil.virtual_memory()
        ram_used = vm.percent

        # VRAM (if available)
        vram_info = "N/A"
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
            vram_info = f"{vram_used:.1f}%"
        elif torch.backends.mps.is_available():
            # MPS shares system memory, so it's correlated with RAM
            vram_info = "Shared"
            
        return ram_used, vram_info

    def _generate_layout(self) -> Layout:
        """Generate the main dashboard layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        layout["header"].update(self._make_header())
        layout["main"].split_row(
            Layout(name="stats", ratio=2),
            Layout(name="history", ratio=3)
        )
        
        layout["stats"].update(self._make_stats_panel())
        layout["history"].update(self._make_history_panel())
        layout["footer"].update(self._make_footer())
        
        return layout

    def _make_header(self) -> Panel:
        trials_done = len(self.trial_history)
        progress = (trials_done / self.total_trials) * 100
        
        return Panel(
            Align.center(
                f"[bold cyan]Blasphemer optimization[/] • "
                f"Model: [white]{self.model_name}[/] • "
                f"Progress: [green]{progress:.1f}%[/] ({trials_done}/{self.total_trials})"
            ),
            style="cyan",
            box=box.ROUNDED
        )

    def _make_stats_panel(self) -> Panel:
        ram_used, vram_info = self._get_resource_usage()
        
        stats_table = Table.grid(padding=(0, 2))
        stats_table.add_column(style="bold cyan")
        stats_table.add_column(justify="right")
        
        stats_table.add_row("Status:", f"[yellow]{self.current_status}[/]")
        stats_table.add_row("Time Elapsed:", f"{int(time.time() - self.start_time)}s")
        stats_table.add_row("System RAM:", f"{ram_used:.1f}%")
        stats_table.add_row("GPU VRAM:", vram_info)
        
        stats_table.add_row("", "") # Spacer
        
        if self.best_trial:
            stats_table.add_row("[bold green]Best Result:[/]", "")
            stats_table.add_row("Trial #:", str(self.best_trial.trial_number))
            stats_table.add_row("KL Divergence:", f"{self.best_trial.kl_divergence:.4f}")
            stats_table.add_row("Refusal Rate:", f"{self.best_trial.refusal_rate:.1f}%")
        else:
            stats_table.add_row("[dim]No results yet...[/]", "")

        return Panel(stats_table, title="Current Status", border_style="blue")

    def _make_history_panel(self) -> Panel:
        table = Table(box=box.SIMPLE_HEAD, expand=True)
        table.add_column("Trial", justify="right", style="cyan", width=6)
        table.add_column("KL Div", justify="right", width=10)
        table.add_column("Refusals", justify="right", width=10)
        table.add_column("Rate", justify="right", width=8)

        # Show last 5 trials
        recent = self.trial_history[-5:]
        for t in reversed(recent):
            table.add_row(
                str(t.trial_number),
                f"{t.kl_divergence:.4f}",
                str(t.refusals),
                f"{t.refusal_rate:.1f}%"
            )
            
        return Panel(table, title="Recent Trials", border_style="blue")

    def _make_footer(self) -> Align:
        return Align.center("[dim]Press Ctrl+C to stop early and save best result[/]")

    def display_completion_summary(self):
        """Display final summary after optimization completes."""
        # Stop live display if running
        if self.live:
            self.live.stop()
            
        self.console.clear()
        self.console.print()
        
        # Header
        header = Panel(
            f"[bold green]✓ Optimization Complete![/]\n"
            f"[dim]Model: {self.model_name}[/]",
            box=box.DOUBLE,
            border_style="green"
        )
        self.console.print(header)
        self.console.print()
        
        # Stats
        total_time = time.time() - self.start_time
        avg_time_per_trial = total_time / len(self.trial_history) if self.trial_history else 0
        
        self.console.print("[bold]Summary:[/]")
        self.console.print(f"  Total trials: {len(self.trial_history)}")
        self.console.print(f"  Total time: {total_time:.1f}s")
        self.console.print(f"  Avg per trial: {avg_time_per_trial:.1f}s")
        self.console.print()
        
        # Best trial
        if self.best_trial:
            self.console.print("[bold green]Best Result:[/]")
            self.console.print(f"  Trial: [green]#{self.best_trial.trial_number}[/]")
            self.console.print(f"  KL Divergence: [green]{self.best_trial.kl_divergence:.3f}[/]")
            self.console.print(f"  Refusals: [green]{self.best_trial.refusals}/{self.best_trial.total_prompts}[/] ([green]{self.best_trial.refusal_rate:.1f}%[/])")
            self.console.print()
        
        # Top 5 trials table
        if len(self.trial_history) >= 5:
            self.console.print("[bold]Top 5 Trials:[/]")
            table = Table(show_header=True, header_style="bold cyan", box=box.SIMPLE)
            table.add_column("Trial", style="cyan", justify="right")
            table.add_column("KL Div", justify="right")
            table.add_column("Refusals", justify="right")
            
            sorted_trials = sorted(
                self.trial_history,
                key=lambda t: (0.6 * t.kl_divergence) + (0.4 * t.refusal_rate / 100)
            )[:5]
            
            for i, trial in enumerate(sorted_trials):
                star = "⭐" if i == 0 else ""
                table.add_row(
                    f"#{trial.trial_number} {star}",
                    f"{trial.kl_divergence:.3f}",
                    f"{trial.refusals} ({trial.refusal_rate:.1f}%)"
                )
            
            self.console.print(table)
            self.console.print()
