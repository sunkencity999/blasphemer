"""
Enhanced progress display and observability for Blasphemer optimization runs.

Provides real-time quality metrics, trend analysis, and visual feedback during
long-running abliteration processes.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
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
    """Tracks and displays optimization progress with quality metrics."""
    
    def __init__(self, total_trials: int, model_name: str):
        self.total_trials = total_trials
        self.model_name = model_name
        self.console = Console()
        self.trial_history: List[TrialMetrics] = []
        self.start_time = time.time()
        self.best_trial: Optional[TrialMetrics] = None
        
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
    
    def display_progress(
        self,
        current_trial: int,
        current_kl: Optional[float] = None,
        current_refusals: Optional[int] = None,
        current_params: Optional[dict] = None
    ):
        """Display enhanced progress information."""
        self.console.clear()
        
        # Header
        header = Panel(
            f"[bold cyan]Blasphemer Optimization Progress[/]\n"
            f"[dim]Model: {self.model_name}[/]",
            box=box.DOUBLE,
            border_style="cyan"
        )
        self.console.print(header)
        self.console.print()
        
        # Progress bar
        progress_pct = (current_trial / self.total_trials) * 100
        elapsed = time.time() - self.start_time
        
        if current_trial > 0:
            time_per_trial = elapsed / current_trial
            remaining = time_per_trial * (self.total_trials - current_trial)
            remaining_str = self._format_duration(remaining)
        else:
            remaining_str = "calculating..."
        
        self.console.print(
            f"[bold]Trial {current_trial}/{self.total_trials}[/] "
            f"([cyan]{progress_pct:.1f}%[/], ~{remaining_str} remaining)"
        )
        
        # Progress bar visual
        bar_width = 40
        filled = int(bar_width * progress_pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        self.console.print(f"[cyan]{bar}[/]")
        self.console.print()
        
        # Current trial info
        if current_kl is not None and current_refusals is not None:
            self.console.print("[bold]Current Trial:[/]")
            
            if current_params:
                # Show one parameter example
                param_sample = list(current_params.items())[0] if current_params else None
                if param_sample:
                    self.console.print(f"  Parameters: {param_sample[0]} (and others)")
            
            self.console.print(f"  KL Divergence: [yellow]{current_kl:.3f}[/]")
            self.console.print(f"  Refusals: [yellow]{current_refusals}[/]")
            self.console.print()
        
        # Best trial so far
        if self.best_trial:
            trend_direction, trend_symbol = self.get_trend()
            trend_color = "green" if trend_direction == "improving" else "red" if trend_direction == "degrading" else "yellow"
            
            self.console.print("[bold]Best Trial So Far:[/]")
            self.console.print(f"  Trial: [green]#{self.best_trial.trial_number}[/]")
            self.console.print(
                f"  KL Divergence: [green]{self.best_trial.kl_divergence:.3f}[/] "
                f"[{trend_color}]({trend_symbol} {trend_direction})[/]"
            )
            self.console.print(
                f"  Refusals: [green]{self.best_trial.refusals}/{self.best_trial.total_prompts}[/] "
                f"([green]{self.best_trial.refusal_rate:.1f}%[/])"
            )
            
            # Quality visualization
            quality_bar = self.get_quality_bar(self.best_trial.kl_divergence)
            self.console.print(f"  Quality: {quality_bar}")
            self.console.print()
        
        # Trend analysis
        if len(self.trial_history) >= 10:
            trend_direction, trend_symbol = self.get_trend()
            trend_color = "green" if trend_direction == "improving" else "red" if trend_direction == "degrading" else "yellow"
            
            self.console.print(f"[bold]Trend:[/] [{trend_color}]{trend_symbol} {trend_direction.upper()}[/]")
            self.console.print()
        
        # Predicted outcome
        if self.best_trial and len(self.trial_history) >= 20:
            quality_level, description = self.predict_outcome()
            quality_colors = {
                "excellent": "green",
                "very_good": "cyan",
                "good": "yellow",
                "acceptable": "magenta",
                "poor": "red"
            }
            color = quality_colors.get(quality_level, "white")
            
            self.console.print(f"[bold]Expected Outcome:[/] [{color}]{description}[/]")
            self.console.print()
        
        # Stats summary
        if len(self.trial_history) >= 5:
            recent_5 = self.trial_history[-5:]
            avg_kl = sum(t.kl_divergence for t in recent_5) / len(recent_5)
            avg_refusal = sum(t.refusal_rate for t in recent_5) / len(recent_5)
            
            self.console.print(f"[dim]Recent average (last 5): KL {avg_kl:.3f}, Refusals {avg_refusal:.1f}%[/]")
    
    def display_completion_summary(self):
        """Display final summary after optimization completes."""
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
        
        self.console.print(f"[bold]Summary:[/]")
        self.console.print(f"  Total trials: {len(self.trial_history)}")
        self.console.print(f"  Total time: {self._format_duration(total_time)}")
        self.console.print(f"  Avg per trial: {self._format_duration(avg_time_per_trial)}")
        self.console.print()
        
        # Best trial
        if self.best_trial:
            self.console.print(f"[bold green]Best Result:[/]")
            self.console.print(f"  Trial: [green]#{self.best_trial.trial_number}[/]")
            self.console.print(f"  KL Divergence: [green]{self.best_trial.kl_divergence:.3f}[/]")
            self.console.print(f"  Refusals: [green]{self.best_trial.refusals}/{self.best_trial.total_prompts}[/] ([green]{self.best_trial.refusal_rate:.1f}%[/])")
            
            quality_level, description = self.predict_outcome()
            quality_colors = {
                "excellent": "green",
                "very_good": "cyan",
                "good": "yellow",
                "acceptable": "magenta",
                "poor": "red"
            }
            color = quality_colors.get(quality_level, "white")
            self.console.print(f"  Quality: [{color}]{description}[/]")
            self.console.print()
        
        # Top 5 trials table
        if len(self.trial_history) >= 5:
            self.console.print("[bold]Top 5 Trials:[/]")
            table = Table(show_header=True, header_style="bold cyan", box=box.SIMPLE)
            table.add_column("Trial", style="cyan", justify="right")
            table.add_column("KL Div", justify="right")
            table.add_column("Refusals", justify="right")
            table.add_column("Quality", justify="center")
            
            sorted_trials = sorted(
                self.trial_history,
                key=lambda t: (0.6 * t.kl_divergence) + (0.4 * t.refusal_rate / 100)
            )[:5]
            
            for i, trial in enumerate(sorted_trials):
                quality_bar = self.get_quality_bar(trial.kl_divergence)
                star = "⭐" if i == 0 else ""
                table.add_row(
                    f"#{trial.trial_number} {star}",
                    f"{trial.kl_divergence:.3f}",
                    f"{trial.refusals} ({trial.refusal_rate:.1f}%)",
                    quality_bar
                )
            
            self.console.print(table)
            self.console.print()
    
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
