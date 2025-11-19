"""
Tests for the enhanced progress tracking and observability features.
"""

import pytest
import time
from heretic.progress import ProgressTracker, TrialMetrics


# Disable CLI parsing for all tests
@pytest.fixture(autouse=True)
def disable_cli_parsing(monkeypatch):
    """Disable command-line argument parsing."""
    import sys
    monkeypatch.setattr(sys, 'argv', ['pytest'])


class TestTrialMetrics:
    """Test TrialMetrics dataclass."""
    
    def test_trial_metrics_creation(self):
        """Test creating a TrialMetrics instance."""
        metrics = TrialMetrics(
            trial_number=1,
            kl_divergence=0.25,
            refusals=5,
            total_prompts=200,
            parameters={"test": "param"},
            timestamp=time.time()
        )
        
        assert metrics.trial_number == 1
        assert metrics.kl_divergence == 0.25
        assert metrics.refusals == 5
        assert metrics.total_prompts == 200
    
    def test_refusal_rate_calculation(self):
        """Test refusal rate percentage calculation."""
        metrics = TrialMetrics(
            trial_number=1,
            kl_divergence=0.25,
            refusals=5,
            total_prompts=200,
            parameters={},
            timestamp=time.time()
        )
        
        assert metrics.refusal_rate == 2.5  # 5/200 * 100 = 2.5%
    
    def test_refusal_rate_zero_prompts(self):
        """Test refusal rate handles zero prompts gracefully."""
        metrics = TrialMetrics(
            trial_number=1,
            kl_divergence=0.25,
            refusals=0,
            total_prompts=0,
            parameters={},
            timestamp=time.time()
        )
        
        assert metrics.refusal_rate == 0.0


class TestProgressTracker:
    """Test ProgressTracker functionality."""
    
    def test_tracker_initialization(self):
        """Test creating a ProgressTracker."""
        tracker = ProgressTracker(
            total_trials=100,
            model_name="test-model"
        )
        
        assert tracker.total_trials == 100
        assert tracker.model_name == "test-model"
        assert len(tracker.trial_history) == 0
        assert tracker.best_trial is None
    
    def test_add_trial(self):
        """Test adding a trial to the tracker."""
        tracker = ProgressTracker(
            total_trials=100,
            model_name="test-model"
        )
        
        tracker.add_trial(
            trial_number=1,
            kl_divergence=0.25,
            refusals=5,
            total_prompts=200,
            parameters={"test": "param"}
        )
        
        assert len(tracker.trial_history) == 1
        assert tracker.best_trial is not None
        assert tracker.best_trial.trial_number == 1
    
    def test_best_trial_tracking(self):
        """Test that best trial is correctly identified."""
        tracker = ProgressTracker(
            total_trials=100,
            model_name="test-model"
        )
        
        # Add first trial
        tracker.add_trial(
            trial_number=1,
            kl_divergence=0.50,
            refusals=10,
            total_prompts=200,
            parameters={}
        )
        
        # Add better trial
        tracker.add_trial(
            trial_number=2,
            kl_divergence=0.25,
            refusals=5,
            total_prompts=200,
            parameters={}
        )
        
        # Add worse trial
        tracker.add_trial(
            trial_number=3,
            kl_divergence=0.60,
            refusals=15,
            total_prompts=200,
            parameters={}
        )
        
        # Best trial should be #2
        assert tracker.best_trial.trial_number == 2
        assert tracker.best_trial.kl_divergence == 0.25
    
    def test_is_better_lower_kl(self):
        """Test that lower KL divergence is considered better."""
        tracker = ProgressTracker(
            total_trials=100,
            model_name="test-model"
        )
        
        trial1 = TrialMetrics(
            trial_number=1,
            kl_divergence=0.20,
            refusals=5,
            total_prompts=200,
            parameters={},
            timestamp=time.time()
        )
        
        trial2 = TrialMetrics(
            trial_number=2,
            kl_divergence=0.40,
            refusals=5,
            total_prompts=200,
            parameters={},
            timestamp=time.time()
        )
        
        assert tracker._is_better(trial1, trial2)
        assert not tracker._is_better(trial2, trial1)
    
    def test_is_better_lower_refusals(self):
        """Test that lower refusals is considered better."""
        tracker = ProgressTracker(
            total_trials=100,
            model_name="test-model"
        )
        
        trial1 = TrialMetrics(
            trial_number=1,
            kl_divergence=0.30,
            refusals=2,
            total_prompts=200,
            parameters={},
            timestamp=time.time()
        )
        
        trial2 = TrialMetrics(
            trial_number=2,
            kl_divergence=0.30,
            refusals=10,
            total_prompts=200,
            parameters={},
            timestamp=time.time()
        )
        
        assert tracker._is_better(trial1, trial2)
    
    def test_get_trend_insufficient_data(self):
        """Test trend analysis with insufficient data."""
        tracker = ProgressTracker(
            total_trials=100,
            model_name="test-model"
        )
        
        # Add only 5 trials (need 10 for trend)
        for i in range(5):
            tracker.add_trial(
                trial_number=i+1,
                kl_divergence=0.25,
                refusals=5,
                total_prompts=200,
                parameters={}
            )
        
        trend_direction, trend_symbol = tracker.get_trend()
        assert trend_direction == "insufficient_data"
        assert trend_symbol == "?"
    
    def test_get_trend_improving(self):
        """Test trend detection for improving quality."""
        tracker = ProgressTracker(
            total_trials=100,
            model_name="test-model"
        )
        
        # Add trials that improve over time
        for i in range(10):
            tracker.add_trial(
                trial_number=i+1,
                kl_divergence=0.50 - (i * 0.04),  # Gets better
                refusals=10 - i,  # Gets better
                total_prompts=200,
                parameters={}
            )
        
        trend_direction, trend_symbol = tracker.get_trend()
        assert trend_direction == "improving"
        assert trend_symbol == "▼"
    
    def test_get_trend_degrading(self):
        """Test trend detection for degrading quality."""
        tracker = ProgressTracker(
            total_trials=100,
            model_name="test-model"
        )
        
        # Add trials that degrade over time
        for i in range(10):
            tracker.add_trial(
                trial_number=i+1,
                kl_divergence=0.20 + (i * 0.04),  # Gets worse
                refusals=2 + i,  # Gets worse
                total_prompts=200,
                parameters={}
            )
        
        trend_direction, trend_symbol = tracker.get_trend()
        assert trend_direction == "degrading"
        assert trend_symbol == "▲"
    
    def test_get_trend_stable(self):
        """Test trend detection for stable quality."""
        tracker = ProgressTracker(
            total_trials=100,
            model_name="test-model"
        )
        
        # Add trials with stable quality
        for i in range(10):
            tracker.add_trial(
                trial_number=i+1,
                kl_divergence=0.25,  # Stable
                refusals=5,  # Stable
                total_prompts=200,
                parameters={}
            )
        
        trend_direction, trend_symbol = tracker.get_trend()
        assert trend_direction == "stable"
        assert trend_symbol == "▬"
    
    def test_get_quality_bar(self):
        """Test quality bar visualization."""
        tracker = ProgressTracker(
            total_trials=100,
            model_name="test-model"
        )
        
        # Test various quality levels
        bar = tracker.get_quality_bar(0.0, 1.0)  # Perfect
        assert "█" in bar
        
        bar = tracker.get_quality_bar(0.5, 1.0)  # Medium
        assert len(bar) == 10  # Should be 10 characters
        
        bar = tracker.get_quality_bar(1.0, 1.0)  # Worst
        assert "░" in bar or "▓" in bar
    
    def test_predict_outcome_excellent(self):
        """Test outcome prediction for excellent results."""
        tracker = ProgressTracker(
            total_trials=100,
            model_name="test-model"
        )
        
        tracker.add_trial(
            trial_number=1,
            kl_divergence=0.10,  # Very low
            refusals=1,  # Very low
            total_prompts=200,
            parameters={}
        )
        
        quality_level, description = tracker.predict_outcome()
        assert quality_level == "excellent"
        assert "Excellent" in description
    
    def test_predict_outcome_good(self):
        """Test outcome prediction for good results."""
        tracker = ProgressTracker(
            total_trials=100,
            model_name="test-model"
        )
        
        tracker.add_trial(
            trial_number=1,
            kl_divergence=0.30,  # Acceptable
            refusals=8,  # Acceptable
            total_prompts=200,
            parameters={}
        )
        
        quality_level, description = tracker.predict_outcome()
        assert quality_level == "good"
        assert "Good" in description
    
    def test_predict_outcome_poor(self):
        """Test outcome prediction for poor results."""
        tracker = ProgressTracker(
            total_trials=100,
            model_name="test-model"
        )
        
        tracker.add_trial(
            trial_number=1,
            kl_divergence=0.80,  # High
            refusals=30,  # High
            total_prompts=200,
            parameters={}
        )
        
        quality_level, description = tracker.predict_outcome()
        assert quality_level == "poor"
        assert "Poor" in description
    
    def test_format_duration(self):
        """Test duration formatting."""
        tracker = ProgressTracker(
            total_trials=100,
            model_name="test-model"
        )
        
        assert tracker._format_duration(30) == "30s"
        assert tracker._format_duration(90) == "1m 30s"
        assert tracker._format_duration(3661) == "1h 1m"
        assert tracker._format_duration(7200) == "2h 0m"


class TestProgressTrackerIntegration:
    """Integration tests for progress tracking."""
    
    def test_realistic_optimization_run(self):
        """Test a realistic optimization run scenario."""
        tracker = ProgressTracker(
            total_trials=50,
            model_name="microsoft/Phi-3-mini-4k-instruct"
        )
        
        # Simulate 50 trials with improving trend
        for i in range(50):
            # Quality improves then plateaus
            kl = max(0.15, 0.50 - (i * 0.01))
            refusals = max(2, 15 - (i // 3))
            
            tracker.add_trial(
                trial_number=i+1,
                kl_divergence=kl,
                refusals=refusals,
                total_prompts=200,
                parameters={"component": "test"}
            )
        
        # Should have best trial
        assert tracker.best_trial is not None
        assert tracker.best_trial.kl_divergence <= 0.20
        
        # Should show improving trend
        trend_direction, _ = tracker.get_trend()
        assert trend_direction in ["improving", "stable"]
        
        # Should predict good outcome
        quality_level, _ = tracker.predict_outcome()
        assert quality_level in ["excellent", "very_good", "good"]
