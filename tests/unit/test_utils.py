"""
Unit tests for Blasphemer utility functions.
"""

import pytest
import torch
from heretic.utils import format_duration, batchify, empty_cache

# Disable CLI parsing for all tests in this module
@pytest.fixture(autouse=True)
def disable_cli_parsing(monkeypatch):
    """Disable command-line argument parsing in Settings for tests."""
    # Mock sys.argv to prevent CLI parsing conflicts
    import sys
    monkeypatch.setattr(sys, 'argv', ['pytest'])


class TestFormatDuration:
    """Test the format_duration utility function."""
    
    def test_seconds_only(self):
        """Test formatting durations under 60 seconds."""
        assert format_duration(30) == "30s"
        assert format_duration(59) == "59s"
    
    def test_minutes_and_seconds(self):
        """Test formatting durations with minutes."""
        assert format_duration(60) == "1m 0s"
        assert format_duration(90) == "1m 30s"
        assert format_duration(125) == "2m 5s"
    
    def test_hours_minutes_seconds(self):
        """Test formatting durations with hours."""
        # When hours are present, seconds are omitted
        assert format_duration(3600) == "1h 0m"
        assert format_duration(3665) == "1h 1m"  # 1h 1m 5s -> 1h 1m (seconds omitted)
        assert format_duration(7325) == "2h 2m"  # 2h 2m 5s -> 2h 2m (seconds omitted)
    
    def test_zero_duration(self):
        """Test formatting zero duration."""
        assert format_duration(0) == "0s"
    
    def test_float_duration(self):
        """Test formatting floating point durations."""
        result = format_duration(90.7)
        assert "1m 30s" in result or "1m 31s" in result


class TestBatchify:
    """Test the batchify utility function."""
    
    def test_exact_batches(self):
        """Test batchifying with exact batch size."""
        items = [1, 2, 3, 4, 5, 6]
        batches = list(batchify(items, 2))
        assert len(batches) == 3
        assert batches[0] == [1, 2]
        assert batches[1] == [3, 4]
        assert batches[2] == [5, 6]
    
    def test_uneven_batches(self):
        """Test batchifying with remainder."""
        items = [1, 2, 3, 4, 5]
        batches = list(batchify(items, 2))
        assert len(batches) == 3
        assert batches[0] == [1, 2]
        assert batches[1] == [3, 4]
        assert batches[2] == [5]
    
    def test_single_batch(self):
        """Test batchifying into a single batch."""
        items = [1, 2, 3]
        batches = list(batchify(items, 10))
        assert len(batches) == 1
        assert batches[0] == [1, 2, 3]
    
    def test_empty_list(self):
        """Test batchifying an empty list."""
        items = []
        batches = list(batchify(items, 2))
        assert len(batches) == 0
    
    def test_batch_size_one(self):
        """Test batchifying with batch size of 1."""
        items = [1, 2, 3]
        batches = list(batchify(items, 1))
        assert len(batches) == 3
        assert batches[0] == [1]
        assert batches[1] == [2]
        assert batches[2] == [3]


class TestEmptyCache:
    """Test the empty_cache utility function."""
    
    def test_empty_cache_cuda(self):
        """Test cache emptying with CUDA."""
        if torch.cuda.is_available():
            empty_cache()
            # Should not raise an error
            assert True
        else:
            pytest.skip("CUDA not available")
    
    def test_empty_cache_mps(self):
        """Test cache emptying with MPS (Apple Silicon)."""
        if torch.backends.mps.is_available():
            empty_cache()
            # Should not raise an error
            assert True
        else:
            pytest.skip("MPS not available")
    
    def test_empty_cache_no_gpu(self):
        """Test cache emptying without GPU (should not crash)."""
        # This should work even without GPU
        empty_cache()
        assert True


class TestTrialFormatting:
    """Test trial parameter formatting functions."""
    
    def test_get_readme_intro(self):
        """Test README introduction generation."""
        from heretic.utils import get_readme_intro
        from heretic.config import Settings
        from unittest.mock import Mock
        
        # Create mock objects
        settings = Settings(model="test-org/test-model")
        trial = Mock()
        trial.params = {"param1": 0.5}
        trial.user_attrs = {
            "direction_index": 1,
            "parameters": {
                # Parameters are now stored as dicts (not dataclasses)
                # This reflects the JSON serialization fix
                "attn.o_proj": {
                    "max_weight": 1.0,
                    "max_weight_position": 16.0,
                    "min_weight": 0.5,
                    "min_weight_distance": 8.0
                }
            },
            "kl_divergence": 0.5,
            "refusals": 2
        }
        
        readme = get_readme_intro(settings, trial, base_refusals=10, bad_prompts=["prompt1", "prompt2"])
        
        assert "test-org/test-model" in readme
        assert "Blasphemer" in readme
        assert "github.com/sunkencity999/blasphemer" in readme
        assert "Heretic" in readme  # Should credit original
        assert "fork" in readme.lower()
    
    def test_readme_has_proper_links(self):
        """Test that generated README has proper markdown links."""
        from heretic.utils import get_readme_intro
        from heretic.config import Settings
        from unittest.mock import Mock
        
        # Create mock objects
        settings = Settings(model="meta-llama/Llama-3-8B")
        trial = Mock()
        trial.params = {}
        trial.user_attrs = {
            "direction_index": 0,
            "parameters": {
                # Parameters stored as dicts after JSON serialization fix
                "mlp.down_proj": {
                    "max_weight": 1.1,
                    "max_weight_position": 18.0,
                    "min_weight": 0.3,
                    "min_weight_distance": 9.0
                }
            },
            "kl_divergence": 0.3,
            "refusals": 1
        }
        
        readme = get_readme_intro(settings, trial, base_refusals=5, bad_prompts=["test"])
        
        # Check for markdown link format
        assert "[Blasphemer]" in readme
        assert "(https://github.com/sunkencity999/blasphemer)" in readme
        assert "[Heretic]" in readme
        assert "(https://github.com/p-e-w/heretic)" in readme
