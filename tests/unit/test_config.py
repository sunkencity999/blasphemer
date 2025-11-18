"""
Unit tests for Blasphemer configuration module.
"""

import pytest
from pathlib import Path
from unittest.mock import patch
from heretic.config import Settings, DatasetSpecification

# Disable CLI parsing for all tests in this module
@pytest.fixture(autouse=True)
def disable_cli_parsing(monkeypatch):
    """Disable command-line argument parsing in Settings for tests."""
    # Mock sys.argv to prevent CLI parsing conflicts
    import sys
    monkeypatch.setattr(sys, 'argv', ['pytest'])


class TestSettings:
    """Test the Settings configuration class."""
    
    def test_default_settings(self):
        """Test that default settings can be instantiated."""
        # This tests basic Pydantic model creation
        settings = Settings(model="test-model")
        assert settings.model == "test-model"
        assert settings.n_trials == 200
        assert settings.n_startup_trials == 60
        assert settings.checkpoint_dir == ".blasphemer_checkpoints"
    
    def test_checkpoint_directory_default(self):
        """Test that checkpoint directory uses blasphemer naming."""
        settings = Settings(model="test-model")
        assert "blasphemer" in settings.checkpoint_dir
        assert not settings.checkpoint_dir.startswith(".")  # Should be relative path
    
    def test_resume_flag_default(self):
        """Test that resume flag defaults to False."""
        settings = Settings(model="test-model")
        assert settings.resume is False
    
    def test_environment_prefix(self):
        """Test that environment variable prefix is correct."""
        # Access model_config to verify settings
        assert Settings.model_config.get('env_prefix') == "BLASPHEMER_"
    
    def test_refusal_markers(self):
        """Test that refusal markers are properly configured."""
        settings = Settings(model="test-model")
        assert isinstance(settings.refusal_markers, list)
        assert len(settings.refusal_markers) > 0
        assert "sorry" in settings.refusal_markers
        assert "i can't" in settings.refusal_markers
    
    def test_dtypes_list(self):
        """Test dtypes list configuration."""
        settings = Settings(model="test-model")
        assert isinstance(settings.dtypes, list)
        assert len(settings.dtypes) > 0
        assert "auto" in settings.dtypes or "float16" in settings.dtypes


class TestDatasetSpecification:
    """Test the DatasetSpecification class."""
    
    def test_dataset_spec_creation(self):
        """Test creating a dataset specification."""
        spec = DatasetSpecification(
            dataset="test/dataset",
            split="train",
            column="text"
        )
        assert spec.dataset == "test/dataset"
        assert spec.split == "train"
        assert spec.column == "text"
    
    def test_dataset_spec_in_settings(self):
        """Test dataset specifications within settings."""
        settings = Settings(model="test-model")
        assert settings.good_prompts.dataset is not None
        assert settings.bad_prompts.dataset is not None
        assert settings.good_evaluation_prompts.dataset is not None


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_batch_size_validation(self):
        """Test that batch size must be positive."""
        settings = Settings(model="test-model", max_batch_size=16)
        assert settings.max_batch_size > 0
    
    def test_trials_validation(self):
        """Test that trial counts are valid."""
        settings = Settings(model="test-model")
        assert settings.n_trials > 0
        assert settings.n_startup_trials >= 0
        assert settings.n_startup_trials <= settings.n_trials
    
    def test_model_required(self):
        """Test that model parameter is required."""
        with pytest.raises((ValueError, TypeError)):
            Settings()  # Should fail without model
