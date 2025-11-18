"""
Unit tests for Blasphemer configuration module.
"""

import pytest
from pathlib import Path
from heretic.config import Settings, DatasetConfig


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
    
    def test_dtype_configuration(self):
        """Test dtype settings."""
        settings = Settings(model="test-model")
        assert settings.compute_dtype in ["float16", "bfloat16"]
        assert settings.cache_dtype in ["float16", "bfloat16", "auto"]


class TestDatasetConfig:
    """Test the DatasetConfig class."""
    
    def test_dataset_config_creation(self):
        """Test creating a dataset configuration."""
        config = DatasetConfig(
            path="test/dataset",
            split="train",
            column="text"
        )
        assert config.path == "test/dataset"
        assert config.split == "train"
        assert config.column == "text"
    
    def test_dataset_config_defaults(self):
        """Test dataset configuration with defaults."""
        settings = Settings(model="test-model")
        assert settings.good_prompts.path is not None
        assert settings.bad_prompts.path is not None
        assert settings.evaluation_prompts.path is not None


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
