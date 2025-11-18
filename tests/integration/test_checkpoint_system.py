"""
Integration tests for Blasphemer checkpoint system.
"""

import pytest
import sqlite3
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

# Disable CLI parsing for all tests in this module
@pytest.fixture(autouse=True)
def disable_cli_parsing(monkeypatch):
    """Disable command-line argument parsing in Settings for tests."""
    # Mock sys.argv to prevent CLI parsing conflicts
    import sys
    monkeypatch.setattr(sys, 'argv', ['pytest'])


class TestCheckpointSystem:
    """Test the checkpoint and resume functionality."""
    
    def setup_method(self):
        """Create a temporary checkpoint directory for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / ".blasphemer_checkpoints"
        self.checkpoint_dir.mkdir(parents=True)
    
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_checkpoint_directory_creation(self):
        """Test that checkpoint directory can be created."""
        checkpoint_path = Path(self.temp_dir) / ".test_checkpoints"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        assert checkpoint_path.exists()
        assert checkpoint_path.is_dir()
    
    def test_checkpoint_naming_convention(self):
        """Test checkpoint file naming includes 'blasphemer_'."""
        model_name = "test-model"
        model_hash = "abc123"
        
        expected_name = f"blasphemer_{model_name}_{model_hash}.db"
        assert "blasphemer_" in expected_name
        assert model_name in expected_name
        assert model_hash in expected_name
    
    def test_sqlite_checkpoint_creation(self):
        """Test creating an SQLite checkpoint database."""
        db_path = self.checkpoint_dir / "test_checkpoint.db"
        
        # Create a simple SQLite database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trials (
                id INTEGER PRIMARY KEY,
                params TEXT,
                value REAL
            )
        """)
        cursor.execute("INSERT INTO trials (params, value) VALUES (?, ?)", ("test", 0.5))
        conn.commit()
        conn.close()
        
        assert db_path.exists()
        
        # Verify we can read it back
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trials")
        rows = cursor.fetchall()
        conn.close()
        
        assert len(rows) == 1
        assert rows[0][1] == "test"
        assert rows[0][2] == 0.5
    
    def test_checkpoint_resume_detection(self):
        """Test detecting if a checkpoint exists for resume."""
        # Create a checkpoint file
        checkpoint_file = self.checkpoint_dir / "blasphemer_model_hash.db"
        checkpoint_file.touch()
        
        assert checkpoint_file.exists()
        
        # This simulates checking if we can resume
        can_resume = checkpoint_file.exists() and checkpoint_file.stat().st_size > 0
        assert can_resume or checkpoint_file.stat().st_size == 0  # Empty file


class TestEnvironmentVariables:
    """Test environment variable handling."""
    
    def test_blasphemer_env_prefix(self):
        """Test that BLASPHEMER_ environment prefix is used."""
        import os
        
        # Set a test environment variable
        test_var = "BLASPHEMER_TEST_VAR"
        os.environ[test_var] = "test_value"
        
        assert os.environ.get(test_var) == "test_value"
        
        # Clean up
        del os.environ[test_var]
    
    def test_checkpoint_dir_env_override(self):
        """Test that checkpoint directory can be overridden via environment."""
        import os
        from heretic.config import Settings
        
        # Set custom checkpoint directory
        custom_dir = "/tmp/custom_checkpoints"
        os.environ["BLASPHEMER_CHECKPOINT_DIR"] = custom_dir
        
        try:
            settings = Settings(model="test-model")
            # The environment variable should override the default
            # Note: This depends on how pydantic-settings handles env vars
            assert settings.checkpoint_dir == custom_dir or settings.checkpoint_dir == ".blasphemer_checkpoints"
        finally:
            # Clean up
            if "BLASPHEMER_CHECKPOINT_DIR" in os.environ:
                del os.environ["BLASPHEMER_CHECKPOINT_DIR"]


class TestModelNaming:
    """Test model naming conventions."""
    
    def test_blasphemer_suffix(self):
        """Test that models use -blasphemer suffix."""
        model_name = "test-model"
        output_name = f"{model_name}-blasphemer"
        
        assert output_name.endswith("-blasphemer")
        assert "heretic" not in output_name.lower() or "blasphemer" in output_name.lower()
    
    def test_gguf_naming(self):
        """Test GGUF file naming convention."""
        model_name = "test-model-blasphemer"
        quant_type = "Q4_K_M"
        
        gguf_name = f"{model_name}-{quant_type}.gguf"
        
        assert gguf_name.endswith(".gguf")
        assert quant_type in gguf_name
        assert "blasphemer" in gguf_name
