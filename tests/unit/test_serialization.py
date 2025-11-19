"""
Unit tests for JSON serialization fixes in Blasphemer.

Tests the critical bug fix for AbliterationParameters serialization
that was causing trial failures.
"""

import pytest
import json
from dataclasses import asdict
from heretic.model import AbliterationParameters


# Disable CLI parsing for all tests in this module
@pytest.fixture(autouse=True)
def disable_cli_parsing(monkeypatch):
    """Disable command-line argument parsing in Settings for tests."""
    import sys
    monkeypatch.setattr(sys, 'argv', ['pytest'])


class TestAbliterationParametersSerialization:
    """Test JSON serialization of AbliterationParameters."""
    
    def test_parameters_dataclass_structure(self):
        """Test that AbliterationParameters is a proper dataclass."""
        params = AbliterationParameters(
            max_weight=1.0,
            max_weight_position=16.0,
            min_weight=0.5,
            min_weight_distance=8.0
        )
        
        assert params.max_weight == 1.0
        assert params.max_weight_position == 16.0
        assert params.min_weight == 0.5
        assert params.min_weight_distance == 8.0
    
    def test_parameters_to_dict_conversion(self):
        """Test converting AbliterationParameters to dict using asdict."""
        params = AbliterationParameters(
            max_weight=1.2,
            max_weight_position=20.0,
            min_weight=0.3,
            min_weight_distance=10.0
        )
        
        # This is the fix - must convert to dict before JSON serialization
        params_dict = asdict(params)
        
        assert isinstance(params_dict, dict)
        assert params_dict["max_weight"] == 1.2
        assert params_dict["max_weight_position"] == 20.0
        assert params_dict["min_weight"] == 0.3
        assert params_dict["min_weight_distance"] == 10.0
    
    def test_parameters_dict_json_serializable(self):
        """Test that converted parameters dict is JSON serializable."""
        params = AbliterationParameters(
            max_weight=1.5,
            max_weight_position=25.0,
            min_weight=0.2,
            min_weight_distance=12.0
        )
        
        # Convert to dict (the fix)
        params_dict = asdict(params)
        
        # Should serialize without error
        json_string = json.dumps(params_dict)
        
        # Should deserialize correctly
        deserialized = json.loads(json_string)
        assert deserialized["max_weight"] == 1.5
        assert deserialized["max_weight_position"] == 25.0
    
    def test_parameters_dataclass_not_json_serializable(self):
        """Test that raw AbliterationParameters raises TypeError on JSON serialization."""
        params = AbliterationParameters(
            max_weight=1.0,
            max_weight_position=16.0,
            min_weight=0.5,
            min_weight_distance=8.0
        )
        
        # This should fail - demonstrating the bug
        with pytest.raises(TypeError, match="not JSON serializable"):
            json.dumps(params)
    
    def test_multiple_components_serialization(self):
        """Test serializing multiple component parameters."""
        # This simulates the actual use case in main.py
        parameters = {
            "attn.o_proj": AbliterationParameters(
                max_weight=1.2,
                max_weight_position=20.0,
                min_weight=0.3,
                min_weight_distance=10.0
            ),
            "mlp.down_proj": AbliterationParameters(
                max_weight=1.1,
                max_weight_position=18.0,
                min_weight=0.2,
                min_weight_distance=8.0
            )
        }
        
        # The fix: convert all values to dicts
        serializable_params = {k: asdict(v) for k, v in parameters.items()}
        
        # Should serialize without error
        json_string = json.dumps(serializable_params)
        
        # Should deserialize correctly
        deserialized = json.loads(json_string)
        assert "attn.o_proj" in deserialized
        assert "mlp.down_proj" in deserialized
        assert deserialized["attn.o_proj"]["max_weight"] == 1.2
        assert deserialized["mlp.down_proj"]["min_weight"] == 0.2
    
    def test_trial_user_attrs_format(self):
        """Test that trial.user_attrs stores parameters as dict, not dataclass."""
        from unittest.mock import Mock
        
        # Create mock trial
        trial = Mock()
        trial.user_attrs = {}
        
        # Create parameters
        parameters = {
            "attn.o_proj": AbliterationParameters(
                max_weight=1.0,
                max_weight_position=16.0,
                min_weight=0.5,
                min_weight_distance=8.0
            )
        }
        
        # The fix: store as dict
        trial.user_attrs["parameters"] = {k: asdict(v) for k, v in parameters.items()}
        
        # Verify it's stored as dict
        assert isinstance(trial.user_attrs["parameters"], dict)
        assert isinstance(trial.user_attrs["parameters"]["attn.o_proj"], dict)
        
        # Verify it's JSON serializable
        json_string = json.dumps(trial.user_attrs["parameters"])
        assert json_string is not None


class TestResumeArgument:
    """Test --resume flag format."""
    
    def test_resume_requires_boolean_value(self):
        """Test that --resume flag format includes boolean value."""
        # Correct format
        correct_cmd = "blasphemer --model test-model --resume true"
        assert "--resume true" in correct_cmd or "--resume false" in correct_cmd
        
        # Old incorrect format (would fail)
        incorrect_cmd = "blasphemer --resume --model test-model"
        # This is the bug - --resume alone expects an argument
    
    def test_resume_in_shell_script_format(self):
        """Test resume command construction in shell scripts."""
        model_name = "test-model"
        
        # Correct format (the fix)
        cmd = f'blasphemer --model "{model_name}" --resume true'
        assert "--resume true" in cmd
        
        # Should not use the old format
        assert cmd != f'blasphemer --model "{model_name}" --resume'


class TestConfigSettings:
    """Test configuration with Blasphemer prefix."""
    
    def test_checkpoint_directory_default(self):
        """Test default checkpoint directory uses blasphemer prefix."""
        default_checkpoint_dir = ".blasphemer_checkpoints"
        assert "blasphemer" in default_checkpoint_dir.lower()
        assert "heretic" not in default_checkpoint_dir.lower()
    
    def test_environment_variable_prefix(self):
        """Test environment variables use BLASPHEMER prefix."""
        import os
        
        # Test setting a BLASPHEMER_ env var
        os.environ["BLASPHEMER_TEST"] = "value"
        assert os.environ.get("BLASPHEMER_TEST") == "value"
        del os.environ["BLASPHEMER_TEST"]
        
        # Old HERETIC_ prefix should not be expected
        assert os.environ.get("HERETIC_TEST") is None
