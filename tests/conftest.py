import pytest
import yaml
import os
from pathlib import Path

@pytest.fixture
def mock_config_data():
    return {
        'google': {'api_key_env': 'TEST_GOOGLE_KEY'},
        'openai': {'api_key_env': 'TEST_OPENAI_KEY'},
        'anthropic': {'api_key_env': 'TEST_ANTHROPIC_KEY'}
    }

@pytest.fixture
def mock_config_file(tmp_path, mock_config_data):
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(mock_config_data, f)
    return config_path
