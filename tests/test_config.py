import os
import pytest
from omniprompt.cli import load_config, get_api_key

def test_load_config_success(mock_config_file, mock_config_data):
    config = load_config(str(mock_config_file))
    assert config == mock_config_data

def test_load_config_not_found(mocker):
    # Mock Path.exists to return False for all paths
    mocker.patch("pathlib.Path.exists", return_value=False)
    config = load_config("non_existent_file.yaml")
    assert config is None

def test_get_api_key_success(mock_config_data, monkeypatch):
    monkeypatch.setenv("TEST_GOOGLE_KEY", "secret-value")
    api_key, env_var = get_api_key("google", mock_config_data)
    assert api_key == "secret-value"
    assert env_var == "TEST_GOOGLE_KEY"

def test_get_api_key_missing_env_var_in_config(mock_config_data):
    # 'unknown' provider is not in mock_config_data
    api_key, env_var = get_api_key("unknown", mock_config_data)
    assert api_key is None
    assert env_var is None

def test_get_api_key_env_var_not_set(mock_config_data, monkeypatch):
    # Ensure the env var is not set
    monkeypatch.delenv("TEST_GOOGLE_KEY", raising=False)
    api_key, env_var = get_api_key("google", mock_config_data)
    assert api_key is None
    assert env_var == "TEST_GOOGLE_KEY"
