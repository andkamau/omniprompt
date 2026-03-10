import pytest
from unittest.mock import MagicMock, patch
from omniprompt.cli import main

@pytest.fixture
def mock_cli_args(mocker):
    # This mock will be used to simulate command line arguments
    mock_parse = mocker.patch("argparse.ArgumentParser.parse_args")
    return mock_parse

def test_polish_friendly(mock_cli_args, mocker, monkeypatch):
    # Setup mock args: omniprompt -s friendly -p "hello"
    mock_cli_args.return_value = MagicMock(
        prompt="hello",
        style="friendly",
        polish=None,
        provider=None,
        model=None,
        generate_image=None,
        list_provider=None,
        all_providers=False
    )
    
    # Mock config and provider
    mocker.patch("omniprompt.cli.load_config", return_value={'google': {'api_key_env': 'GOOGLE_API_KEY'}})
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    
    mock_provider = MagicMock()
    mocker.patch("omniprompt.cli.ProviderFactory.get_provider", return_value=mock_provider)
    
    main()
    
    # Verify that the correct prompt was sent to the provider
    expected_prompt = """Rephrase the following text in a friendly, approachable tone:

hello"""
    mock_provider.generate_text.assert_called_once_with("gemini-1.5-flash", expected_prompt)

def test_custom_polish(mock_cli_args, mocker, monkeypatch):
    # Setup mock args: omniprompt --polish "Make it funny:" -p "dry joke"
    mock_cli_args.return_value = MagicMock(
        prompt="dry joke",
        style=None,
        polish="Make it funny:",
        provider="openai",
        model="gpt-4o",
        generate_image=None,
        list_provider=None,
        all_providers=False
    )
    
    mocker.patch("omniprompt.cli.load_config", return_value={'openai': {'api_key_env': 'OPENAI_API_KEY'}})
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    
    mock_provider = MagicMock()
    mocker.patch("omniprompt.cli.ProviderFactory.get_provider", return_value=mock_provider)
    
    main()
    
    expected_prompt = """Make it funny:

dry joke"""
    mock_provider.generate_text.assert_called_once_with("gpt-4o", expected_prompt)

def test_polish_defaults(mock_cli_args, mocker, monkeypatch):
    # Setup mock args: omniprompt -p "standard prompt"
    mock_cli_args.return_value = MagicMock(
        prompt="standard prompt",
        style=None,
        polish=None,
        provider=None,
        model=None,
        generate_image=None,
        list_provider=None,
        all_providers=False
    )
    
    mocker.patch("omniprompt.cli.load_config", return_value={'google': {'api_key_env': 'GOOGLE_API_KEY'}})
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    
    mock_provider = MagicMock()
    mocker.patch("omniprompt.cli.ProviderFactory.get_provider", return_value=mock_provider)
    
    main()
    
    # Defaults should be google and gemini-1.5-flash
    mock_provider.generate_text.assert_called_once_with("gemini-1.5-flash", "standard prompt")
