import pytest
from pathlib import Path
from unittest.mock import MagicMock
from omniprompt.providers import (
    ProviderFactory, GoogleProvider, OpenAIProvider, AnthropicProvider, 
    AlibabaProvider, OpenAICompatibleProvider
)

# --- Factory Tests ---

def test_provider_factory_google():
    p = ProviderFactory.get_provider("google", "test_key")
    assert isinstance(p, GoogleProvider)
    assert p.api_key == "test_key"

def test_provider_factory_openai():
    p = ProviderFactory.get_provider("openai", "test_key")
    assert isinstance(p, OpenAIProvider)

def test_provider_factory_invalid():
    p = ProviderFactory.get_provider("invalid", "test_key")
    assert p is None

# --- Google Provider Tests ---

def test_google_generate_text(mocker, capsys):
    mock_genai_module = mocker.patch("omniprompt.providers.genai")
    mock_client_cls = mock_genai_module.Client
    mock_client_instance = mock_client_cls.return_value
    
    mock_response = MagicMock()
    mock_response.text = "Google response"
    mock_client_instance.models.generate_content.return_value = mock_response

    provider = GoogleProvider("test_key")
    provider.generate_text("gemini-pro", "hello")

    mock_client_cls.assert_called_with(api_key="test_key")
    mock_client_instance.models.generate_content.assert_called_with(
        model="gemini-pro", contents="hello"
    )
    
    captured = capsys.readouterr()
    assert "Response from google/gemini-pro" in captured.out
    assert "Google response" in captured.out

def test_google_list_models(mocker, capsys):
    mock_genai_module = mocker.patch("omniprompt.providers.genai")
    mock_client_instance = mock_genai_module.Client.return_value
    
    model1 = MagicMock()
    model1.name = "models/gemini-pro"
    model1.supported_actions = ["generateContent"]
    
    model2 = MagicMock()
    model2.name = "models/embedding-001"
    model2.supported_actions = ["embedContent"] 

    mock_client_instance.models.list.return_value = [model1, model2]

    provider = GoogleProvider("test_key")
    provider.list_models()

    captured = capsys.readouterr()
    assert "models/gemini-pro" in captured.out
    assert "models/embedding-001" not in captured.out

def test_google_generate_image_mock(mocker, capsys):
    mock_run = mocker.patch("omniprompt.providers.run_with_dynamic_captions")
    def side_effect(console, action, *args, **kwargs):
        return action()
    mock_run.side_effect = side_effect

    mock_genai_module = mocker.patch("omniprompt.providers.genai")
    mock_client_instance = mock_genai_module.Client.return_value

    mock_response = MagicMock()
    mock_part = MagicMock()
    mock_part.inline_data.mime_type = "image/png"
    mock_part.inline_data.data = b"fake_image_data"
    mock_candidate = MagicMock()
    mock_candidate.content.parts = [mock_part]
    mock_response.candidates = [mock_candidate]
    mock_client_instance.models.generate_content.return_value = mock_response

    mock_save_image = mocker.patch("omniprompt.providers.save_image")
    mock_save_image.return_value = Path("generated_images/test.png")
    mocker.patch("omniprompt.providers.Console")

    provider = GoogleProvider("test_key")
    provider.generate_image("imagen-3", "draw a cat")

    mock_save_image.assert_called_once()

# --- OpenAI Provider Tests ---

def test_openai_generate_text(mocker, capsys):
    mock_openai_class = mocker.patch("omniprompt.providers.OpenAI")
    mock_client = mock_openai_class.return_value
    
    mock_completion = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "OpenAI response"
    mock_completion.choices = [MagicMock(message=mock_message)]
    
    mock_client.chat.completions.create.return_value = mock_completion
    mock_console_cls = mocker.patch("omniprompt.providers.Console")
    mock_console_instance = mock_console_cls.return_value

    provider = OpenAIProvider("test_key")
    provider.generate_text("gpt-4", "hello")

    mock_client.chat.completions.create.assert_called_once()
    
    calls = mock_console_instance.print.call_args_list
    assert any("Response from openai/gpt-4" in str(call) for call in calls)

# --- Anthropic Provider Tests ---

def test_anthropic_list_models(capsys):
    provider = AnthropicProvider("test_key")
    provider.list_models()
    captured = capsys.readouterr()
    assert "claude-3-opus-20240229" in captured.out
