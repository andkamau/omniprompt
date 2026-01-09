import pytest
from pathlib import Path
from unittest.mock import MagicMock
from omniprompt import (
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
    mock_genai = mocker.patch("omniprompt.genai")
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Google response"
    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model

    provider = GoogleProvider("test_key")
    provider.generate_text("gemini-pro", "hello")

    mock_genai.configure.assert_called_with(api_key="test_key")
    mock_genai.GenerativeModel.assert_called_with("gemini-pro")
    mock_model.generate_content.assert_called_with("hello")
    
    captured = capsys.readouterr()
    assert "--- Response from google/gemini-pro ---" in captured.out
    assert "Google response" in captured.out

def test_google_list_models(mocker, capsys):
    mock_genai = mocker.patch("omniprompt.genai")
    model1 = MagicMock()
    model1.name = "models/gemini-pro"
    model1.supported_generation_methods = ["generateContent"]
    
    model2 = MagicMock()
    model2.name = "models/embedding-001"
    model2.supported_generation_methods = ["embedContent"] # Should be filtered out

    mock_genai.list_models.return_value = [model1, model2]

    provider = GoogleProvider("test_key")
    provider.list_models()

    captured = capsys.readouterr()
    assert "models/gemini-pro" in captured.out
    assert "models/embedding-001" not in captured.out

# --- OpenAI Provider Tests ---

def test_openai_generate_text(mocker, capsys):
    mock_openai_class = mocker.patch("omniprompt.OpenAI")
    mock_client = mock_openai_class.return_value
    
    mock_completion = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "OpenAI response"
    mock_completion.choices = [MagicMock(message=mock_message)]
    
    mock_client.chat.completions.create.return_value = mock_completion

    provider = OpenAIProvider("test_key")
    provider.generate_text("gpt-4", "hello")

    mock_client.chat.completions.create.assert_called_once()
    captured = capsys.readouterr()
    assert "--- Response from openai/gpt-4 ---" in captured.out
    assert "OpenAI response" in captured.out

# --- Anthropic Provider Tests ---

def test_anthropic_list_models(capsys):
    provider = AnthropicProvider("test_key")
    provider.list_models()
    captured = capsys.readouterr()
    assert "claude-3-opus-20240229" in captured.out

# --- Image Generation Mock Test (Simple) ---

def test_google_generate_image_mock(mocker, capsys):
    # This tests the flow, not the complex threading/Rich UI extensively
    # We mock run_with_dynamic_captions to just execute the function
    
    mock_run = mocker.patch("omniprompt.run_with_dynamic_captions")
    # Define a side effect that just calls the passed function (args[1] is action)
    def side_effect(console, action, *args, **kwargs):
        return action()
    mock_run.side_effect = side_effect

    mock_genai = mocker.patch("omniprompt.genai")
    mock_model = MagicMock()
    mock_response = MagicMock()
    
    # Mocking parts for image
    mock_part = MagicMock()
    mock_part.mime_type = "image/png"
    mock_part.blob = b"fake_image_data"
    # Ensure inline_data is False-y or not present to hit the blob path
    mock_part.inline_data = None
    
    mock_response.parts = [mock_part]
    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model

    mock_save_image = mocker.patch("omniprompt.save_image")
    mock_save_image.return_value = Path("generated_images/test.png")
    
    # Mock Console to avoid actual printing clutter
    mocker.patch("omniprompt.Console")

    provider = GoogleProvider("test_key")
    provider.generate_image("imagen-3", "draw a cat")

    mock_save_image.assert_called_once()
