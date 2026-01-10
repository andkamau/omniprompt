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
    mock_genai_module = mocker.patch("omniprompt.genai")
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
    
    # We can check for output, but Rich might strip styling in simple text capture or add ansi codes
    captured = capsys.readouterr()
    assert "Response from google/gemini-pro" in captured.out
    # Rich renders markdown, so checking for the text content is generally safe
    assert "Google response" in captured.out

def test_google_list_models(mocker, capsys):
    mock_genai_module = mocker.patch("omniprompt.genai")
    mock_client_instance = mock_genai_module.Client.return_value
    
    model1 = MagicMock()
    model1.name = "models/gemini-pro"
    model1.supported_generation_methods = ["generateContent"]
    
    model2 = MagicMock()
    model2.name = "models/embedding-001"
    model2.supported_generation_methods = ["embedContent"] 

    mock_client_instance.models.list.return_value = [model1, model2]

    provider = GoogleProvider("test_key")
    provider.list_models()

    captured = capsys.readouterr()
    assert "models/gemini-pro" in captured.out
    assert "models/embedding-001" not in captured.out

# ... (rest of tests)

def test_google_generate_image_mock(mocker, capsys):
    # This tests the flow, not the complex threading/Rich UI extensively
    
    mock_run = mocker.patch("omniprompt.run_with_dynamic_captions")
    # Define a side effect that just calls the passed function (args[1] is action)
    def side_effect(console, action, *args, **kwargs):
        return action()
    mock_run.side_effect = side_effect

    mock_genai_module = mocker.patch("omniprompt.genai")
    mock_client_instance = mock_genai_module.Client.return_value

    # Mock response structure for new SDK
    # response.candidates[0].content.parts[0].inline_data.data
    mock_response = MagicMock()
    
    mock_part = MagicMock()
    mock_part.inline_data.mime_type = "image/png"
    mock_part.inline_data.data = b"fake_image_data"
    
    mock_candidate = MagicMock()
    mock_candidate.content.parts = [mock_part]
    
    mock_response.candidates = [mock_candidate]
    
    mock_client_instance.models.generate_content.return_value = mock_response

    mock_save_image = mocker.patch("omniprompt.save_image")
    mock_save_image.return_value = Path("generated_images/test.png")
    
    # Mock Console to avoid actual printing clutter
    mocker.patch("omniprompt.Console")

    provider = GoogleProvider("test_key")
    provider.generate_image("imagen-3", "draw a cat")

    mock_save_image.assert_called_once()
