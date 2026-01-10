"""
LLM Provider implementations for OmniPrompt.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Any
import base64

# Import provider-specific libraries
from google import genai
from openai import OpenAI
from anthropic import Anthropic
import dashscope

# Import Rich for UI
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

# Local imports
from .utils import save_image, download_image, run_with_dynamic_captions

class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(self, api_key: str, provider_name: str, base_url: Optional[str] = None):
        """
        Initialize the provider.

        Args:
            api_key: The API key for the provider.
            provider_name: The display name of the provider.
            base_url: Optional base URL for OpenAI-compatible APIs.
        """
        self.api_key = api_key
        self.provider_name = provider_name
        self.base_url = base_url

    @abstractmethod
    def generate_text(self, model: str, prompt: str) -> None:
        """Generates text from the model and prints it to the console."""
        pass

    def generate_image(self, model: str, prompt: str) -> None:
        """Generates an image from the model and saves it."""
        print(f"Error: Image generation not supported for provider '{self.provider_name}'.")

    @abstractmethod
    def list_models(self) -> None:
        """Lists available models for this provider."""
        pass

class GoogleProvider(LLMProvider):
    """Google Gemini / Imagen provider implementation."""

    def __init__(self, api_key: str):
        super().__init__(api_key, 'google')

    def generate_text(self, model: str, prompt: str) -> None:
        console = Console()
        try:
            client = genai.Client(api_key=self.api_key)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Thinking..."),
                transient=True,
                console=console
            ) as progress:
                progress.add_task("generate", total=None)
                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )
            
            console.print(f"[bold blue]--- Response from google/{model} ---[/bold blue]")
            console.print(Markdown(response.text))
            console.print("\n")
        except Exception as e:
            console.print(f"[bold red]--- Error from google/{model} ---[/bold red]")
            console.print(f"An error occurred: {e}\n")

    def generate_image(self, model: str, prompt: str) -> None:
        console = Console()
        try:
            client = genai.Client(api_key=self.api_key)
            
            def _call_google():
                return client.models.generate_content(
                    model=model,
                    contents=prompt
                )

            response = run_with_dynamic_captions(console, _call_google)
            
            if response.candidates:
                for candidate in response.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if part.inline_data:
                                if part.inline_data.mime_type.startswith('image/'):
                                    image_data = part.inline_data.data
                                    if isinstance(image_data, str):
                                         image_data = base64.b64decode(image_data)
                                    
                                    filepath = save_image(image_data, 'google', prompt)
                                    console.print(f"[bold green]Image generated successfully![/bold green]")
                                    console.print(f"Saved to: [bold]{filepath}[/bold]")
                                    return
            
            console.print(f"--- Response from google/{model} ---")
            console.print("Raw response received. Could not automatically extract image.")
            try:
                 if response.text:
                    console.print(response.text)
            except Exception:
                 console.print("[It seems the response contains non-text data that the CLI could not extract]")

        except Exception as e:
            console.print(f"[bold red]--- Error from google/{model} ---[/bold red]")
            console.print(f"An error occurred: {e}\n")

    def list_models(self) -> None:
        console = Console()
        try:
            client = genai.Client(api_key=self.api_key)
            console.print("[bold blue]--- Available models for google ---[/bold blue]")
            for m in client.models.list():
                 if m.supported_actions and 'generateContent' in m.supported_actions:
                    console.print(f" - {m.name}")
        except Exception as e:
            console.print(f"[bold red]--- Error listing google models ---[/bold red]")
            console.print(f"An error occurred: {e}\n")

class OpenAICompatibleProvider(LLMProvider):
    """Generic provider for OpenAI-compatible APIs (Groq, Moonshot, etc.)."""

    def generate_text(self, model: str, prompt: str) -> None:
        console = Console()
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold green]Thinking..."),
                transient=True,
                console=console
            ) as progress:
                progress.add_task("generate", total=None)
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                )
            
            response_text = chat_completion.choices[0].message.content
            console.print(f"[bold green]--- Response from {self.provider_name}/{model} ---[/bold green]")
            console.print(Markdown(response_text))
            console.print("\n")
        except Exception as e:
            console.print(f"[bold red]--- Error from {self.provider_name}/{model} ---[/bold red]")
            console.print(f"An error occurred: {e}\n")

    def list_models(self) -> None:
        console = Console()
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            models = client.models.list()
            console.print(f"[bold green]--- Available models for {self.provider_name} ---[/bold green]")
            for model in sorted(models.data, key=lambda m: m.id):
                console.print(f" - {model.id}")
        except Exception as e:
            console.print(f"[bold red]--- Error listing {self.provider_name} models ---[/bold red]")
            console.print(f"An error occurred: {e}\n")

class OpenAIProvider(OpenAICompatibleProvider):
    """Official OpenAI provider, adding image support."""

    def __init__(self, api_key: str):
        super().__init__(api_key, 'openai')

    def generate_image(self, model: str, prompt: str) -> None:
        console = Console()
        try:
            client = OpenAI(api_key=self.api_key)
            
            def _call_openai():
                return client.images.generate(
                    model=model,
                    prompt=prompt,
                    n=1,
                    size="1024x1024"
                )

            response = run_with_dynamic_captions(console, _call_openai)
            image_url = response.data[0].url
            
            with Progress(SpinnerColumn(), TextColumn("[bold green]Downloading image..."), transient=True, console=console) as dl_progress:
                dl_progress.add_task("Download", total=None)
                image_data = download_image(image_url)
            
            filepath = save_image(image_data, 'openai', prompt, extension="png")
            console.print(f"[bold green]Image generated successfully![/bold green]")
            console.print(f"Saved to: [bold]{filepath}[/bold]")
                
        except Exception as e:
            console.print(f"[bold red]--- Error from openai/{model} ---[/bold red]")
            console.print(f"An error occurred: {e}\n")

class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation."""

    def __init__(self, api_key: str):
        super().__init__(api_key, 'anthropic')

    def generate_text(self, model: str, prompt: str) -> None:
        console = Console()
        try:
            client = Anthropic(api_key=self.api_key)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold magenta]Thinking..."),
                transient=True,
                console=console
            ) as progress:
                progress.add_task("generate", total=None)
                message = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
            
            response_text = message.content[0].text
            console.print(f"[bold magenta]--- Response from anthropic/{model} ---[/bold magenta]")
            console.print(Markdown(response_text))
            console.print("\n")
        except Exception as e:
            console.print(f"[bold red]--- Error from anthropic/{model} ---[/bold red]")
            console.print(f"An error occurred: {e}\n")

    def list_models(self) -> None:
        console = Console()
        console.print("[bold magenta]--- Available models for anthropic ---[/bold magenta]")
        console.print("[italic]Note: Anthropic API does not support listing models. This is a curated list.[/italic]")
        console.print(" - claude-3-opus-20240229")
        console.print(" - claude-3-sonnet-20240229")
        console.print(" - claude-3-haiku-20240307")

class AlibabaProvider(LLMProvider):
    """Alibaba Qwen (Dashscope) provider implementation."""

    def __init__(self, api_key: str):
        super().__init__(api_key, 'alibaba')

    def generate_text(self, model: str, prompt: str) -> None:
        console = Console()
        try:
            dashscope.api_key = self.api_key
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold yellow]Thinking..."),
                transient=True,
                console=console
            ) as progress:
                progress.add_task("generate", total=None)
                response = dashscope.Generation.call(
                    model=model,
                    prompt=prompt
                )
            
            response_text = response.output.text
            console.print(f"[bold yellow]--- Response from alibaba/{model} ---[/bold yellow]")
            console.print(Markdown(response_text))
            console.print("\n")
        except Exception as e:
            console.print(f"[bold red]--- Error from alibaba/{model} ---[/bold red]")
            console.print(f"An error occurred: {e}\n")

    def list_models(self) -> None:
        console = Console()
        try:
            dashscope.api_key = self.api_key
            models = dashscope.Generation.list_models()
            console.print("[bold yellow]--- Available models for alibaba ---[/bold yellow]")
            for model in sorted([m.id for m in models if m.id and 'qwen' in m.id]):
                console.print(f" - {model}")
        except Exception as e:
            console.print(f"[bold red]--- Error listing alibaba models ---[/bold red]")
            console.print(f"An error occurred: {e}\n")

class ProviderFactory:
    """Factory class to create provider instances."""

    @staticmethod
    def get_provider(provider_name: str, api_key: str) -> Optional[LLMProvider]:
        """
        Returns a provider instance based on the provider name.

        Args:
            provider_name: The name of the provider (e.g., 'google', 'openai').
            api_key: The API key for the provider.

        Returns:
            An LLMProvider instance or None if the provider is not supported.
        """
        if provider_name == 'google':
            return GoogleProvider(api_key)
        elif provider_name == 'openai':
            return OpenAIProvider(api_key)
        elif provider_name == 'anthropic':
            return AnthropicProvider(api_key)
        elif provider_name == 'alibaba':
            return AlibabaProvider(api_key)
        elif provider_name == 'groq':
            return OpenAICompatibleProvider(api_key, 'groq', 'https://api.groq.com/openai/v1')
        elif provider_name == 'moonshot':
            return OpenAICompatibleProvider(api_key, 'moonshot', 'https://api.moonshot.cn/v1')
        else:
            return None
