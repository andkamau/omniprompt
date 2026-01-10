"""
OmniPrompt

A command-line utility for quickly testing and interacting with various
large language model (LLM) APIs from different providers.
"""

import argparse
import yaml
import os
import time
import random
import base64
import requests
import concurrent.futures
from datetime import datetime
from pathlib import Path

# Import provider-specific libraries
from google import genai
from google.genai import types
from openai import OpenAI
from anthropic import Anthropic
from groq import Groq
import dashscope

# Import Rich for UI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.text import Text
from rich.markdown import Markdown

# --- Constants ---
GENERATED_IMAGES_DIR = Path("generated_images")

FUN_CAPTIONS = [
    "Convincing the pixels to cooperate...",
    "Mixing red, green, and blue in a cauldron...",
    "Consulting the oracle of aesthetics...",
    "Teaching the AI art history in 5 seconds...",
    "Dreaming in electric sheep...",
    "Summoning the muse from the cloud...",
    "Applying virtual paint to digital canvas...",
    "Negotiating with the GPU...",
    "Connecting the dots... all million of them...",
    "Polishing the pixels...",
]

# --- Configuration ---

def load_config(config_path='config.yaml'):
    """
    Loads the non-sensitive configuration from a YAML file.
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found. Please ensure it exists.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None

def get_api_key(provider, config):
    """
    Retrieves the API key for a given provider by reading an environment variable.

    The name of the environment variable is defined in the config.yaml file.

    Args:
        provider (str): The name of the provider (e.g., 'google').
        config (dict): The configuration dictionary.

    Returns:
        tuple(str or None, str or None): A tuple containing the API key and the
                                         name of the environment variable.
    """
    provider_config = config.get(provider, {})
    env_var_name = provider_config.get('api_key_env')

    if not env_var_name:
        return None, None

    api_key = os.getenv(env_var_name)
    return api_key, env_var_name

# --- Helper Functions ---

def get_fun_caption():
    """Returns a random fun caption."""
    return random.choice(FUN_CAPTIONS)

def save_image(data, provider, prompt, extension="png"):
    """
    Saves image data (bytes) to the generated_images directory.
    """
    GENERATED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create a filename based on timestamp and prompt (sanitized)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_prompt = "".join(x for x in prompt if x.isalnum() or x in " -_")[:50].strip().replace(" ", "_")
    filename = f"{provider}_{timestamp}_{sanitized_prompt}.{extension}"
    filepath = GENERATED_IMAGES_DIR / filename
    
    with open(filepath, "wb") as f:
        f.write(data)
    
    return filepath

def download_image(url):
    """Downloads an image from a URL."""
    response = requests.get(url)
    response.raise_for_status()
    return response.content

def run_with_dynamic_captions(console, action, *args, **kwargs):
    """Runs an action in a separate thread while updating captions."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        transient=True,
        console=console
    ) as progress:
        task = progress.add_task(get_fun_caption(), total=None)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(action, *args, **kwargs)
            
            while not future.done():
                time.sleep(2.0) # Change caption every 2 seconds
                if not future.done():
                    progress.update(task, description=get_fun_caption())
            
            return future.result()

# --- Provider Classes ---

from abc import ABC, abstractmethod

class LLMProvider(ABC):
    def __init__(self, api_key, provider_name, base_url=None):
        self.api_key = api_key
        self.provider_name = provider_name
        self.base_url = base_url

    @abstractmethod
    def generate_text(self, model, prompt):
        pass

    def generate_image(self, model, prompt):
        print(f"Error: Image generation not supported for provider '{self.provider_name}'.")

    def list_models(self):
        print(f"Error: Listing models not supported for provider '{self.provider_name}'.")

class GoogleProvider(LLMProvider):
    def __init__(self, api_key):
        super().__init__(api_key, 'google')

    def generate_text(self, model, prompt):
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

    def generate_image(self, model, prompt):
        console = Console()
        try:
            client = genai.Client(api_key=self.api_key)
            
            def _call_google():
                return client.models.generate_content(
                    model=model,
                    contents=prompt
                )

            response = run_with_dynamic_captions(console, _call_google)
            
            # Check response for images
            # The new SDK response structure: response.candidates[0].content.parts[0].inline_data
            if response.candidates:
                for candidate in response.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                             # Check for inline_data
                            if part.inline_data:
                                if part.inline_data.mime_type.startswith('image/'):
                                    # Decode base64 if it's a string, or use bytes if it's bytes
                                    image_data = part.inline_data.data
                                    if isinstance(image_data, str):
                                         image_data = base64.b64decode(image_data)
                                    
                                    filepath = save_image(image_data, 'google', prompt)
                                    console.print(f"[bold green]Image generated successfully![/bold green]")
                                    console.print(f"Saved to: [bold]{filepath}[/bold]")
                                    return
            
            # Fallback/Placeholder
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

    def list_models(self):
        console = Console()
        try:
            client = genai.Client(api_key=self.api_key)
            console.print("[bold blue]--- Available models for google ---[/bold blue]")
            # list_models returns an iterator of Model objects
            for m in client.models.list():
                 # We filter for models that support content generation
                 if 'generateContent' in m.supported_generation_methods:
                    console.print(f" - {m.name}")
        except Exception as e:
            console.print(f"[bold red]--- Error listing google models ---[/bold red]")
            console.print(f"An error occurred: {e}\n")

class OpenAICompatibleProvider(LLMProvider):
    def generate_text(self, model, prompt):
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

    def list_models(self):
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
    def __init__(self, api_key):
        super().__init__(api_key, 'openai')

    def generate_image(self, model, prompt):
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
            
            # We can also wrap the download with a simpler spinner or just print
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
    def __init__(self, api_key):
        super().__init__(api_key, 'anthropic')

    def generate_text(self, model, prompt):
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

    def list_models(self):
        console = Console()
        console.print("[bold magenta]--- Available models for anthropic ---[/bold magenta]")
        console.print("[italic]Note: Anthropic API does not support listing models. This is a curated list.[/italic]")
        console.print(" - claude-3-opus-20240229")
        console.print(" - claude-3-sonnet-20240229")
        console.print(" - claude-3-haiku-20240307")

class AlibabaProvider(LLMProvider):
    def __init__(self, api_key):
        super().__init__(api_key, 'alibaba')

    def generate_text(self, model, prompt):
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

    def list_models(self):
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
    @staticmethod
    def get_provider(provider_name, api_key):
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

# --- Main Execution ---

def setup_arg_parser():
    """Sets up the command-line argument parser."""
    parser = argparse.ArgumentParser(description="A CLI for interacting with multiple LLM APIs.")
    parser.add_argument("-P", "--provider", help="The API provider (e.g., google, openai).")
    parser.add_argument("-m", "--model", help="The specific model to use.")
    parser.add_argument("-p", "--prompt", help="The text prompt to send to the model.")
    parser.add_argument("-i", "--generate-image", help="The prompt for image generation.")
    parser.add_argument("-a", "--all-providers", action="store_true", help="Send a prompt to all configured providers.")
    parser.add_argument("-l", "--list-models", dest="list_provider", help="List available models for a given provider.")
    return parser

def main():
    """The main function to run the OmniPrompt CLI."""
    script_dir = Path(__file__).parent
    config_path = script_dir / 'config.yaml'
    config = load_config(config_path)
    if config is None:
        return

    parser = setup_arg_parser()
    args = parser.parse_args()

    # --- Handle Model Listing ---
    if args.list_provider:
        provider_name = args.list_provider
        api_key, env_var_name = get_api_key(provider_name, config)

        if not env_var_name:
             print(f"Error: Provider '{provider_name}' not found or 'api_key_env' not set in config.yaml.")
             return

        # Special case: Anthropic listing doesn't require an API key
        if not api_key and provider_name != 'anthropic':
            print(f"Error: API key for '{provider_name}' not found. Please set the '{env_var_name}' environment variable.")
            return

        provider = ProviderFactory.get_provider(provider_name, api_key)
        if provider:
            provider.list_models()
        else:
            print(f"Error: Provider '{provider_name}' is not supported for listing models.")
        return

    # --- Handle All Providers Query ---
    if args.all_providers:
        # This feature requires iterating over all config keys and creating providers
        # Not fully refactored in this step as requested by "Refactor Provider Architecture" for main flow first.
        print("The --all-providers feature is not yet fully refactored to use the new Provider classes.")
        return

    # --- Handle Image Generation ---
    if args.generate_image:
        # Default to OpenAI if no provider specified, or use specified provider
        provider_name = args.provider if args.provider else 'openai'
        # Default models
        model = args.model
        if not model:
            if provider_name == 'openai':
                model = 'dall-e-3'
            elif provider_name == 'google':
                model = 'gemini-3-pro-image-preview'
        
        api_key, env_var_name = get_api_key(provider_name, config)
        
        if not env_var_name:
             print(f"Error: Provider '{provider_name}' not found or 'api_key_env' not set in config.yaml.")
             return

        if not api_key:
            print(f"Error: API key for '{provider_name}' not found. Please set the '{env_var_name}' environment variable.")
            return

        provider = ProviderFactory.get_provider(provider_name, api_key)
        if provider:
            provider.generate_image(model, args.generate_image)
        else:
            print(f"Error: Image generation not supported for provider '{provider_name}'.")
        return

    # --- Handle Standard Query ---
    if args.provider and args.model and args.prompt:
        provider_name = args.provider
        api_key, env_var_name = get_api_key(provider_name, config)

        if not env_var_name:
             print(f"Error: Provider '{provider_name}' not found or 'api_key_env' not set in config.yaml.")
             return

        if not api_key:
            print(f"Error: API key for '{provider_name}' not found. Please set the '{env_var_name}' environment variable.")
            return

        provider = ProviderFactory.get_provider(provider_name, api_key)
        if provider:
            provider.generate_text(args.model, args.prompt)
        else:
            print(f"Error: Provider '{provider_name}' is not supported.")
        return

    parser.print_help()

if __name__ == "__main__":
    main()
