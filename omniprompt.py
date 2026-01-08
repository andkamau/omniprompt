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
import google.generativeai as genai
from openai import OpenAI
from anthropic import Anthropic
from groq import Groq
import dashscope

# Import Rich for UI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.text import Text

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

# --- Provider API Functions ---
# (No changes needed in these functions)
def query_google(api_key, model, prompt):
    """Sends a prompt to the Google Gemini API."""
    try:
        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(prompt)
        print(f"--- Response from google/{model} ---\n{response.text}\n")
    except Exception as e:
        print(f"--- Error from google/{model} ---\nAn error occurred: {e}\n")

def generate_google_image(api_key, model, prompt):
    """Generates an image using the Google Imagen model."""
    console = Console()
    try:
        genai.configure(api_key=api_key)
        
        def _call_google():
            model_instance = genai.GenerativeModel(model)
            return model_instance.generate_content(prompt)

        response = run_with_dynamic_captions(console, _call_google)
        
        # Check response for images
        if hasattr(response, 'parts'):
            for part in response.parts:
                # Check for inline_data (common in newer Gemini SDKs for images)
                if hasattr(part, 'inline_data') and part.inline_data:
                    if hasattr(part.inline_data, 'mime_type') and part.inline_data.mime_type.startswith('image/'):
                        filepath = save_image(part.inline_data.data, 'google', prompt)
                        console.print(f"[bold green]Image generated successfully![/bold green]")
                        console.print(f"Saved to: [bold]{filepath}[/bold]")
                        return

                # Check for direct mime_type/blob (older/vertex SDKs sometimes)
                elif hasattr(part, 'mime_type') and part.mime_type.startswith('image/'):
                    # It's an image!
                    filepath = save_image(part.blob, 'google', prompt)
                    console.print(f"[bold green]Image generated successfully![/bold green]")
                    console.print(f"Saved to: [bold]{filepath}[/bold]")
                    return
        
        # Fallback/Placeholder
        console.print(f"--- Response from google/{model} ---")
        console.print("Raw response received. Could not automatically extract image.")
        # Only print text if it looks like text, to avoid errors with inline_data
        try:
             if response.text:
                console.print(response.text)
        except Exception:
             console.print("[It seems the response contains non-text data that the CLI could not extract]")

    except Exception as e:
        console.print(f"[bold red]--- Error from google/{model} ---[/bold red]")
        console.print(f"An error occurred: {e}\n")

def generate_openai_image(api_key, model, prompt):
    """Generates an image using OpenAI's DALL-E."""
    console = Console()
    try:
        client = OpenAI(api_key=api_key)
        
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


def query_openai_compatible(api_key, model, prompt, provider_name, base_url=None):
    """Sends a prompt to an OpenAI-compatible API."""
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
        )
        response_text = chat_completion.choices[0].message.content
        print(f"--- Response from {provider_name}/{model} ---\n{response_text}\n")
    except Exception as e:
        print(f"--- Error from {provider_name}/{model} ---\nAn error occurred: {e}\n")

def query_anthropic(api_key, model, prompt):
    """Sends a prompt to the Anthropic Claude API."""
    try:
        client = Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = message.content[0].text
        print(f"--- Response from anthropic/{model} ---\n{response_text}\n")
    except Exception as e:
        print(f"--- Error from anthropic/{model} ---\nAn error occurred: {e}\n")

def query_alibaba(api_key, model, prompt):
    """Sends a prompt to the Alibaba Qwen API (Dashscope)."""
    try:
        dashscope.api_key = api_key
        response = dashscope.Generation.call(
            model=model,
            prompt=prompt
        )
        response_text = response.output.text
        print(f"--- Response from alibaba/{model} ---\n{response_text}\n")
    except Exception as e:
        print(f"--- Error from alibaba/{model} ---\nAn error occurred: {e}\n")

# --- Model Listing Functions ---
def list_google_models(api_key):
    """Lists available models from the Google Gemini API."""
    try:
        genai.configure(api_key=api_key)
        print("--- Available models for google ---")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
    except Exception as e:
        print(f"--- Error listing google models ---\nAn error occurred: {e}\n")

def list_openai_compatible_models(api_key, provider_name, base_url=None):
    """Lists available models from an OpenAI-compatible API."""
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        models = client.models.list()
        print(f"--- Available models for {provider_name} ---")
        for model in sorted(models.data, key=lambda m: m.id):
            print(model.id)
    except Exception as e:
        print(f"--- Error listing {provider_name} models ---\nAn error occurred: {e}\n")

def list_anthropic_models():
    """Lists curated models for Anthropic, as their API doesn't support listing."""
    print("--- Available models for anthropic ---")
    print("Note: Anthropic API does not support listing models. This is a curated list.")
    print("claude-3-opus-20240229")
    print("claude-3-sonnet-20240229")
    print("claude-3-haiku-20240307")

def list_alibaba_models(api_key):
    """Lists available models from the Alibaba Qwen API (Dashscope)."""
    try:
        dashscope.api_key = api_key
        models = dashscope.Generation.list_models()
        print("--- Available models for alibaba ---")
        for model in sorted([m.id for m in models if m.id and 'qwen' in m.id]):
            print(model)
    except Exception as e:
        print(f"--- Error listing alibaba models ---\nAn error occurred: {e}\n")

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
        provider = args.list_provider
        api_key, env_var_name = get_api_key(provider, config)

        if not env_var_name:
             print(f"Error: Provider '{provider}' not found or 'api_key_env' not set in config.yaml.")
             return

        if not api_key and provider != 'anthropic':
            print(f"Error: API key for '{provider}' not found. Please set the '{env_var_name}' environment variable.")
            return

        if provider == 'google':
            list_google_models(api_key)
        elif provider == 'openai':
            list_openai_compatible_models(api_key, 'openai')
        elif provider == 'anthropic':
            list_anthropic_models()
        elif provider == 'groq':
            list_openai_compatible_models(api_key, 'groq', 'https://api.groq.com/openai/v1')
        elif provider == 'moonshot':
            list_openai_compatible_models(api_key, 'moonshot', 'https://api.moonshot.cn/v1')
        elif provider == 'alibaba':
            list_alibaba_models(api_key)
        else:
            print(f"Error: Provider '{provider}' is not supported for listing models.")
        return

    # --- Handle All Providers Query ---
    if args.all_providers:
        # ... (This can be implemented similarly)
        print("The --all-providers feature is not yet fully refactored.")
        return

    # --- Handle Image Generation ---
    if args.generate_image:
        # Default to OpenAI if no provider specified, or use specified provider
        provider = args.provider if args.provider else 'openai'
        # Default models
        model = args.model
        if not model:
            if provider == 'openai':
                model = 'dall-e-3'
            elif provider == 'google':
                model = 'gemini-3-pro-image-preview'
        
        api_key, env_var_name = get_api_key(provider, config)
        
        if not env_var_name:
             print(f"Error: Provider '{provider}' not found or 'api_key_env' not set in config.yaml.")
             return

        if not api_key:
            print(f"Error: API key for '{provider}' not found. Please set the '{env_var_name}' environment variable.")
            return

        if provider == 'openai':
            generate_openai_image(api_key, model, args.generate_image)
        elif provider == 'google':
            generate_google_image(api_key, model, args.generate_image)
        else:
            print(f"Error: Image generation not supported for provider '{provider}'.")
        return

    # --- Handle Standard Query ---
    if args.provider and args.model and args.prompt:
        api_key, env_var_name = get_api_key(args.provider, config)

        if not env_var_name:
             print(f"Error: Provider '{args.provider}' not found or 'api_key_env' not set in config.yaml.")
             return

        if not api_key:
            print(f"Error: API key for '{args.provider}' not found. Please set the '{env_var_name}' environment variable.")
            return

        if args.provider == 'google':
            query_google(api_key, args.model, args.prompt)
        elif args.provider == 'openai':
            query_openai_compatible(api_key, args.model, args.prompt, 'openai')
        elif args.provider == 'anthropic':
            query_anthropic(api_key, args.model, args.prompt)
        elif args.provider == 'groq':
            query_openai_compatible(api_key, args.model, args.prompt, 'groq', 'https://api.groq.com/openai/v1')
        elif args.provider == 'moonshot':
            query_openai_compatible(api_key, args.model, args.prompt, 'moonshot', 'https://api.moonshot.cn/v1')
        elif args.provider == 'alibaba':
            query_alibaba(api_key, args.model, args.prompt)
        else:
            print(f"Error: Provider '{args.provider}' is not supported.")
        return

    parser.print_help()

if __name__ == "__main__":
    main()
