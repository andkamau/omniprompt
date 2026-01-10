"""
OmniPrompt CLI

A command-line utility for quickly testing and interacting with various
large language model (LLM) APIs from different providers.
"""

import argparse
import yaml
import os
from pathlib import Path
from typing import Optional, Dict, Tuple, Any

# Local imports
from .providers import ProviderFactory
from .utils import save_image, download_image, run_with_dynamic_captions

# --- Constants ---

DEFAULT_CONFIG = {
    'google': {
        'api_key_env': 'GOOGLE_API_KEY',
        'api_key_url': 'https://aistudio.google.com/app/apikey'
    },
    'openai': {
        'api_key_env': 'OPENAI_API_KEY',
        'api_key_url': 'https://platform.openai.com/api-keys'
    },
    'anthropic': {
        'api_key_env': 'ANTHROPIC_API_KEY',
        'api_key_url': 'https://console.anthropic.com/settings/keys'
    },
    'groq': {
        'api_key_env': 'GROQ_API_KEY',
        'api_key_url': 'https://console.groq.com/keys'
    },
    'moonshot': {
        'api_key_env': 'MOONSHOT_API_KEY',
        'api_key_url': 'https://platform.moonshot.cn/console/api-keys'
    },
    'alibaba': {
        'api_key_env': 'ALIBABA_API_KEY',
        'api_key_url': 'https://dashscope.console.aliyun.com/apiKey'
    }
}

# --- Configuration ---

def load_config(config_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Loads the non-sensitive configuration from a YAML file.
    Searches in standard paths or returns DEFAULT_CONFIG if not found.

    Args:
        config_path: Optional specific path to a config file.

    Returns:
        The configuration dictionary or None if a file exists but is invalid.
    """
    paths_to_check = []
    if config_path:
        paths_to_check.append(Path(config_path))
    
    paths_to_check.extend([
        Path.home() / ".config" / "omniprompt" / "config.yaml",
        Path.home() / ".omniprompt.yaml",
        Path("config.yaml")
    ])

    for path in paths_to_check:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(f"Error parsing YAML file '{path}': {e}")
                return None
            except Exception as e:
                print(f"Error reading file '{path}': {e}")
                return None
    
    # Fallback to hardcoded defaults if no config file is found
    return DEFAULT_CONFIG

def get_api_key(provider: str, config: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Retrieves the API key for a given provider by reading an environment variable.

    Args:
        provider: The name of the provider (e.g., 'google').
        config: The configuration dictionary.

    Returns:
        A tuple containing (API key, environment variable name).
    """
    provider_config = config.get(provider, {})
    env_var_name = provider_config.get('api_key_env')

    if not env_var_name:
        return None, None

    api_key = os.getenv(env_var_name)
    return api_key, env_var_name

# --- Main Execution ---

def setup_arg_parser() -> argparse.ArgumentParser:
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
    config = load_config()
    if config is None:
        return

    parser = setup_arg_parser()
    args = parser.parse_args()

    # --- Handle Model Listing ---
    if args.list_provider:
        provider_name = args.list_provider
        api_key, env_var_name = get_api_key(provider_name, config)

        if not env_var_name:
             print(f"Error: Provider '{provider_name}' not found or 'api_key_env' not set in config.")
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
        print("The --all-providers feature is not yet fully implemented.")
        return

    # --- Handle Image Generation ---
    if args.generate_image:
        provider_name = args.provider if args.provider else 'openai'
        model = args.model
        if not model:
            if provider_name == 'openai':
                model = 'dall-e-3'
            elif provider_name == 'google':
                model = 'gemini-3-pro-image-preview'
        
        api_key, env_var_name = get_api_key(provider_name, config)
        
        if not env_var_name:
             print(f"Error: Provider '{provider_name}' not found or 'api_key_env' not set in config.")
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
             print(f"Error: Provider '{provider_name}' not found or 'api_key_env' not set in config.")
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