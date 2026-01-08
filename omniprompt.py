"""
OmniPrompt

A command-line utility for quickly testing and interacting with various
large language model (LLM) APIs from different providers.
"""

import argparse
import yaml
import os
from pathlib import Path

# Import provider-specific libraries
import google.generativeai as genai
from openai import OpenAI
from anthropic import Anthropic
from groq import Groq
import dashscope

# --- Configuration ---

def load_config(config_path='config.yaml'):
    """
    Loads the API configuration from a YAML file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        dict: A dictionary containing the API configurations.
              Returns an empty dictionary if the file is not found.
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return {}

# --- Provider API Functions ---

def query_google(api_key, model, prompt):
    """
    Sends a prompt to the Google Gemini API.

    Args:
        api_key (str): The API key for the Google AI service.
        model (str): The model to use (e.g., 'gemini-1.5-pro-latest').
        prompt (str): The user's prompt.
    """
    try:
        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(prompt)
        print(f"--- Response from google/{model} ---\n{response.text}\n")
    except Exception as e:
        print(f"--- Error from google/{model} ---\nAn error occurred: {e}\n")

def generate_google_image(api_key, model, prompt):
    """
    Generates an image using the Google Imagen model.

    Args:
        api_key (str): The API key for the Google AI service.
        model (str): The model to use (e.g., 'imagen-3').
        prompt (str): The user's prompt for image generation.
    """
    try:
        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(prompt)
        # As of the latest APIs, image generation might be part of the standard
        # generative model flow or require a specific model that outputs image data.
        # This is a placeholder for how one might handle an image response.
        # For now, we assume the response contains a descriptive text or URL.
        # In a real implementation, you would handle image bytes or URLs.
        print(f"--- Image generation task from google/{model} ---")
        print("NOTE: This is a placeholder for actual image file handling.")
        print(f"Response received for prompt: '{prompt}'\n")

    except Exception as e:
        print(f"--- Error from google/{model} ---\nAn error occurred: {e}\n")


def query_openai_compatible(api_key, model, prompt, provider_name, base_url=None):
    """
    Sends a prompt to an OpenAI-compatible API.
    This includes OpenAI, Groq, and Moonshot.

    Args:
        api_key (str): The API key for the service.
        model (str): The model to use.
        prompt (str): The user's prompt.
        provider_name (str): The name of the provider (e.g., 'openai', 'groq').
        base_url (str, optional): The base URL for the API endpoint.
    """
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
    """
    Sends a prompt to the Anthropic Claude API.

    Args:
        api_key (str): The API key for the Anthropic service.
        model (str): The model to use (e.g., 'claude-3-opus-20240229').
        prompt (str): The user's prompt.
    """
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
    """
    Sends a prompt to the Alibaba Qwen API (Dashscope).

    Args:
        api_key (str): The API key for the Alibaba Dashscope service.
        model (str): The model to use (e.g., 'qwen-turbo').
        prompt (str): The user's prompt.
    """
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


# --- Main Execution ---

def setup_arg_parser():
    """
    Sets up the command-line argument parser.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(description="A CLI for interacting with multiple LLM APIs.")
    parser.add_argument("-P", "--provider", help="The API provider (e.g., google, openai).")
    parser.add_argument("-m", "--model", help="The specific model to use.")
    parser.add_argument("-p", "--prompt", help="The text prompt to send to the model.")
    parser.add_argument("-i", "--generate-image", help="The prompt for image generation.")
    parser.add_argument("-a", "--all-providers", action="store_true", help="Send a prompt to all configured providers.")
    return parser

def main():
    """
    The main function to run the OmniPrompt CLI.
    """
    # Find config file relative to the script's location
    script_dir = Path(__file__).parent
    config_path = script_dir / 'config.yaml'
    config = load_config(config_path)

    if not config:
        return

    parser = setup_arg_parser()
    args = parser.parse_args()

    # Default models for --all-providers mode
    default_models = {
        "google": "gemini-1.5-flash-latest",
        "openai": "gpt-3.5-turbo",
        "anthropic": "claude-3-haiku-20240307",
        "groq": "llama3-8b-8192",
        "moonshot": "moonshot-v1-8k",
        "alibaba": "qwen-turbo"
    }

    if args.all_providers:
        if not args.prompt:
            print("Error: --all-providers requires a --prompt (-p) to be set.")
            return
        print(f"Querying all configured providers with prompt: '{args.prompt}'\n")
        for provider, settings in config.items():
            api_key = settings.get('api_key', 'YOUR_API_KEY_HERE')
            if api_key and api_key != 'YOUR_API_KEY_HERE':
                model = default_models.get(provider)
                if not model:
                    print(f"--- No default model specified for {provider} ---")
                    continue

                if provider == 'google':
                    query_google(api_key, model, args.prompt)
                elif provider == 'openai':
                    query_openai_compatible(api_key, model, args.prompt, 'openai')
                elif provider == 'anthropic':
                    query_anthropic(api_key, model, args.prompt)
                elif provider == 'groq':
                    query_openai_compatible(api_key, model, args.prompt, 'groq', 'https://api.groq.com/openai/v1')
                elif provider == 'moonshot':
                    query_openai_compatible(api_key, model, args.prompt, 'moonshot', 'https://api.moonshot.cn/v1')
                elif provider == 'alibaba':
                    query_alibaba(api_key, model, args.prompt)
        return

    if args.generate_image:
        if args.provider != 'google':
            print("Error: Image generation is currently only supported for the 'google' provider.")
            return
        api_key = config.get('google', {}).get('api_key')
        if not api_key or api_key == 'YOUR_API_KEY_HERE':
            print("Error: Google API key not configured in config.yaml.")
            return
        # Using a specific model for image generation
        image_model = 'imagen-3' # Placeholder, actual model may vary
        generate_google_image(api_key, image_model, args.generate_image)
        return

    if not args.provider or not args.model or not args.prompt:
        parser.print_help()
        return

    provider_config = config.get(args.provider, {})
    api_key = provider_config.get('api_key')

    if not api_key or api_key == 'YOUR_API_KEY_HERE':
        print(f"Error: API key for '{args.provider}' is not configured in config.yaml.")
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


if __name__ == "__main__":
    main()
