# OmniPrompt

OmniPrompt is a command-line utility for quickly testing and interacting with various large language model (LLM) APIs from different providers. "Omni" reflects the tool's ability to connect to all different AI providers, and "Prompt" is the core action.

## Features

-   Test prompts against multiple AI providers from a single interface.
-   Support for major providers: Google, OpenAI, Anthropic, Groq, and more.
-   List available models for each provider.
-   Generate images using Google's Imagen model.
-   Run a single prompt against all configured providers simultaneously.
-   Secure and discoverable API key configuration using environment variables.

## Setup

1.  **Clone the repository or download the files.**

2.  **Install Dependencies:**
    Navigate to the `omniprompt` directory and install the required Python packages into the virtual environment.
    ```bash
    # First-time setup of the virtual environment
    python3 -m venv env
    source env/bin/activate
    
    # Install or update dependencies
    pip install -r requirements.txt
    ```

3.  **Configure API Keys:**
    OmniPrompt reads API keys from environment variables. The `config.yaml` file tells the script **which environment variable to read** for each provider.

    **Step 1: Find the Environment Variable Name**
    Look inside `config.yaml` to find the `api_key_env` for the provider you want to use.
    
    *Example from `config.yaml`:*
    ```yaml
    openai:
      api_key_env: "OPENAI_API_KEY"
      api_key_url: "https://platform.openai.com/api-keys"
    ```
    This tells you the script will look for an environment variable named `OPENAI_API_KEY`.

    **Step 2: Set the Environment Variable**
    In your shell, set the environment variable with your actual API key.

    *Example:*
    ```bash
    export OPENAI_API_KEY="sk-..."
    export GOOGLE_API_KEY="AIza..."
    ```
    To make these settings permanent, add the `export` lines to your shell's startup file (e.g., `~/.zshrc`, `~/.bash_profile`, or `~/.profile`).

## Usage

All commands are run from the `omniprompt` directory using `python omniprompt.py`.

### Basic Prompt
```bash
python omniprompt.py -P openai -m gpt-4o -p "What are the three laws of thermodynamics?"
```

### List Available Models
```bash
python omniprompt.py -l google
```

### Image Generation
```bash
python omniprompt.py -P google -i "A futuristic cityscape at sunset, digital art."
```

### Test All Providers
```bash
python omniprompt.py -a -p "Write a haiku about a robot learning to paint."
```

### Arguments

| Full Argument      | Short Argument | Description                                           |
| ------------------ | -------------- | ----------------------------------------------------- |
| `--provider`       | `-P`           | The API provider (e.g., `google`, `openai`).          |
| `--model`          | `-m`           | The specific model to use (e.g., `gpt-4o`).           |
| `--prompt`         | `-p`           | The text prompt to send to the model.                 |
| `--list-models`    | `-l`           | List available models for a given provider.           |
| `--generate-image` | `-i`           | The prompt for image generation.                      |
| `--all-providers`  | `-a`           | A flag to send a prompt to all configured providers.  |