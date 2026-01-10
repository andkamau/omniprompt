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

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/andkamau/omniprompt.git
    cd omniprompt
    ```

2.  **Install the package:**
    You can install OmniPrompt in editable mode for development:
    ```bash
    pip install -e .
    ```
    This will install all dependencies and create the `omniprompt` command.

3.  **Configure API Keys:**
    OmniPrompt reads API keys from environment variables. It looks for a `config.yaml` file in the following locations:
    -   `~/.config/omniprompt/config.yaml`
    -   `~/.omniprompt.yaml`
    -   The current directory (`./config.yaml`)

    **Step 1: Create your config file**
    Copy the provided `config.yaml` to one of the locations above.

    **Step 2: Set the Environment Variables**
    Set the environment variables with your actual API keys as defined in your `config.yaml`.

    *Example:*
    ```bash
    export OPENAI_API_KEY="sk-..."
    export GOOGLE_API_KEY="AIza..."
    ```

## Usage

Once installed, you can use the `omniprompt` command from anywhere.

### Basic Prompt
```bash
omniprompt -P openai -m gpt-4o -p "What are the three laws of thermodynamics?"
```

### List Available Models
```bash
omniprompt -l google
```

### Image Generation
```bash
omniprompt -P google -i "A futuristic cityscape at sunset, digital art."
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