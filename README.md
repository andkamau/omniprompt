# OmniPrompt

OmniPrompt is a command-line utility for quickly testing and interacting with various large language model (LLM) APIs from different providers. "Omni" reflects the tool's ability to connect to all different AI providers, and "Prompt" is the core action.

## Features

-   Test prompts against multiple AI providers from a single interface.
-   Support for major providers: Google, OpenAI, Anthropic, Groq, and more.
-   Generate images using Google's Imagen model.
-   Run a single prompt against all configured providers simultaneously.
-   Simple configuration for managing API keys.

## Setup

1.  **Clone the repository or download the files.**

2.  **Install Dependencies:**
    Navigate to the `omniprompt` directory and install the required Python packages.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Keys:**
    Open the `config.yaml` file and replace `"YOUR_API_KEY_HERE"` with your actual API keys for each service you intend to use. The file also contains links to where you can generate the keys.
    ```yaml
    google:
      api_key: "YOUR_API_KEY_HERE"
      api_key_url: "https://aistudio.google.com/app/apikey"

    openai:
      api_key: "YOUR_API_KEY_HERE"
      api_key_url: "https://platform.openai.com/api-keys"

    # ... and so on for other providers.
    ```
    *Note: You only need to fill in the keys for the providers you wish to use.*

## Usage

All commands are run from the `omniprompt` directory using `python omniprompt.py`.

### Basic Prompt

To send a prompt to a specific provider and model, use the `-P` (`--provider`), `-m` (`--model`), and `-p` (`--prompt`) arguments.

```bash
python omniprompt.py -P openai -m gpt-4o -p "What are the three laws of thermodynamics?"
```

### Image Generation

To generate an image, use the `-P` (`--provider`) and `-i` (`--generate-image`) arguments. Currently, this is supported for the `google` provider.

```bash
python omniprompt.py -P google -i "A futuristic cityscape at sunset, digital art."
```

### Test All Providers

To send the same prompt to all configured providers, use the `-a` (`--all-providers`) flag. The tool will use a default model for each provider.

```bash
python omniprompt.py -a -p "Write a haiku about a robot learning to paint."
```

### Arguments

| Full Argument      | Short Argument | Description                                           |
| ------------------ | -------------- | ----------------------------------------------------- |
| `--provider`       | `-P`           | The API provider (e.g., `google`, `openai`).          |
| `--model`          | `-m`           | The specific model to use (e.g., `gpt-4o`).           |
| `--prompt`         | `-p`           | The text prompt to send to the model.                 |
| `--generate-image` | `-i`           | The prompt for image generation.                      |
| `--all-providers`  | `-a`           | A flag to send a prompt to all configured providers.  |
