# OmniPrompt

OmniPrompt is a unified command-line interface (CLI) for interacting with various Large Language Model (LLM) providers. It simplifies testing prompts, listing models, and generating images across different AI services like Google Gemini, OpenAI, Anthropic Claude, and others.

## 🚀 Features

*   **Unified Interface:** One command (`omniprompt`) to rule them all. Switch providers with a simple flag.
*   **Multi-Provider Support:** First-class support for:
    *   **Google** (Gemini & Imagen)
    *   **OpenAI** (GPT & DALL-E)
    *   **Anthropic** (Claude)
    *   **Groq**
    *   **Alibaba** (Qwen)
    *   **Moonshot** (Kimi)
*   **Image Generation:** Generate images directly from the CLI (supports Google Imagen & DALL-E 3).
*   **Model Discovery:** Easily list available models for each provider.
*   **Rich UI:** Beautiful Markdown rendering and progress spinners for a modern CLI experience.
*   **Secure:** API keys are managed via environment variables, keeping your secrets safe.

## 📦 Installation

Install OmniPrompt directly from PyPI:

```bash
pip install omniprompt
```

## 🔑 Configuration

OmniPrompt relies on environment variables for authentication. You can set these in your shell session, add them to your `~/.bashrc` or `~/.zshrc`, or use a `.env` file manager.

**Required Environment Variables:**

| Provider | Environment Variable | Get Key |
| :--- | :--- | :--- |
| **Google** | `GOOGLE_API_KEY` | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| **OpenAI** | `OPENAI_API_KEY` | [OpenAI Platform](https://platform.openai.com/api-keys) |
| **Anthropic** | `ANTHROPIC_API_KEY` | [Anthropic Console](https://console.anthropic.com/settings/keys) |
| **Groq** | `GROQ_API_KEY` | [Groq Console](https://console.groq.com/keys) |
| **Alibaba** | `ALIBABA_API_KEY` | [DashScope](https://dashscope.console.aliyun.com/apiKey) |
| **Moonshot** | `MOONSHOT_API_KEY` | [Moonshot Platform](https://platform.moonshot.cn/console/api-keys) |

*Note: You only need to set variables for the providers you intend to use.*

## 💡 Usage

### Basic Text Generation
Send a prompt to a specific provider and model.

```bash
# Ask Claude a question
omniprompt --provider anthropic --model claude-3-haiku-20240307 --prompt "Describe a futuristic Nairobi where technology and nature coexist in harmony."

# Ask GPT-4o
omniprompt -P openai -m gpt-4o -p "Write a poem celebrating the resilience and beauty of the Serengeti."
```

### Image Generation
Generate an image using DALL-E 3 (default for OpenAI) or Google Imagen.

```bash
# Generate with OpenAI (DALL-E 3)
omniprompt --provider openai --generate-image "A masai warrior wearing vibranium armor in a high-tech Wakanda-style city"

# Generate with Google (Imagen 3)
omniprompt -P google -i "A majestic digital art portrait of an African queen with galaxy hair"
```
*Images are saved to the `generated_images/` directory in your current path.*

### List Available Models
See which models are available for a provider.

```bash
omniprompt --list-models google
```

### Full Argument List

| Argument | Short | Description |
| :--- | :--- | :--- |
| `--provider` | `-P` | The API provider (e.g., `google`, `openai`, `anthropic`). |
| `--model` | `-m` | The specific model ID to use. |
| `--prompt` | `-p` | The text prompt to send to the model. |
| `--generate-image` | `-i` | Prompt for image generation. |
| `--list-models` | `-l` | List available models for the specified provider. |
| `--help` | `-h` | Show the help message and exit. |

## 🤝 Contributing

Contributions are welcome! Please check out our [GitHub repository](https://github.com/andkamau/omniprompt) to report issues or submit pull requests.

## 📄 License

This project is licensed under the **MIT License**.
