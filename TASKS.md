# OmniPrompt Tasks

## Phase 1: Core Functionality (Completed)
- [x] Basic CLI structure with `argparse`.
- [x] Google Gemini integration via `google-genai`.
- [x] OpenAI integration.
- [x] Anthropic integration.
- [x] Groq integration.
- [x] Alibaba (Qwen) integration.
- [x] Moonshot (Kimi) integration.
- [x] Image generation support for Google (Imagen) and OpenAI (DALL-E).
- [x] Rich UI with Markdown rendering and progress spinners.
- [x] Model discovery (listing models per provider).
- [x] Configuration management (YAML + Environment Variables).
- [x] Unit tests for utils, config, and providers.
- [x] Native text "polishing" feature with pre-canned styles and custom instructions.

## Phase 2: Refactoring & Quality (Active)
- [x] Modularize codebase (`cli.py`, `providers.py`, `utils.py`).
- [x] Standardize `manage.sh` for development tasks.
- [x] Add comprehensive docstrings to all modules and functions.
- [ ] Implement robust error handling for API timeouts and rate limits.
- [ ] Add more edge-case tests (e.g., invalid API keys, empty prompts).
- [ ] Integrate linting and type-checking into `manage.sh` (e.g., `mypy`).

## Phase 3: Release & Distribution (Ongoing)
- [x] Prepare `pyproject.toml` and `MANIFEST.in`.
- [x] Update documentation (`README.md`).
- [ ] Tag the initial release (v1.0.0).
- [ ] Publish to PyPI.
- [ ] Set up GitHub Actions for automated testing and deployment.

## User & Environment Context
- **Environment:** Local macOS (Darwin) development.
- **Tools:** Python 3, `pip`, `pytest`, `ruff`, `git`.
- **Capabilities:** User can run tests locally using `./manage.sh test`, perform linting with `./manage.sh lint`, and test the CLI application directly using `./manage.sh start`.
- **API Access:** Configured via environment variables for Google, OpenAI, Anthropic, etc.

## Active Gaps
- [x] `TASKS.md` tracked and updated.
- [x] `.gitignore` updated for `.DS_Store`.
- [ ] No dedicated security scan for secrets (mandate mentions config.yaml check).
