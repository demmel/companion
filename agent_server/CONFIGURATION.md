# Configuration Guide

This document explains how to configure the agent system using environment variables.

## Quick Start

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and fill in your values:**
   ```bash
   nano .env  # or use your preferred editor
   ```

3. **Add your API keys and configuration**

4. **Never commit `.env` to git** (it's already in `.gitignore`)

## Configuration System

The agent uses a centralized configuration system in `src/agent/config.py` that:
- Loads environment variables from `.env` file automatically
- Provides sensible defaults for optional settings
- Validates configuration at runtime

## Environment Variables

### LLM Provider Configuration

#### Ollama (Local Models)
```bash
# Default: localhost:11434
OLLAMA_HOST=localhost:11434
```

Configure the host for your local Ollama instance. This is used for running models like Mistral, Llama, etc. locally.

#### Anthropic (Claude API)
```bash
# Required for Claude models
ANTHROPIC_API_KEY=sk-ant-api03-...
```

Get your API key from [Anthropic Console](https://console.anthropic.com/).

**Required for:** Claude Sonnet 4.5, Claude Opus 4, Claude Haiku 4

### Logging

```bash
# Log level (default: INFO)
# Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=INFO
```

The logging system is automatically configured on startup. Use:
- `DEBUG` - Detailed information, typically for diagnosing problems
- `INFO` - General informational messages (default)
- `WARNING` - Warning messages, but application still works
- `ERROR` - Error messages, some functionality may be affected
- `CRITICAL` - Critical errors, application may not be able to continue

## Using Configuration in Code

### Import the config module:
```python
from agent.config import config

# Access configuration values
ollama_host = config.ollama_host()
api_key = config.anthropic_api_key()
log_level = config.log_level()

# Check if Anthropic is configured
if config.validate_anthropic_config():
    print("Anthropic is ready to use!")

# Check for missing configuration
missing = config.get_missing_config()
if missing:
    print(f"Missing config: {missing}")
```

### LLM Creation:
```python
from agent.llm import create_llm

# Automatically uses config from .env
llm = create_llm()

# Or override with explicit values
llm = create_llm(
    ollama_host="custom-host:11434",
    anthropic_api_key="sk-ant-..."
)
```

### Running the Server:

Server host and port are specified via uvicorn CLI arguments, not .env:

```bash
# Development (defaults to 127.0.0.1:8000)
uvicorn agent.api_server:app --reload

# Production
uvicorn agent.api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

## File Overview

- **`.env`** - Your actual configuration (gitignored, never commit!)
- **`.env.example`** - Template with documentation (commit this!)
- **`src/agent/config.py`** - Configuration module (loads and validates)

## Adding New Configuration

When adding new environment variables:

1. Add to `.env.example` with documentation
2. Add accessor method in `config.py`:
   ```python
   @staticmethod
   def my_new_setting() -> str:
       """Description of the setting"""
       return os.getenv("MY_NEW_SETTING", "default_value")
   ```
3. Update this documentation

## Validation

The config module includes validation helpers:

- `config.validate_anthropic_config()` - Check if Anthropic API key is valid
- `config.get_missing_config()` - Get list of missing required configuration

Add your own validation methods for new configuration as needed.

## Best Practices

1. **Never commit `.env`** - Keep secrets out of version control
2. **Keep `.env.example` updated** - Document all available options
3. **Provide sensible defaults** - Make optional settings truly optional
4. **Validate at startup** - Catch configuration errors early
5. **Use the config module** - Don't scatter `os.getenv()` throughout code

## Troubleshooting

### "Anthropic API key not configured"
- Check that `ANTHROPIC_API_KEY` is set in `.env`
- Verify it's not set to the placeholder value
- Ensure the key starts with `sk-ant-`

### "Config file not loading"
- Verify `.env` exists in the project root (`agent_server/`)
- Check file permissions (should be readable)
- Try running from the project root directory

### "Environment variable not updating"
- Restart your application after changing `.env`
- The file is loaded once at startup, not dynamically
