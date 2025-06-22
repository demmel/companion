# Agent Project - Modular AI Assistant

A modular Python-based AI agent system built with pluggable configurations, tools, and LLM backends. The system supports different agent types (roleplay, coding, general) through a clean configuration system.

## Features

### Core System
- **Modular Architecture**: Configuration-driven agent types with specialized prompts and tools
- **Configurable Context Management**: Smart context window tracking with auto-summarization
- **Advanced Tool System**: Pydantic-based tools with automatic schema generation
- **Multiple Agent Types**: Roleplay, coding, and general-purpose configurations
- **Rich CLI Interface**: Beautiful command-line interface with real-time context display

### Roleplay Agent
- **Immersive Character Embodiment**: Full character consistency with personality, mood, and memory tracking
- **Advanced State Management**: Per-character memories, relationships, and emotional states
- **Visual Flair**: Emoji-enhanced responses with mood-based styling
- **Multi-Character Support**: Switch between characters in single conversations
- **Correction System**: User can correct established facts with automatic memory updates
- **Scene Management**: Location, atmosphere, and time tracking

### Context Management
- **Real-time Tracking**: Always-visible context window usage with color-coded warnings
- **Intelligent Summarization**: Config-specific summarization strategies for different agent types
- **Auto-management**: Automatic summarization when approaching context limits
- **Configurable Windows**: Adjustable context window sizes for different models

### LLM Support
- **Ollama Integration**: Optimized local inference with smart parameter tuning
- **Model Flexibility**: Easy model switching with automatic parameter adjustment
- **Performance Tuning**: Temperature, sampling, and context settings optimized for coherence

## Quick Start

### Prerequisites

1. **Install Ollama**: 
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Pull the recommended model**:
   ```bash
   ollama pull aqualaguna/gemma-3-27b-it-abliterated-GGUF:q4_k_m
   ```

### Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   uv sync
   ```

### Basic Usage

```bash
# Run with default roleplay agent
uv run python main.py

# Check if model is available
uv run python main.py --check

# Use a different model
uv run python main.py --model llama3.1:8b

# Enable verbose output
uv run python main.py --verbose
```

## Agent Types

### Roleplay Agent (Default)
Perfect for character-driven conversations and immersive storytelling:

```
You: Please roleplay as Elena, a mysterious vampire librarian
Agent: [Uses assume_character tool automatically]

🎭 **Elena** 😏 *(mysterious - moderate)*
📍 *Ancient library filled with dusty tomes*

*I look up from an ancient manuscript, my pale fingers tracing the faded text. The candlelight flickers across my sharp features as I notice your presence.*

"Welcome to my sanctuary... Few mortals find their way here. Tell me, what knowledge do you seek in these forgotten halls?"

*I close the book with a soft thud, my dark eyes studying you with centuries of wisdom behind them.*
```

### Context Display
The CLI automatically shows context usage:
- `Context: 2048/8192 tokens (25.0%)` - Green (healthy)
- `Context: 5000/8192 tokens (61.0%)` - Yellow (moderate)
- `⚠ Approaching context limit, auto-summarizing...` - Auto-management

## Interactive Commands

- Type your message to chat with the agent
- `tools` - Show available tools  
- `reset` - Clear conversation history
- `exit` - Quit the agent

## Architecture

### Core Components

1. **Agent Core** (`src/agent/core.py`)
   - Generic agent class with no domain-specific logic
   - Handles conversation flow, tool execution, and response formatting
   - Configuration-driven prompts and tools
   - Smart context management with auto-summarization

2. **Configuration System** (`src/agent/config.py` + `src/agent/configs/`)
   - `AgentConfig` base class with prompt template and tool definitions
   - `RoleplayConfig` for character roleplay with visual flair
   - `CodingConfig` and `GeneralConfig` (placeholder implementations)
   - Configuration-specific response formatters and summarization strategies

3. **Tool System** (`src/agent/tools/`)
   - `BaseTool` abstract class with Pydantic validation
   - Tool classes with `name`, `description`, `input_schema`, and `run()` method
   - `ToolRegistry` for managing tool instances
   - Clean separation between tool logic and agent core

4. **LLM Backend** (`src/agent/llm.py`)
   - `LLMClient` for Ollama integration with configurable context windows
   - Optimized parameters for coherence and performance
   - Smart parameter tuning for different model sizes

### Roleplay Tools

- `AssumeCharacterTool` - Create/switch characters with personality and background
- `SetMoodTool` - Set character emotional state with intensity levels
- `RememberDetailTool` - Store conversation memories in categorized system
- `InternalThoughtTool` - Character internal thoughts and motivations
- `SceneSettingTool` - Set scene location, atmosphere, and time
- `CharacterActionTool` - Physical actions with reasoning
- `SwitchCharacterTool` - Multi-character scenarios
- `CorrectDetailTool` - Fix or change established story details

## Project Structure

```
agent/
├── src/agent/
│   ├── core.py              # Generic agent implementation
│   ├── config.py            # Configuration base classes
│   ├── configs/             # Agent-specific configurations
│   │   ├── roleplay.py      # Roleplay agent config
│   │   ├── coding.py        # Coding agent config (placeholder)
│   │   └── general.py       # General agent config (placeholder)
│   ├── tools/               # Tool system
│   │   ├── __init__.py      # Tool base classes and registry
│   │   └── roleplay_tools.py # Roleplay tool implementations
│   ├── character_state.py   # Character state management
│   └── llm.py              # Ollama client
├── main.py                  # CLI interface
├── pyproject.toml          # Dependencies
└── README.md               # This file
```

## Configuration Examples

### Creating Custom Agent Types

```python
from agent.config import AgentConfig
from my_tools import MyCustomTool

class MyAgentConfig(AgentConfig):
    def __init__(self):
        super().__init__(
            name="custom",
            description="Custom agent type",
            prompt_template="You are a {specialty} assistant.\n{tools_description}",
            tools=[MyCustomTool()],
            default_state={"specialty": "helpful"},
            summarization_prompt="Custom summary prompt for {conversation_text}"
        )
```

### Adding New Tools

```python
from agent.tools import BaseTool, ToolInput
from pydantic import Field

class MyToolInput(ToolInput):
    message: str = Field(description="Message to process")

class MyTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_tool"
    
    @property 
    def description(self) -> str:
        return "Processes a message"
    
    @property
    def input_schema(self):
        return MyToolInput
    
    def run(self, agent, input_data):
        return f"Processed: {input_data.message}"
```

## Advanced Features

### Character Correction System
Users can correct established facts naturally:
```
User: "Actually, Elena isn't a librarian, she's a museum curator"
Agent: [Uses correct_detail tool automatically]
       "Ah yes, of course - the museum archives. My apologies for the confusion."
```

### Multi-Character Scenarios
```
User: "Now switch to playing Marcus, Elena's vampire rival"
Agent: [Uses switch_character tool]
       🎭 **Marcus** 😠 *(annoyed - high)*
       "Elena thinks she's so clever with her precious artifacts..."
```

### Smart Context Management
- Automatic summarization preserves character details, relationships, and plot
- Roleplay-specific summaries focus on narrative continuity
- Configurable context windows for different model capabilities

## Dependencies

- **Core**: `click`, `rich`, `pydantic`, `requests`, `ollama`
- **Package Manager**: `uv` for fast dependency management
- **Python**: 3.8+ required

## Performance & Model Recommendations

### Recommended Models
- **Primary**: `aqualaguna/gemma-3-27b-it-abliterated-GGUF:q4_k_m` (16GB, good balance)
- **Fallback**: `llama3.1:8b` (4.9GB, faster but less coherent)
- **High-end**: `llama3.3:70b` (42GB, best quality, requires significant RAM)

### Performance Notes
- 27B model: ~15-25 tokens/sec with 8192 context window
- Optimized for long conversations with automatic context management
- Smart parameter tuning reduces hallucination and improves coherence

## Troubleshooting

1. **Model not found**: Run `ollama pull <model-name>` to download
2. **Ollama connection error**: Make sure Ollama is running (`ollama serve`)
3. **Context issues**: Monitor the real-time context display for automatic management
4. **Character consistency**: Use the correction tools if the agent makes mistakes
5. **Memory usage**: Use smaller models for systems with limited RAM

## Current Status

### Working Features
✅ Modular configuration system  
✅ Roleplay agent with full character management  
✅ Tool call parsing and execution  
✅ Visual flair for roleplay responses  
✅ Multi-character scenarios  
✅ Context window management with auto-summarization  
✅ User correction system  
✅ Configurable summarization strategies  
✅ Real-time context tracking  

### Future Enhancements
- Implement coding and general tool sets
- Add conversation persistence
- Web interface for easier interaction
- Additional LLM backend options
- Enhanced character relationship tracking