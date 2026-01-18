# AI Agent System

A sophisticated AI agent system built with a modern action-based architecture, featuring real-time streaming, intelligent memory management, and advanced image generation capabilities.

## 🌟 Key Features

### 🧠 Advanced Agent Architecture

- **Action-Based Reasoning**: Structured action planning with 12 action types including cognitive, communication, state management, and information gathering
- **Intent-Based Communication**: Intelligent separation between high-level communication intents and natural language generation
- **Trigger-Based History**: Stream-of-consciousness approach that tracks stimuli and responses rather than simple conversation turns
- **Autonomous Decision Making**: Agent makes authentic choices based on values, priorities, and emotional state

### 🔮 Intelligent Memory System

- **Semantic Memory Retrieval**: Embedding-based similarity search using sentence transformers for contextual memory recall
- **Automatic Compression**: Intelligent summarization that preserves key details while maintaining emotional continuity
- **Long-Term Memory**: Persistent memory across sessions with efficient context management
- **Memory-Augmented Responses**: Relevant past experiences automatically inform current interactions

### 🎨 Advanced Image Generation

- **SDXL Integration**: High-quality image generation with Stable Diffusion XL
- **Intelligent Prompt Optimization**: Multi-chunk strategic prompt engineering for optimal attention control
- **Civitai Model Support**: Compatible with custom models from Civitai
- **Dynamic Visual Updates**: Agent can update appearance and environment contextually

### ⚡ Real-Time Streaming

- **WebSocket Communication**: Real-time bidirectional communication
- **Streaming Events**: Live progress updates for actions, image generation, and thinking processes
- **Background Processing**: Non-blocking architecture for responsive user experience
- **Event-Driven Architecture**: Structured event system for frontend integration

### 🌐 Modern Web Interface

- **React Frontend**: Modern, responsive web interface built with React and TypeScript
- **Real-Time Updates**: Live streaming of agent thoughts, actions, and responses
- **Visual Timeline**: Interactive timeline showing agent's stream of consciousness
- **Progress Indicators**: Real-time progress for image generation and long-running tasks

## 🏗️ Architecture

### Core Components

```
┌────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                       │
├────────────────────────────────────────────────────────────────┤
│                      FastAPI Server                            │
├────────────────────────────────────────────────────────────────┤
│  Agent Core  │  Action System  │  Memory System  │  LLM Client │
├──────────────┼─────────────────┼─────────────────┼─────────────┤
│ Trigger      │ Action Planner  │ Embedding       │ Ollama      │
│ History      │ Action Registry │ Service         │ Client      │
│ State Mgmt   │ Action Executor │ Similarity      │ Streaming   │
│ Streaming    │ Base Actions    │ Retrieval       │ Generation  │
└──────────────┴─────────────────┴─────────────────┴─────────────┘
```

### Action System

The agent uses a structured action system where each action has:

- **Specific Purpose**: Think, speak, update mood/appearance, manage priorities
- **Typed Inputs**: Pydantic models with validation and clear descriptions
- **Context Awareness**: Access to full conversation history and relevant memories
- **Progress Streaming**: Real-time updates during execution

### Memory Architecture

- **DAG-Based Memory Graph**: Memories stored as nodes in a directed acyclic graph with typed edges
- **Edge Types**: Semantic relationships between memories (explains, caused, contradicts, clarifies, retracts, supersedes, corrects)
- **Trigger-Based Storage**: Each interaction stored as trigger + agent response
- **Embedding Generation**: Automatic semantic embeddings for all interactions
- **Similarity Search**: Cosine similarity matching for relevant memory retrieval
- **Temporal Filtering**: Time-based memory filtering with relative and absolute queries
- **Automatic Memory Formation**: New memories automatically connected to relevant existing memories

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- Node.js 18+
- CUDA-compatible GPU (recommended for image generation)
- [Ollama](https://ollama.ai/) with supported models

### Installation

```bash
git clone <repository-url>
cd companion/agent_server
cp .env.example .env   # Configure API keys
uv sync
```

### Running

```bash
uv run uvicorn agent.api_server:app --host 0.0.0.0 --port 8080
```

Server runs on `http://localhost:8080`. See the top-level [README](../README.md) for full setup including the client build.

## 🎯 Usage Examples

### Basic Interaction

The agent responds naturally to conversation while maintaining internal state:

```
User: "I'm feeling overwhelmed with work lately."
Agent: [thinks] How to best support them through this stressful period
Agent: [speaks] I can hear the weight in your words. Work stress can be really draining...
Agent: [updates mood] Concerned and supportive
```

### Memory Integration

The agent recalls relevant past conversations:

```
User: "Remember that project I mentioned last week?"
Agent: [retrieves memories] "project discussion, work challenges, timeline concerns"
Agent: [speaks] Yes, you were worried about the tight deadline and team coordination...
```

### Visual Updates

The agent can generate contextual images:

```
User: "I'm redecorating my living room with a cozy theme."
Agent: [thinks] How to reflect a warm, comfortable environment
Agent: [updates appearance] *Generates image of agent in cozy sweater*
Agent: [speaks] That sounds wonderful! I love creating warm, inviting spaces...
```

## 🛠️ Configuration

See [CONFIGURATION.md](CONFIGURATION.md) for detailed environment variables and settings including:

- LLM providers (Anthropic Claude, Ollama)
- Image generation (SDXL)
- Text-to-speech (Chatterbox TTS)
- Memory and embedding settings

## 📡 API Reference

### REST Endpoints

| Endpoint                                 | Method | Description                              |
| ---------------------------------------- | ------ | ---------------------------------------- |
| `/api/health`                            | GET    | Health check and system status           |
| `/api/context`                           | GET    | Token usage and context information      |
| `/api/timeline`                          | GET    | Paginated trigger history with actions   |
| `/api/reset`                             | POST   | Reset agent state and history            |
| `/api/auto-wakeup`                       | GET    | Get auto-wakeup scheduling status        |
| `/api/auto-wakeup`                       | POST   | Configure auto-wakeup scheduling         |
| `/api/supported-models`                  | GET    | List available LLM models                |
| `/api/model-config`                      | GET    | Get current model assignments per action |
| `/api/model-config`                      | POST   | Update model assignments per action      |
| `/api/upload-image`                      | POST   | Upload image (max 10MB)                  |
| `/api/regenerate-image`                  | POST   | Regenerate image with new seed           |
| `/api/audio/{trigger_id}/{action_index}` | GET    | Fetch TTS audio for action               |

### WebSocket

| Endpoint    | Description                                                   |
| ----------- | ------------------------------------------------------------- |
| `/api/chat` | Real-time bidirectional communication for streaming responses |

### Model Configuration

The system supports per-action model assignment. Each action type can use a different model:

```python
# Action types that can have individual model assignments:
# think, speak, update_mood, update_appearance, update_environment,
# add_priority, remove_priority, evaluate_priorities,
# fetch_url, search_web, wait, get_creative_inspiration
```

## 🎯 Available Actions

| Action                     | Description                                         |
| -------------------------- | --------------------------------------------------- |
| `think`                    | Process emotional reactions and analyze situation   |
| `speak`                    | Generate conversational response                    |
| `update_mood`              | Change current emotional state                      |
| `update_appearance`        | Generate visual changes (triggers image generation) |
| `update_environment`       | Change setting/environment context                  |
| `add_priority`             | Add new priority to track                           |
| `remove_priority`          | Remove/complete a priority                          |
| `evaluate_priorities`      | Holistically reevaluate all priorities              |
| `search_web`               | Search the web for information                      |
| `fetch_url`                | Fetch and analyze content from a URL                |
| `wait`                     | Wait for user response before continuing            |
| `get_creative_inspiration` | Get random words for creative stimulus              |

## 📁 Project Structure

```
agent_server/
├── src/agent/                  # Core agent system
│   ├── chain_of_action/        # Action-based reasoning system
│   │   ├── action/             # Action implementations
│   │   │   └── actions/        # Individual action classes
│   │   ├── action_planner.py   # Plans action sequences
│   │   └── reasoning_loop.py   # Main processing loop
│   ├── memory/                 # Memory and retrieval system
│   │   ├── dag/                # DAG-based memory graph
│   │   └── query_extraction.py # Memory query processing
│   ├── llm/                    # LLM provider implementations
│   │   ├── anthropic.py        # Anthropic Claude client
│   │   ├── ollama.py           # Ollama client
│   │   └── router.py           # Model routing
│   ├── tts/                    # Text-to-speech system
│   ├── api_types/              # API request/response models
│   ├── api_server.py           # FastAPI backend
│   ├── core.py                 # Main agent class
│   └── embedding_service.py    # Sentence transformer embeddings
├── conversations/              # Persistent conversation storage
├── generated_images/           # Generated images
└── tests/                      # Test files
```

## 🧪 Testing

### Run All Tests

```bash
uv run pytest
```

### Specific Test Categories

```bash
# Unit tests
uv run pytest -m unit

# Integration tests
uv run pytest -m integration

# Test specific functionality
uv run python test_think_action_contexts.py
uv run python llm_performance_test.py
```

### Performance Testing

Test LLM performance across different input/output sizes:

```bash
uv run python llm_performance_test.py
```

## 🔧 Development

### Key Directories

- `src/agent/chain_of_action/action/actions/` - Action implementations
- `src/agent/memory/dag/` - DAG memory system
- `src/agent/llm/` - LLM provider clients
- `src/agent/api_server.py` - API endpoints

### Adding New Actions

1. Create action class in `chain_of_action/action/actions/`
2. Implement `execute()` method with typed inputs
3. Register in `action_registry.py`

## 📊 Monitoring & Debugging

### Logging

- **Backend Logs**: Structured logging with performance metrics
- **LLM Call Tracking**: Automatic timing and usage statistics
- **Memory Performance**: Embedding generation and retrieval metrics
- **Action Execution**: Detailed action timing and success rates

### Performance Metrics

- Token generation speed (tokens/second)
- Memory retrieval timing
- Image generation progress
- WebSocket connection health

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

### Code Style

- Python: Black formatting, type hints required
- TypeScript: ESLint configuration in `client/`
- Tests: Pytest with good coverage

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ollama**: Local LLM inference
- **Sentence Transformers**: Semantic embeddings
- **Stable Diffusion XL**: Image generation
- **FastAPI**: Modern API framework
- **React**: Frontend framework

---

**Version**: 0.1.0  
**Python**: 3.12+  
**Node.js**: 18+  
**Last Updated**: January 2025
