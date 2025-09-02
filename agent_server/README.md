# AI Agent System

A sophisticated AI agent system built with a modern action-based architecture, featuring real-time streaming, intelligent memory management, and advanced image generation capabilities.

## 🌟 Key Features

### 🧠 Advanced Agent Architecture
- **Action-Based Reasoning**: Structured action planning with think, speak, update_mood, update_appearance, and utility actions
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
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                        │
├─────────────────────────────────────────────────────────────────┤
│                      FastAPI Server                            │
├─────────────────────────────────────────────────────────────────┤
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
- **Trigger-Based Storage**: Each interaction stored as trigger + agent response
- **Embedding Generation**: Automatic semantic embeddings for all interactions
- **Similarity Search**: Cosine similarity matching for relevant memory retrieval
- **Temporal Filtering**: Time-based memory filtering with relative and absolute queries
- **Compression Pipeline**: Intelligent summarization preserving key details

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- Node.js 18+
- CUDA-compatible GPU (recommended for image generation)
- [Ollama](https://ollama.ai/) with supported models

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd agent
   ```

2. **Install Python dependencies**
   ```bash
   uv sync
   ```

3. **Install frontend dependencies**
   ```bash
   cd client
   npm install
   cd ..
   ```

4. **Set up Ollama models**
   ```bash
   # Install recommended models
   ollama pull mistral-small3.2:latest
   ollama pull mistral-nemo:latest
   ```

5. **Optional: Set up image generation**
   - Download SDXL-compatible models to `models/` directory
   - Supported formats: `.safetensors` files from Civitai or Hugging Face

### Running the System

1. **Start the backend server**
   ```bash
   uv run python -m agent.api_server
   ```
   Server runs on `http://localhost:8000`

2. **Start the frontend** (in a new terminal)
   ```bash
   cd client
   npm run dev
   ```
   Frontend runs on `http://localhost:5173`

3. **Access the application**
   Open `http://localhost:5173` in your browser

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

### LLM Models
Supported models in `src/agent/llm.py`:
- `mistral-small3.2:latest` (recommended)
- `mistral-nemo:latest`  
- `llama3.1:8b`
- Custom models via Ollama

### Memory Settings
- **Embedding Model**: `all-MiniLM-L6-v2` (configurable)
- **Context Window**: 32k tokens (model-dependent)
- **Memory Retrieval**: 5 similar memories by default
- **Compression Trigger**: Automatic based on context usage

### Image Generation
- **Model Support**: SDXL-compatible `.safetensors` files
- **Resolution**: Portrait (768x1024), Landscape (1024x768), Square (1024x1024)
- **Optimization**: Multi-chunk prompt strategy for attention control
- **Negative Prompts**: Automatic quality enhancement

## 📁 Project Structure

```
agent/
├── src/agent/                  # Core agent system
│   ├── chain_of_action/        # Action-based reasoning system
│   │   ├── actions/            # Individual action implementations  
│   │   ├── action_planner.py   # Plans action sequences
│   │   └── reasoning_loop.py   # Main processing loop
│   ├── memory/                 # Memory and retrieval system
│   │   ├── embedding_service.py
│   │   └── similarity_retrieval.py
│   ├── tools/                  # External tools (image generation)
│   ├── api_server.py           # FastAPI backend
│   ├── core.py                 # Main agent class
│   └── llm.py                  # LLM client interface
├── client/                     # React frontend
│   ├── src/components/         # UI components
│   ├── src/hooks/              # React hooks for WebSocket
│   └── src/types.ts            # TypeScript definitions
├── conversations/              # Persistent conversation storage
├── generated_images/           # Generated images
├── docs/                       # Documentation
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

### Key Development Files
- **Action Implementation**: Add new actions in `src/agent/chain_of_action/actions/`
- **Memory System**: Extend memory capabilities in `src/agent/memory/`
- **Frontend Components**: React components in `client/src/components/`
- **API Endpoints**: Extend API in `src/agent/api_server.py`

### Adding New Actions
1. Create action class in `actions/` directory
2. Implement `execute()` method with typed inputs
3. Register in `action_registry.py`
4. Add frontend support in React components

### Memory System Extension
- Embedding models configurable in `embedding_service.py`
- Retrieval strategies in `similarity_retrieval.py`
- Memory extraction logic in `memory_extraction.py`

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

## 🚧 Known Issues & Limitations

See `docs/status.md` for detailed issue tracking including:
- Image generation blocking (being addressed)
- Context usage optimization opportunities  
- Action planning context staleness
- Memory summarization improvements

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