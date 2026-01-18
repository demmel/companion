# Companion

An AI agent system with real-time streaming, intelligent memory management, and advanced visual generation capabilities. The agent features autonomous decision-making based on values, priorities, and emotional state.

## Project Structure

```
companion/
├── agent_server/    # Python backend (FastAPI) - Agent core, memory system, LLM integration
└── client/          # React frontend - Real-time chat interface
```

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+
- [Ollama](https://ollama.ai/) or Anthropic API key

### Setup

1. **Clone and configure**

   ```bash
   git clone <repository-url>
   cd companion
   cp agent_server/.env.example agent_server/.env
   # Edit .env with your API keys
   ```

2. **Install dependencies**

   ```bash
   # Backend
   cd agent_server
   uv sync

   # Frontend
   cd ../client
   npm install
   ```

3. **Build the client**

   ```bash
   cd client
   npm run build
   ```

4. **Start the server** (from agent_server/)

   ```bash
   cd ../agent_server
   uv run uvicorn agent.api_server:app --host 0.0.0.0 --port 8080
   ```

5. **Open** `http://localhost:8080` in your browser

## Documentation

| Document                                                       | Description                                        |
| -------------------------------------------------------------- | -------------------------------------------------- |
| [agent_server/README.md](agent_server/README.md)               | Backend architecture, API reference, memory system |
| [agent_server/CONFIGURATION.md](agent_server/CONFIGURATION.md) | Environment variables and settings                 |
| [client/README.md](client/README.md)                           | Frontend setup and development                     |

## Tech Stack

**Backend**

- FastAPI + WebSockets
- LLM: Anthropic Claude / Ollama (local)
- Sentence Transformers for embeddings
- SDXL for image generation
- Chatterbox TTS for voice synthesis

**Frontend**

- React 19 + TypeScript
- Panda CSS
- Real-time streaming via WebSocket
