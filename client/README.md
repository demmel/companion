# Agent Chat Client

A React-based web client for the Agent system with real-time streaming conversations and advanced roleplay features.

## Development Setup

### Prerequisites
- Node.js (18+)
- npm or yarn
- Running Agent server (Python backend)

### Development Workflow

1. **Start the Agent server** (in separate terminal):
   ```bash
   cd /path/to/agent
   python -m src.agent.api_server
   ```

2. **Start the client development server**:
   ```bash
   cd client
   npm install
   npm run dev
   ```

3. **Open browser** to `http://localhost:5173`

### Configuration

The client automatically configures itself based on the environment:

- **Development**: Connects to `localhost:8000/api/` (configurable via `.env.development`)
- **Production**: Connects to the same server that serves the client at `/api/`

All API endpoints are prefixed with `/api/` to avoid conflicts with client routing.

#### Environment Variables (Development Only)

Create `.env.development` to customize the agent server connection:

```env
VITE_AGENT_HOST=localhost
VITE_AGENT_PORT=8000
```

## Production Deployment

1. **Build the client**:
   ```bash
   npm run build
   ```

2. **Serve from Agent server**: The built files in `dist/` should be served by the Python API server

3. **Single endpoint**: Users access the chat interface and API from the same URL

## Architecture

- **React 19** with TypeScript for modern development
- **Panda CSS** for performant styling with build-time generation
- **Presenter pattern** for extensible conversation types (roleplay, coding, general)
- **Real-time streaming** via WebSocket with intelligent batching
- **Comprehensive testing** with Vitest and React Testing Library

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm test` - Run tests
- `npm test:ui` - Run tests with UI
- `npm run lint` - Lint code

## Features

### Chat Interface
- Real-time streaming conversations
- Smart scrolling that follows new messages
- Message grouping for natural conversation flow
- Connection status indicators

### Roleplay Features
- Character creation and switching
- Mood tracking with emoji indicators
- Internal thoughts and character actions
- Scene setting and atmosphere
- Multi-character conversations

### Developer Features
- Hot reload during development
- Comprehensive TypeScript coverage
- Component testing with React Testing Library
- Performance optimizations with React.memo and batching