# Agent Chat Client

A React-based web client for the Agent system with real-time streaming conversations and advanced roleplay features.

## Setup

### Prerequisites
- Node.js (18+)

### Build

```bash
npm install
npm run build
```

The built files in `dist/` are served by the agent server. See the top-level [README](../README.md) for full setup.

### Development (Optional)

For hot reload during development:

```bash
npm run dev
```

This starts a dev server at `http://localhost:5173`. The client connects directly to the agent server at port 8080 (configurable via `VITE_AGENT_HOST` and `VITE_AGENT_PORT` env vars).

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

## Hook Architecture

Custom React hooks for agent communication and state management:

| Hook | Description |
|------|-------------|
| `useAgentWebSocket` | WebSocket connection with auto-reconnect and message handling |
| `useTimeline` | Timeline state management and updates |
| `useTimelineHistory` | Paginated history loading |
| `useTriggerEvents` | Event handling for triggers |
| `useStreamBatcher` | Batches rapid streaming updates for performance |
| `useSmartScroll` | Auto-scroll with user override detection |
| `useImageUpload` | Image upload handling |
| `useAudioPlayback` | Audio player state management |
| `useActionAudio` | TTS audio fetching per action |

## Audio System

The client supports TTS audio playback for agent speech:

- **On-demand loading**: Audio fetched when user clicks play
- **Streaming support**: Audio can play while still generating
- **Action-level audio**: Each speak action can have associated audio
- **Playback controls**: Play/pause per action

## API Client

The `AgentClient` class (`src/client.ts`) provides:

- **HTTP methods**: Timeline fetching, image upload, model configuration
- **WebSocket**: Real-time message streaming with hydration protocol
- **Reconnection**: Automatic reconnection with state recovery