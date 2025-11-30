# DAG Memory Evolution Visualizer

Interactive web-based tool for stepping through memory graph evolution from action logs, with visual differentiation for memory types, confidence levels, and context status.

## Features

- **Step-by-step navigation** through memory graph evolution
- **Visual differentiation** for:
  - Memory types (colors): Commitment, Identity, Emotional, Preference, Factual, Procedural
  - Context status (outlines): In context vs. out of context
  - Confidence levels (patterns): User confirmed, strong inference, reasonable assumption, etc.
  - Edge types (colors/styles): Temporal, causal, explanatory, correction, clarification
- **Interactive controls**:
  - Previous/Next step navigation
  - Play/pause for automatic progression
  - Speed control
  - Jump to checkpoints
  - Step slider
- **Information panels**:
  - Current action details
  - Graph statistics
  - Selected memory details
  - Visual legend

## Usage

### 1. Start the Visualizer

```bash
cd agent_server
uv run python src/agent/experiments/temporal_context_dag/visualizer/run_visualizer.py
```

Optional arguments:
- `--host HOST`: Host to bind to (default: localhost)
- `--port PORT`: Port to bind to (default: 5000)
- `--debug`: Enable debug mode

### 2. Open Web Interface

Open your web browser and navigate to:
```
http://localhost:5000
```

### 3. Load Action Log

1. Click "Choose File" and select a JSON action log file
2. The visualizer will process the action log and display the step count
3. Use the navigation controls to step through the graph evolution

### 4. Navigation

- **First/Last**: Jump to first or last step
- **Previous/Next**: Step backward/forward one action
- **Play/Pause**: Auto-advance through steps (use speed slider to adjust)
- **Step Slider**: Jump to any specific step
- **Checkpoints**: Jump to major milestones in the action log

### 5. Interaction

- **Click nodes**: View detailed memory information
- **Hover over nodes/edges**: Highlight connections
- **Pan and zoom**: Use mouse to navigate the graph
- **Keyboard shortcuts**:
  - Space: Play/pause
  - Left/Right arrows: Previous/next step
  - Home/End: First/last step

## Visual Legend

### Memory Types (Node Colors)
- **Red**: Commitments - promises, rules, boundaries
- **Purple**: Identity - core self-knowledge, role, purpose
- **Pink**: Emotional - feelings, emotional states
- **Orange**: Preference - user likes/dislikes, opinions
- **Blue**: Factual - objective information, data
- **Green**: Procedural - how-to knowledge, methods

### Context Status (Node Outlines)
- **Thick solid border**: Memory is in current context
- **Thin dashed border**: Memory is out of context

### Confidence Levels (Node Patterns)
- **Solid**: User confirmed
- **Diagonal stripes**: Strong inference
- **Dots**: Reasonable assumption
- **Transparent**: Speculative
- **Red outline**: Likely error
- **Red X**: Known false

### Edge Types (Line Colors/Styles)
- **Gray solid**: Temporal relationships (follows)
- **Blue solid**: Causal relationships
- **Green solid**: Explanatory relationships
- **Red dashed**: Correction relationships (contradicted, retracted, superseded)
- **Orange dotted**: Clarification relationships

## Testing

Run the test suite to verify functionality:

```bash
uv run python src/agent/experiments/temporal_context_dag/visualizer/test_visualizer.py
```

This creates a sample action log and tests the core components.

## Architecture

### Core Components

- **`action_processor.py`**: `StepwiseGraphReconstructor` - processes action logs one step at a time
- **`graph_extractor.py`**: `GraphExtractor` - converts graph states to visualization format
- **`web_app.py`**: Flask web server with REST API
- **`static/js/graph_renderer.js`**: D3.js-based graph rendering
- **`static/js/controls.js`**: UI controls and interaction logic

### API Endpoints

- `POST /api/load_action_log`: Load action log file
- `GET /api/get_step/<step>`: Get visualization data for specific step
- `GET /api/next_step`: Advance to next step
- `GET /api/prev_step`: Go to previous step
- `GET /api/jump_to_checkpoint/<label>`: Jump to checkpoint
- `GET /api/get_legend`: Get legend data
- `GET /api/get_step_info/<step>`: Get step information

## File Structure

```
visualizer/
├── __init__.py
├── README.md
├── run_visualizer.py          # Main entry point
├── test_visualizer.py         # Test suite
├── action_processor.py        # Step-by-step action processing
├── graph_extractor.py         # Visualization data extraction
├── web_app.py                # Flask web server
├── static/
│   ├── css/
│   │   └── visualizer.css     # Styling
│   └── js/
│       ├── graph_renderer.js  # D3.js graph rendering
│       └── controls.js        # UI controls
└── templates/
    └── visualizer.html        # Main HTML template
```

## Requirements

- Python 3.8+
- Flask
- D3.js (loaded from CDN)
- Existing DAG memory system components

## Troubleshooting

### "No file provided" error
- Ensure you're selecting a valid JSON file
- Check that the file contains a valid action log format

### Graph not displaying
- Check browser console for JavaScript errors
- Ensure the action log has been loaded successfully
- Try refreshing the page

### Performance issues
- Large action logs (>1000 steps) may be slow to process
- Consider using checkpoints to jump to specific sections
- Close other browser tabs to free memory

### Port already in use
- Use `--port` to specify a different port
- Check for other running Flask applications