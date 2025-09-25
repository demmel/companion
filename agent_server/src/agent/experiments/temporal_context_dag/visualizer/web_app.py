"""
Web interface for DAG memory evolution visualization.

Flask web application providing interactive step-by-step navigation through
memory graph evolution with visual differentiation and information panels.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from flask import Flask, render_template, request, jsonify, send_from_directory

from .action_processor import StepwiseGraphReconstructor
from .graph_extractor import GraphExtractor

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global instances
reconstructor = StepwiseGraphReconstructor()
extractor = GraphExtractor()

# Get the directory of this file
VISUALIZER_DIR = Path(__file__).parent


@app.route('/')
def index():
    """Main visualization interface."""
    return render_template('visualizer.html')


@app.route('/api/load_action_log', methods=['POST'])
def load_action_log():
    """Load action log file and prepare for visualization."""
    try:
        if 'file' not in request.files:
            # Try JSON data for file path (for testing)
            data = request.get_json()
            filepath = data.get('filepath') if data else None

            if filepath and os.path.exists(filepath):
                # Load the action log from file path
                reconstructor.load_action_log(filepath)
            else:
                return jsonify({"error": "No file provided"}), 400
        else:
            # Handle file upload
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400

            if not file.filename.endswith('.json'):
                return jsonify({"error": "Please select a JSON file"}), 400

            # Save uploaded file temporarily and load it
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w+b', suffix='.json', delete=False) as temp_file:
                file.save(temp_file.name)
                reconstructor.load_action_log(temp_file.name)
                # Clean up temp file after loading
                os.unlink(temp_file.name)

        # Get basic info
        step_count = reconstructor.get_step_count()
        checkpoints = reconstructor.get_checkpoints()

        checkpoint_info = [
            {
                "step_index": step_idx,
                "label": checkpoint.label,
                "description": checkpoint.description,
                "timestamp": checkpoint.timestamp.isoformat()
            }
            for step_idx, checkpoint in checkpoints
        ]

        return jsonify({
            "success": True,
            "step_count": step_count,
            "checkpoints": checkpoint_info,
            "message": f"Loaded {step_count} steps from action log"
        })

    except Exception as e:
        logger.error(f"Error loading action log: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/get_step/<int:step_index>')
def get_step(step_index: int):
    """Get visualization data for a specific step."""
    try:
        state = reconstructor.set_step(step_index)
        if not state:
            return jsonify({"error": f"Invalid step index: {step_index}"}), 404

        # Extract visualization data
        viz_data = extractor.extract_visualization_data(state)

        # Debug logging
        logger.info(f"Step {step_index}: {len(viz_data.nodes)} nodes, {len(viz_data.edges)} edges")
        if viz_data.nodes:
            logger.info(f"First node: {viz_data.nodes[0].id} - {viz_data.nodes[0].content_preview}")

        return jsonify({
            "success": True,
            "current_step": step_index,
            "total_steps": reconstructor.get_step_count(),
            "data": {
                "step_index": viz_data.step_index,
                "action_type": viz_data.action_type,
                "action_description": viz_data.action_description,
                "timestamp": viz_data.timestamp,
                "nodes": [node.__dict__ for node in viz_data.nodes],
                "edges": [edge.__dict__ for edge in viz_data.edges],
                "stats": {
                    "total_memories": viz_data.total_memories,
                    "context_memories": viz_data.context_memories,
                    "total_edges": viz_data.total_edges,
                    "context_edges": viz_data.context_edges,
                    "total_tokens": viz_data.total_tokens
                }
            }
        })

    except Exception as e:
        logger.error(f"Error getting step {step_index}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/next_step')
def next_step():
    """Advance to the next step."""
    try:
        state = reconstructor.next_step()
        if not state:
            return jsonify({
                "success": False,
                "message": "Already at last step",
                "current_step": reconstructor.get_current_step(),
                "total_steps": reconstructor.get_step_count()
            })

        viz_data = extractor.extract_visualization_data(state)

        return jsonify({
            "success": True,
            "current_step": reconstructor.get_current_step(),
            "total_steps": reconstructor.get_step_count(),
            "data": {
                "step_index": viz_data.step_index,
                "action_type": viz_data.action_type,
                "action_description": viz_data.action_description,
                "timestamp": viz_data.timestamp,
                "nodes": [node.__dict__ for node in viz_data.nodes],
                "edges": [edge.__dict__ for edge in viz_data.edges],
                "stats": {
                    "total_memories": viz_data.total_memories,
                    "context_memories": viz_data.context_memories,
                    "total_edges": viz_data.total_edges,
                    "context_edges": viz_data.context_edges,
                    "total_tokens": viz_data.total_tokens
                }
            }
        })

    except Exception as e:
        logger.error(f"Error advancing to next step: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/prev_step')
def prev_step():
    """Go back to the previous step."""
    try:
        state = reconstructor.prev_step()
        if not state:
            return jsonify({
                "success": False,
                "message": "Already at first step",
                "current_step": reconstructor.get_current_step(),
                "total_steps": reconstructor.get_step_count()
            })

        viz_data = extractor.extract_visualization_data(state)

        return jsonify({
            "success": True,
            "current_step": reconstructor.get_current_step(),
            "total_steps": reconstructor.get_step_count(),
            "data": {
                "step_index": viz_data.step_index,
                "action_type": viz_data.action_type,
                "action_description": viz_data.action_description,
                "timestamp": viz_data.timestamp,
                "nodes": [node.__dict__ for node in viz_data.nodes],
                "edges": [edge.__dict__ for edge in viz_data.edges],
                "stats": {
                    "total_memories": viz_data.total_memories,
                    "context_memories": viz_data.context_memories,
                    "total_edges": viz_data.total_edges,
                    "context_edges": viz_data.context_edges,
                    "total_tokens": viz_data.total_tokens
                }
            }
        })

    except Exception as e:
        logger.error(f"Error going to previous step: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/jump_to_checkpoint/<path:checkpoint_label>')
def jump_to_checkpoint(checkpoint_label: str):
    """Jump to a specific checkpoint."""
    try:
        # URL decode the checkpoint label
        from urllib.parse import unquote
        checkpoint_label = unquote(checkpoint_label)

        state = reconstructor.jump_to_checkpoint(checkpoint_label)
        if not state:
            return jsonify({"error": f"Checkpoint '{checkpoint_label}' not found"}), 404

        viz_data = extractor.extract_visualization_data(state)

        return jsonify({
            "success": True,
            "current_step": reconstructor.get_current_step(),
            "total_steps": reconstructor.get_step_count(),
            "data": {
                "step_index": viz_data.step_index,
                "action_type": viz_data.action_type,
                "action_description": viz_data.action_description,
                "timestamp": viz_data.timestamp,
                "nodes": [node.__dict__ for node in viz_data.nodes],
                "edges": [edge.__dict__ for edge in viz_data.edges],
                "stats": {
                    "total_memories": viz_data.total_memories,
                    "context_memories": viz_data.context_memories,
                    "total_edges": viz_data.total_edges,
                    "context_edges": viz_data.context_edges,
                    "total_tokens": viz_data.total_tokens
                }
            }
        })

    except Exception as e:
        logger.error(f"Error jumping to checkpoint {checkpoint_label}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/get_legend')
def get_legend():
    """Get legend data for the visualization."""
    try:
        legend_data = extractor.get_legend_data()
        return jsonify({
            "success": True,
            "legend": legend_data
        })
    except Exception as e:
        logger.error(f"Error getting legend data: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/get_step_info/<int:step_index>')
def get_step_info(step_index: int):
    """Get detailed information about a specific step."""
    try:
        step_info = reconstructor.get_step_info(step_index)
        if not step_info:
            return jsonify({"error": f"Invalid step index: {step_index}"}), 404

        return jsonify({
            "success": True,
            "step_info": {
                "step_index": step_info.step_index,
                "action_type": step_info.action_type,
                "action_id": step_info.action_id,
                "timestamp": step_info.timestamp.isoformat(),
                "description": step_info.description,
                "is_checkpoint": step_info.is_checkpoint,
                "checkpoint_label": step_info.checkpoint_label
            }
        })

    except Exception as e:
        logger.error(f"Error getting step info for {step_index}: {e}")
        return jsonify({"error": str(e)}), 500


# Static file serving
@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files."""
    return send_from_directory(VISUALIZER_DIR / 'static', filename)


def run_visualizer(host='localhost', port=5000, debug=True):
    """Run the visualization web application."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info(f"Starting DAG Memory Visualizer on {host}:{port}")

    # Update template folder to use absolute path
    app.template_folder = str(VISUALIZER_DIR / 'templates')
    app.static_folder = str(VISUALIZER_DIR / 'static')

    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_visualizer()