/**
 * Controls and interaction logic for DAG memory visualizer
 * Handles UI controls, API communication, and information panel updates
 */

class VisualizerApp {
    constructor() {
        // Use force-directed renderer for planar graph layout
        this.graphRenderer = new ForceGraphRenderer('#graph-svg');
        this.currentStep = 0;
        this.totalSteps = 0;
        this.isPlaying = false;
        this.playInterval = null;
        this.playSpeed = 1000; // milliseconds

        this.initializeEventListeners();
        this.loadLegend();
    }

    initializeEventListeners() {
        // File loading
        document.getElementById('file-input').addEventListener('change', (e) => {
            this.handleFileLoad(e);
        });

        // Navigation controls
        document.getElementById('first-step').addEventListener('click', () => {
            this.goToStep(0);
        });

        document.getElementById('prev-step').addEventListener('click', () => {
            this.previousStep();
        });

        document.getElementById('play-pause').addEventListener('click', () => {
            this.togglePlay();
        });

        document.getElementById('next-step').addEventListener('click', () => {
            this.nextStep();
        });

        document.getElementById('last-step').addEventListener('click', () => {
            this.goToStep(this.totalSteps - 1);
        });

        // Step slider
        document.getElementById('step-slider').addEventListener('input', (e) => {
            this.goToStep(parseInt(e.target.value));
        });

        // Speed control
        document.getElementById('speed-slider').addEventListener('input', (e) => {
            this.playSpeed = parseInt(e.target.value);
            document.getElementById('speed-display').textContent = (this.playSpeed / 1000).toFixed(1) + 's';

            // Restart play with new speed if currently playing
            if (this.isPlaying) {
                this.stopPlay();
                this.startPlay();
            }
        });

        // Checkpoint selection
        document.getElementById('checkpoint-select').addEventListener('change', (e) => {
            if (e.target.value) {
                this.jumpToCheckpoint(e.target.value);
            }
        });

        // Fit to screen button
        document.getElementById('fit-screen').addEventListener('click', () => {
            this.graphRenderer.fitToScreen();
        });

        // Node selection from graph
        window.addEventListener('nodeSelected', (e) => {
            this.displayNodeDetails(e.detail.node);
        });

        // Edge selection from graph
        window.addEventListener('edgeSelected', (e) => {
            this.displayEdgeDetails(e.detail.edge);
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.handleKeyboard(e);
        });
    }

    async handleFileLoad(event) {
        const file = event.target.files[0];
        if (!file) return;

        if (!file.name.endsWith('.json')) {
            this.showError('Please select a JSON file');
            return;
        }

        try {
            this.showLoading(true);
            document.getElementById('load-status').textContent = 'Loading...';

            // Create FormData for file upload
            const formData = new FormData();
            formData.append('file', file);

            // Send file to server
            const response = await fetch('/api/load_action_log', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Failed to load action log');
            }

            const result = await response.json();

            if (result.success) {
                this.totalSteps = result.step_count;
                this.currentStep = 0;

                this.updateUI();
                this.populateCheckpoints(result.checkpoints);

                document.getElementById('load-status').textContent =
                    `Loaded ${result.step_count} steps`;

                // Load first step
                await this.goToStep(0);
            } else {
                throw new Error(result.error || 'Unknown error');
            }

        } catch (error) {
            console.error('Error loading file:', error);
            this.showError('Error loading file: ' + error.message);
            document.getElementById('load-status').textContent = 'Load failed';
        } finally {
            this.showLoading(false);
        }
    }

    readFileAsText(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = (e) => reject(e);
            reader.readAsText(file);
        });
    }

    async goToStep(stepIndex) {
        if (stepIndex < 0 || stepIndex >= this.totalSteps) return;

        try {
            this.showLoading(true);

            const response = await fetch(`/api/get_step/${stepIndex}`);
            if (!response.ok) {
                throw new Error('Failed to get step data');
            }

            const result = await response.json();

            if (result.success) {
                this.currentStep = stepIndex;
                this.updateGraph(result.data);
                this.updateActionInfo(result.data);
                this.updateStats(result.data.stats);
                this.updateUI();
            } else {
                throw new Error(result.error || 'Unknown error');
            }

        } catch (error) {
            console.error('Error getting step:', error);
            this.showError('Error loading step: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    async nextStep() {
        if (this.currentStep >= this.totalSteps - 1) return;

        try {
            this.showLoading(true);

            const response = await fetch('/api/next_step');
            if (!response.ok) {
                throw new Error('Failed to get next step');
            }

            const result = await response.json();

            if (result.success) {
                this.currentStep = result.current_step;
                this.updateGraph(result.data);
                this.updateActionInfo(result.data);
                this.updateStats(result.data.stats);
                this.updateUI();
            } else {
                // Already at last step or other error
                console.log(result.message);
            }

        } catch (error) {
            console.error('Error getting next step:', error);
            this.showError('Error loading next step: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    async previousStep() {
        if (this.currentStep <= 0) return;

        try {
            this.showLoading(true);

            const response = await fetch('/api/prev_step');
            if (!response.ok) {
                throw new Error('Failed to get previous step');
            }

            const result = await response.json();

            if (result.success) {
                this.currentStep = result.current_step;
                this.updateGraph(result.data);
                this.updateActionInfo(result.data);
                this.updateStats(result.data.stats);
                this.updateUI();
            } else {
                // Already at first step or other error
                console.log(result.message);
            }

        } catch (error) {
            console.error('Error getting previous step:', error);
            this.showError('Error loading previous step: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    async jumpToCheckpoint(checkpointLabel) {
        try {
            this.showLoading(true);

            const response = await fetch(`/api/jump_to_checkpoint/${encodeURIComponent(checkpointLabel)}`);
            if (!response.ok) {
                throw new Error('Failed to jump to checkpoint');
            }

            const result = await response.json();

            if (result.success) {
                this.currentStep = result.current_step;
                this.updateGraph(result.data);
                this.updateActionInfo(result.data);
                this.updateStats(result.data.stats);
                this.updateUI();
            } else {
                throw new Error(result.error || 'Checkpoint not found');
            }

        } catch (error) {
            console.error('Error jumping to checkpoint:', error);
            this.showError('Error jumping to checkpoint: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    togglePlay() {
        if (this.isPlaying) {
            this.stopPlay();
        } else {
            this.startPlay();
        }
    }

    startPlay() {
        if (this.currentStep >= this.totalSteps - 1) return;

        this.isPlaying = true;
        document.getElementById('play-pause').textContent = '⏸️ Pause';

        this.playInterval = setInterval(async () => {
            if (this.currentStep >= this.totalSteps - 1) {
                this.stopPlay();
                return;
            }
            await this.nextStep();
        }, this.playSpeed);
    }

    stopPlay() {
        this.isPlaying = false;
        document.getElementById('play-pause').textContent = '▶️ Play';

        if (this.playInterval) {
            clearInterval(this.playInterval);
            this.playInterval = null;
        }
    }

    updateGraph(data) {
        console.log('Updating graph:', data.nodes.length, 'nodes', data.edges.length, 'edges');

        // Store data for node details display
        this.currentData = data;

        // Pass data directly to force renderer
        this.graphRenderer.updateGraph(data);

        // Only auto-fit on first load or if user explicitly requests it
        // This preserves the graph evolution view between steps
        if (this.currentStep === 0 && data.nodes.length > 0) {
            setTimeout(() => {
                this.graphRenderer.fitToScreen();
            }, 1500); // Give time for initial positioning
        }
    }

    updateActionInfo(data) {
        document.getElementById('action-type').textContent = data.action_type;
        document.getElementById('action-description').textContent = data.action_description;
        document.getElementById('action-timestamp').textContent =
            new Date(data.timestamp).toLocaleString();
    }

    updateStats(stats) {
        document.getElementById('total-memories').textContent = stats.total_memories;
        document.getElementById('context-memories').textContent = stats.context_memories;
        document.getElementById('total-edges').textContent = stats.total_edges;
        document.getElementById('context-edges').textContent = stats.context_edges;
        document.getElementById('total-tokens').textContent = stats.total_tokens;
    }

    updateUI() {
        // Update step counter and slider
        document.getElementById('current-step').textContent = this.currentStep + 1;
        document.getElementById('total-steps').textContent = this.totalSteps;
        document.getElementById('step-slider').value = this.currentStep;
        document.getElementById('step-slider').max = this.totalSteps - 1;

        // Enable/disable navigation buttons
        const hasSteps = this.totalSteps > 0;
        const isFirst = this.currentStep === 0;
        const isLast = this.currentStep === this.totalSteps - 1;

        document.getElementById('first-step').disabled = !hasSteps || isFirst;
        document.getElementById('prev-step').disabled = !hasSteps || isFirst;
        document.getElementById('play-pause').disabled = !hasSteps || isLast;
        document.getElementById('next-step').disabled = !hasSteps || isLast;
        document.getElementById('last-step').disabled = !hasSteps || isLast;
        document.getElementById('step-slider').disabled = !hasSteps;
        document.getElementById('checkpoint-select').disabled = !hasSteps;
        document.getElementById('fit-screen').disabled = !hasSteps;
    }

    populateCheckpoints(checkpoints) {
        const select = document.getElementById('checkpoint-select');

        // Clear existing options except the first
        while (select.children.length > 1) {
            select.removeChild(select.lastChild);
        }

        // Add checkpoint options
        checkpoints.forEach(checkpoint => {
            const option = document.createElement('option');
            option.value = checkpoint.label;
            option.textContent = `Step ${checkpoint.step_index + 1}: ${checkpoint.label}`;
            select.appendChild(option);
        });
    }

    displayNodeDetails(node) {
        const container = document.getElementById('selected-node-info');

        // Find outgoing edges from this node
        const outgoingEdges = this.currentData?.edges?.filter(edge => edge.source === node.id) || [];

        // Find target nodes for outgoing edges
        const targetNodes = new Map();
        this.currentData?.nodes?.forEach(n => targetNodes.set(n.id, n));

        let edgesHtml = '';
        if (outgoingEdges.length > 0) {
            edgesHtml = `
            <div style="margin-top: 1rem;">
                <strong>Outgoing Connections (${outgoingEdges.length}):</strong>
                <div style="margin-top: 0.5rem; font-size: 0.85rem;">
                    ${outgoingEdges.map(edge => {
                        const target = targetNodes.get(edge.target);
                        const targetLabel = target ?
                            `${target.memory_type.substring(0, 4).toUpperCase()}: ${target.content.substring(0, 50)}...` :
                            edge.target.substring(0, 8);
                        const contextStatus = edge.in_context ? '🔗' : '⚬';
                        const edgeStyle = edge.in_context ? 'font-weight: bold; color: #333;' : 'color: #666;';
                        const edgeId = edge.id.substring(0, 8);
                        const createdTime = new Date(edge.created_at).toLocaleTimeString();
                        return `
                        <div style="margin: 0.25rem 0; padding: 0.25rem; background: #f5f5f5; border-radius: 3px; ${edgeStyle}">
                            ${contextStatus} <strong>${edge.edge_type}</strong> → ${targetLabel}
                            <div style="font-size: 0.75rem; color: #888; margin-top: 0.25rem;">
                                ID: ${edgeId} | Created: ${createdTime}
                            </div>
                        </div>`;
                    }).join('')}
                </div>
            </div>`;
        }

        container.innerHTML = `
            <div class="info-row">
                <span class="label">ID:</span>
                <span>${node.id.substring(0, 8)}...</span>
            </div>
            <div class="info-row">
                <span class="label">Type:</span>
                <span>${node.memory_type}</span>
            </div>
            <div class="info-row">
                <span class="label">Confidence:</span>
                <span>${node.confidence_level}</span>
            </div>
            <div class="info-row">
                <span class="label">Significance:</span>
                <span>${(node.emotional_significance * 100).toFixed(1)}%</span>
            </div>
            <div class="info-row">
                <span class="label">In Context:</span>
                <span>${node.in_context ? 'Yes' : 'No'}</span>
            </div>
            ${node.tokens ? `
            <div class="info-row">
                <span class="label">Tokens:</span>
                <span>${node.tokens}</span>
            </div>
            ` : ''}
            ${edgesHtml}
            <div style="margin-top: 1rem;">
                <strong>Content:</strong>
                <p style="margin-top: 0.5rem; font-size: 0.9rem;">${node.content}</p>
            </div>
            <div style="margin-top: 1rem;">
                <strong>Evidence:</strong>
                <p style="margin-top: 0.5rem; font-size: 0.9rem; font-style: italic;">${node.evidence}</p>
            </div>
        `;
    }

    displayEdgeDetails(edge) {
        // For now, just log edge details. Could be extended to show edge info panel
        console.log('Edge selected:', edge);
    }

    async loadLegend() {
        try {
            const response = await fetch('/api/get_legend');
            if (!response.ok) return;

            const result = await response.json();
            if (result.success) {
                this.renderLegend(result.legend);
            }
        } catch (error) {
            console.error('Error loading legend:', error);
        }
    }

    renderLegend(legend) {
        const container = document.getElementById('legend-container');

        let html = '';

        // Memory types
        html += '<div class="legend-section">';
        html += '<h4>Memory Types</h4>';
        legend.memory_types.forEach(type => {
            html += `
                <div class="legend-item">
                    <div class="legend-color" style="background-color: ${type.color}"></div>
                    <span>${type.name}</span>
                </div>
            `;
        });
        html += '</div>';

        // Edge types
        html += '<div class="legend-section">';
        html += '<h4>Edge Types</h4>';
        legend.edge_types.forEach(type => {
            const className = type.style === 'dashed' ? 'dashed' : type.style === 'dotted' ? 'dotted' : '';
            html += `
                <div class="legend-item">
                    <div class="legend-line ${className}" style="background-color: ${type.color}"></div>
                    <span>${type.name}</span>
                </div>
            `;
        });
        html += '</div>';

        // Context status
        html += '<div class="legend-section">';
        html += '<h4>Context Status</h4>';
        legend.context_status.forEach(status => {
            html += `
                <div class="legend-item">
                    <div class="legend-outline ${status.outline}"></div>
                    <span>${status.name}</span>
                </div>
            `;
        });
        html += '</div>';

        container.innerHTML = html;
    }

    handleKeyboard(event) {
        if (event.target.tagName === 'INPUT' || event.target.tagName === 'SELECT') {
            return; // Don't interfere with input fields
        }

        switch (event.code) {
            case 'Space':
                event.preventDefault();
                this.togglePlay();
                break;
            case 'ArrowLeft':
                event.preventDefault();
                this.previousStep();
                break;
            case 'ArrowRight':
                event.preventDefault();
                this.nextStep();
                break;
            case 'Home':
                event.preventDefault();
                this.goToStep(0);
                break;
            case 'End':
                event.preventDefault();
                this.goToStep(this.totalSteps - 1);
                break;
        }
    }

    showLoading(show) {
        const indicator = document.getElementById('loading-indicator');
        if (show) {
            indicator.classList.remove('hidden');
        } else {
            indicator.classList.add('hidden');
        }
    }

    showError(message) {
        const errorDisplay = document.getElementById('error-display');
        errorDisplay.textContent = message;
        errorDisplay.classList.remove('hidden');

        // Auto-hide after 5 seconds
        setTimeout(() => {
            errorDisplay.classList.add('hidden');
        }, 5000);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new VisualizerApp();
});