/**
 * Force-directed graph renderer for DAG memory visualization
 * Uses proper D3.js force simulation for planar graph layout
 */

class ForceGraphRenderer {
    constructor(svgSelector) {
        this.svg = d3.select(svgSelector);
        this.container = this.svg.append('g').attr('class', 'graph-container');

        // Initialize zoom behavior
        this.zoom = d3.zoom()
            .scaleExtent([0.1, 5])
            .on('zoom', (event) => {
                this.container.attr('transform', event.transform);
            });
        this.svg.call(this.zoom);

        // Get SVG dimensions
        this.updateDimensions();

        // Initialize force simulation with improved parameters for better separation
        this.simulation = d3.forceSimulation()
            .force('link', d3.forceLink().id(d => d.id).distance(120).strength(0.3))
            .force('charge', d3.forceManyBody().strength(-800).distanceMin(50))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(d => d.in_context ? 30 : 23).strength(1))
            .force('x', d3.forceX(this.width / 2).strength(0.05))
            .force('y', d3.forceY(this.height / 2).strength(0.05))
            .alphaDecay(0.01)
            .velocityDecay(0.3);

        // Add arrow marker for directed edges
        this.addArrowMarker();

        this.currentData = { nodes: [], edges: [] };
        this.selectedNode = null;
        this.nodePositions = new Map(); // Store node positions between updates
    }

    updateDimensions() {
        const rect = this.svg.node().getBoundingClientRect();
        this.width = rect.width || 800;
        this.height = rect.height || 600;
    }

    addArrowMarker() {
        const defs = this.svg.select('defs').empty() ? this.svg.append('defs') : this.svg.select('defs');

        defs.append('marker')
            .attr('id', 'arrowhead')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 20)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', '#666');
    }

    updateGraph(data) {
        console.log('ForceGraphRenderer: updating with', data.nodes.length, 'nodes', data.edges.length, 'edges');

        this.currentData = data;
        this.updateDimensions();

        // Update center force
        this.simulation.force('center', d3.forceCenter(this.width / 2, this.height / 2));

        // Create copies of nodes and edges for D3 (D3 modifies these objects)
        const nodes = data.nodes.map(d => ({ ...d }));
        const edges = data.edges.map(d => ({ ...d }));

        // Restore previous positions for existing nodes and identify new nodes
        let hasNewNodes = false;
        nodes.forEach(node => {
            const prevPos = this.nodePositions.get(node.id);
            if (prevPos) {
                // Restore previous position
                node.x = prevPos.x;
                node.y = prevPos.y;
                node.fx = prevPos.x; // Pin temporarily
                node.fy = prevPos.y;
            } else {
                // New node - will be positioned by force simulation
                hasNewNodes = true;
                // Start new nodes near center with some random offset
                node.x = this.width / 2 + (Math.random() - 0.5) * 100;
                node.y = this.height / 2 + (Math.random() - 0.5) * 100;
            }
        });

        // Validate edge references
        const nodeIds = new Set(nodes.map(n => n.id));
        const validEdges = edges.filter(edge => {
            const valid = nodeIds.has(edge.source) && nodeIds.has(edge.target);
            if (!valid) {
                console.warn('Invalid edge - node not found:', edge);
            }
            return valid;
        });

        console.log(`Using ${validEdges.length}/${edges.length} valid edges`);

        // Update simulation
        this.simulation.nodes(nodes);
        this.simulation.force('link').links(validEdges);

        // Render elements
        this.renderEdges(validEdges);
        this.renderNodes(nodes);

        // Only restart simulation if there are new nodes, otherwise just warm start
        if (hasNewNodes) {
            this.simulation.alpha(0.3).restart();

            // Unpin existing nodes after a short delay to allow gentle adjustment
            setTimeout(() => {
                nodes.forEach(node => {
                    if (this.nodePositions.has(node.id)) {
                        node.fx = null;
                        node.fy = null;
                    }
                });
            }, 500);
        } else {
            // Just update positions without major simulation restart
            this.simulation.alpha(0.1).restart();
        }

        // Set up tick handler
        this.simulation.on('tick', () => {
            this.updatePositions();
        });

        // Save current positions
        setTimeout(() => {
            nodes.forEach(node => {
                if (node.x !== undefined && node.y !== undefined) {
                    this.nodePositions.set(node.id, { x: node.x, y: node.y });
                }
            });
        }, 1000);
    }

    renderEdges(edges) {
        // Debug duplicate edge detection
        const duplicateReport = new Map();
        edges.forEach(edge => {
            const key = `${edge.source}→${edge.edge_type}→${edge.target}`;
            if (!duplicateReport.has(key)) {
                duplicateReport.set(key, []);
            }
            duplicateReport.get(key).push({
                id: edge.id.substring(0, 8),
                created: edge.created_at,
                in_context: edge.in_context
            });
        });

        // Log potential duplicates
        duplicateReport.forEach((edgeList, key) => {
            if (edgeList.length > 1) {
                console.warn(`🔄 DUPLICATE EDGES: ${key}`, edgeList);
            }
        });

        // Group edges by node pairs to handle overlapping edges
        const edgeGroups = new Map();
        edges.forEach(edge => {
            const key = `${edge.source}-${edge.target}`;
            const reverseKey = `${edge.target}-${edge.source}`;

            if (!edgeGroups.has(key)) {
                edgeGroups.set(key, []);
            }
            edgeGroups.get(key).push({...edge, groupKey: key, isReverse: false});
        });

        // Flatten edges with offset information
        const processedEdges = [];
        edgeGroups.forEach((groupEdges, key) => {
            groupEdges.forEach((edge, index) => {
                edge.groupIndex = index;
                edge.groupTotal = groupEdges.length;
                processedEdges.push(edge);
            });
        });

        const edgeSelection = this.container
            .selectAll('.edge')
            .data(processedEdges, d => d.id);

        // Remove old edges
        edgeSelection.exit().remove();

        // Add new edges as paths (for curved lines)
        const edgeEnter = edgeSelection.enter()
            .append('path')
            .attr('class', 'edge')
            .attr('marker-end', 'url(#arrowhead)')
            .attr('fill', 'none')
            .on('click', (event, d) => this.onEdgeClick(event, d))
            .on('mouseover', (event, d) => this.onEdgeHover(event, d))
            .on('mouseout', (event, d) => this.onEdgeLeave(event, d));

        // Update all edges
        const edgeUpdate = edgeEnter.merge(edgeSelection);

        edgeUpdate
            .attr('stroke', d => d.in_context ? (d.color || '#666') : '#CCC')
            .attr('stroke-width', d => d.in_context ? (d.width || 3) : 1)
            .attr('stroke-dasharray', d => this.getStrokeDashArray(d.line_style))
            .attr('opacity', d => d.in_context ? 0.8 : 0.3)
            .classed('highlighted', d => d.is_new)
            .classed('context-edge', d => d.in_context);

        this.edges = edgeUpdate;
    }

    renderNodes(nodes) {
        const nodeSelection = this.container
            .selectAll('.node-group')
            .data(nodes, d => d.id);

        // Remove old nodes
        nodeSelection.exit().remove();

        // Add new node groups
        const nodeEnter = nodeSelection.enter()
            .append('g')
            .attr('class', 'node-group')
            .call(d3.drag()
                .on('start', (event, d) => this.dragStarted(event, d))
                .on('drag', (event, d) => this.dragged(event, d))
                .on('end', (event, d) => this.dragEnded(event, d)));

        // Add circles
        nodeEnter.append('circle')
            .attr('class', 'node')
            .attr('r', 20);

        // Add labels
        nodeEnter.append('text')
            .attr('class', 'node-label')
            .attr('dy', '0.35em')
            .attr('text-anchor', 'middle')
            .attr('font-size', '10px')
            .attr('font-weight', 'bold')
            .attr('pointer-events', 'none');

        // Update all nodes
        const nodeUpdate = nodeEnter.merge(nodeSelection);

        // Update circles with better context differentiation
        nodeUpdate.select('.node')
            .attr('fill', d => d.in_context ? d.color || '#4488FF' : '#E8E8E8')
            .attr('stroke', d => d.in_context ? '#000' : '#999')
            .attr('stroke-width', d => d.in_context ? 4 : 2)
            .attr('stroke-dasharray', d => d.in_context ? 'none' : '3,3')
            .attr('opacity', d => d.in_context ? 1.0 : 0.4)
            .attr('r', d => d.in_context ? 25 : 18)
            .classed('highlighted', d => d.is_new || d.is_modified)
            .classed('selected', d => this.selectedNode && this.selectedNode.id === d.id)
            .classed('context-node', d => d.in_context)
            .on('click', (event, d) => this.onNodeClick(event, d))
            .on('mouseover', (event, d) => this.onNodeHover(event, d))
            .on('mouseout', (event, d) => this.onNodeLeave(event, d));

        // Update labels with better context differentiation
        nodeUpdate.select('.node-label')
            .text(d => this.getNodeLabel(d))
            .attr('font-weight', d => d.in_context ? 'bold' : 'normal')
            .attr('font-size', d => d.in_context ? '12px' : '9px')
            .attr('fill', d => d.in_context ? '#000' : '#666');

        this.nodes = nodeUpdate;
    }

    updatePositions() {
        if (this.edges) {
            this.edges.attr('d', d => this.createEdgePath(d));
        }

        if (this.nodes) {
            this.nodes
                .attr('transform', d => `translate(${d.x},${d.y})`);
        }
    }

    createEdgePath(d) {
        const source = d.source;
        const target = d.target;

        if (!source || !target || source.x === undefined || target.x === undefined) {
            return 'M0,0L0,0'; // Invalid path for missing data
        }

        // Single edge - straight line
        if (d.groupTotal === 1) {
            return `M${source.x},${source.y}L${target.x},${target.y}`;
        }

        // Multiple edges - create curved paths
        const dx = target.x - source.x;
        const dy = target.y - source.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        // Calculate perpendicular offset
        const offsetDistance = 15 * (d.groupIndex - (d.groupTotal - 1) / 2);

        // Perpendicular vector (rotated 90 degrees)
        const perpX = -dy / distance * offsetDistance;
        const perpY = dx / distance * offsetDistance;

        // Control point for quadratic curve
        const midX = (source.x + target.x) / 2 + perpX;
        const midY = (source.y + target.y) / 2 + perpY;

        return `M${source.x},${source.y}Q${midX},${midY} ${target.x},${target.y}`;
    }

    getStrokeDashArray(lineStyle) {
        switch (lineStyle) {
            case 'dashed': return '8,4';
            case 'dotted': return '3,3';
            default: return 'none';
        }
    }

    getNodeLabel(node) {
        // Show memory type abbreviation
        if (node.memory_type) {
            const typeShort = node.memory_type.substring(0, 4).toUpperCase();
            return typeShort;
        }
        return 'MEM';
    }

    // Event handlers
    onNodeClick(event, node) {
        event.stopPropagation();
        this.selectNode(node);

        // Dispatch custom event
        window.dispatchEvent(new CustomEvent('nodeSelected', {
            detail: { node }
        }));
    }

    onNodeHover(event, node) {
        this.highlightConnectedEdges(node.id);
    }

    onNodeLeave(event, node) {
        this.clearHighlights();
    }

    onEdgeClick(event, edge) {
        event.stopPropagation();

        // Dispatch custom event
        window.dispatchEvent(new CustomEvent('edgeSelected', {
            detail: { edge }
        }));
    }

    onEdgeHover(event, edge) {
        // Highlight connected nodes
        this.highlightNodes([edge.source.id, edge.target.id]);
    }

    onEdgeLeave(event, edge) {
        this.clearHighlights();
    }

    // Interaction methods
    selectNode(node) {
        this.selectedNode = node;

        if (this.nodes) {
            this.nodes.select('.node')
                .classed('selected', d => d.id === node.id);
        }
    }

    highlightConnectedEdges(nodeId) {
        if (this.edges) {
            this.edges
                .classed('highlighted', d =>
                    d.source.id === nodeId || d.target.id === nodeId
                );
        }
    }

    highlightNodes(nodeIds) {
        if (this.nodes) {
            const idSet = new Set(nodeIds);
            this.nodes.select('.node')
                .classed('highlighted', d => idSet.has(d.id));
        }
    }

    clearHighlights() {
        if (this.edges) {
            this.edges.classed('highlighted', false);
        }
        if (this.nodes) {
            this.nodes.select('.node').classed('highlighted', false);
        }
    }

    // Drag behavior
    dragStarted(event, d) {
        if (!event.active) this.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    dragEnded(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    // Utility methods
    fitToScreen() {
        if (this.currentData.nodes.length === 0) return;

        const bounds = this.getBounds();
        const padding = 50;

        const width = bounds.maxX - bounds.minX + 2 * padding;
        const height = bounds.maxY - bounds.minY + 2 * padding;

        const centerX = (bounds.minX + bounds.maxX) / 2;
        const centerY = (bounds.minY + bounds.maxY) / 2;

        const scale = Math.min(this.width / width, this.height / height) * 0.9;

        const translateX = this.width / 2 - centerX * scale;
        const translateY = this.height / 2 - centerY * scale;

        this.svg.transition()
            .duration(750)
            .call(this.zoom.transform, d3.zoomIdentity.translate(translateX, translateY).scale(scale));
    }

    getBounds() {
        const nodes = this.currentData.nodes.filter(d => d.x !== undefined && d.y !== undefined);
        if (nodes.length === 0) {
            return { minX: 0, minY: 0, maxX: this.width, maxY: this.height };
        }

        return {
            minX: d3.min(nodes, d => d.x),
            minY: d3.min(nodes, d => d.y),
            maxX: d3.max(nodes, d => d.x),
            maxY: d3.max(nodes, d => d.y)
        };
    }

    resetView() {
        this.svg.transition()
            .duration(750)
            .call(this.zoom.transform, d3.zoomIdentity);
    }

    clear() {
        this.container.selectAll('*').remove();
        this.selectedNode = null;
        this.currentData = { nodes: [], edges: [] };
        this.nodePositions.clear(); // Reset position memory
        this.simulation.nodes([]).force('link').links([]);
        this.addArrowMarker();
    }
}