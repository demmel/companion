/**
 * Graph renderer for DAG memory visualization using D3.js
 * Handles rendering of nodes and edges with visual differentiation
 */

class GraphRenderer {
    constructor(svgSelector) {
        this.svg = d3.select(svgSelector);
        this.container = this.svg.append('g').attr('class', 'graph-container');

        // Initialize zoom behavior
        this.zoom = d3.zoom()
            .scaleExtent([0.1, 3])
            .on('zoom', (event) => {
                this.container.attr('transform', event.transform);
            });

        this.svg.call(this.zoom);

        // Initialize simulation
        this.simulation = d3.forceSimulation()
            .force('link', d3.forceLink().id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter())
            .force('collision', d3.forceCollide().radius(30));

        // Define patterns for confidence levels
        this.definePatterns();

        // Initialize empty data
        this.currentData = { nodes: [], edges: [] };
        this.selectedNode = null;
    }

    definePatterns() {
        const defs = this.svg.append('defs');

        // Diagonal stripes pattern
        const stripesPattern = defs.append('pattern')
            .attr('id', 'diagonal-stripes')
            .attr('patternUnits', 'userSpaceOnUse')
            .attr('width', 8)
            .attr('height', 8);

        stripesPattern.append('rect')
            .attr('width', 8)
            .attr('height', 8)
            .attr('fill', 'currentColor');

        stripesPattern.append('path')
            .attr('d', 'M0,8 L8,0 M-2,2 L2,-2 M6,10 L10,6')
            .attr('stroke', 'white')
            .attr('stroke-width', 2);

        // Dots pattern
        const dotsPattern = defs.append('pattern')
            .attr('id', 'dots')
            .attr('patternUnits', 'userSpaceOnUse')
            .attr('width', 12)
            .attr('height', 12);

        dotsPattern.append('circle')
            .attr('cx', 6)
            .attr('cy', 6)
            .attr('r', 2)
            .attr('fill', 'white');
    }

    updateGraph(data) {
        console.log('GraphRenderer.updateGraph called with:', {
            nodeCount: data.nodes.length,
            edgeCount: data.edges.length,
            sampleNode: data.nodes[0],
            sampleEdge: data.edges[0]
        });

        this.currentData = data;

        // Update simulation size
        const rect = this.svg.node().getBoundingClientRect();
        this.simulation
            .force('center', d3.forceCenter(rect.width / 2, rect.height / 2))
            .alpha(1);

        this.renderNodes();
        this.renderEdges();

        // Update simulation with new data
        this.simulation.nodes(data.nodes);
        this.simulation.force('link').links(data.edges);

        // Debug simulation state
        console.log('Simulation nodes:', this.simulation.nodes().length);
        console.log('Simulation links:', this.simulation.force('link').links().length);

        // Restart simulation
        this.simulation.alpha(1).restart();
    }

    renderEdges() {
        const edges = this.container
            .selectAll('.edge')
            .data(this.currentData.edges, d => d.id);

        // Remove old edges
        edges.exit().remove();

        // Add new edges
        const edgeEnter = edges.enter()
            .append('line')
            .attr('class', 'edge')
            .attr('marker-end', 'url(#arrowhead)');

        // Update all edges
        const edgeUpdate = edgeEnter.merge(edges);

        edgeUpdate
            .attr('stroke', d => d.color)
            .attr('stroke-width', d => d.width)
            .attr('stroke-dasharray', d => this.getStrokeDashArray(d.line_style))
            .classed('highlighted', d => d.is_new)
            .on('click', (event, d) => this.onEdgeClick(event, d))
            .on('mouseover', (event, d) => this.onEdgeHover(event, d))
            .on('mouseout', (event, d) => this.onEdgeLeave(event, d));

        // Add arrowhead marker
        if (!this.svg.select('#arrowhead').node()) {
            this.svg.select('defs').append('marker')
                .attr('id', 'arrowhead')
                .attr('viewBox', '0 -5 10 10')
                .attr('refX', 15)
                .attr('refY', 0)
                .attr('markerWidth', 6)
                .attr('markerHeight', 6)
                .attr('orient', 'auto')
                .append('path')
                .attr('d', 'M0,-5L10,0L0,5')
                .attr('fill', '#666');
        }

        // Update positions on simulation tick
        this.simulation.on('tick', () => {
            edgeUpdate
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
        });
    }

    renderNodes() {
        const nodes = this.container
            .selectAll('.node-group')
            .data(this.currentData.nodes, d => d.id);

        // Remove old nodes
        nodes.exit().remove();

        // Add new node groups
        const nodeEnter = nodes.enter()
            .append('g')
            .attr('class', 'node-group')
            .call(d3.drag()
                .on('start', (event, d) => this.dragStarted(event, d))
                .on('drag', (event, d) => this.dragged(event, d))
                .on('end', (event, d) => this.dragEnded(event, d)));

        // Add circles to new nodes
        nodeEnter.append('circle')
            .attr('class', 'node')
            .attr('r', 20);

        // Add labels to new nodes
        nodeEnter.append('text')
            .attr('class', 'node-label')
            .attr('dy', '0.35em');

        // Update all nodes
        const nodeUpdate = nodeEnter.merge(nodes);

        // Update circles
        nodeUpdate.select('.node')
            .attr('fill', d => d.color)
            .attr('stroke', d => d.in_context ? '#333' : '#666')
            .attr('stroke-width', d => d.in_context ? 3 : 1)
            .attr('stroke-dasharray', d => d.in_context ? 'none' : '5,5')
            .attr('class', d => `node ${this.getConfidenceClass(d.pattern)}`)
            .classed('highlighted', d => d.is_new || d.is_modified)
            .classed('selected', d => this.selectedNode && this.selectedNode.id === d.id)
            .on('click', (event, d) => this.onNodeClick(event, d))
            .on('mouseover', (event, d) => this.onNodeHover(event, d))
            .on('mouseout', (event, d) => this.onNodeLeave(event, d));

        // Update labels
        nodeUpdate.select('.node-label')
            .text(d => this.getNodeLabel(d));

        // Update positions on simulation tick
        this.simulation.on('tick', () => {
            nodeUpdate.attr('transform', d => `translate(${d.x},${d.y})`);
        });
    }

    getStrokeDashArray(lineStyle) {
        switch (lineStyle) {
            case 'dashed': return '8,4';
            case 'dotted': return '3,3';
            default: return 'none';
        }
    }

    getConfidenceClass(pattern) {
        switch (pattern) {
            case 'diagonal-stripes': return 'confidence-diagonal-stripes';
            case 'dots': return 'confidence-dots';
            case 'transparent': return 'confidence-transparent';
            case 'error-outline': return 'confidence-error-outline';
            case 'false-overlay': return 'confidence-false-overlay';
            default: return 'confidence-solid';
        }
    }

    getNodeLabel(node) {
        // Show memory type and significance
        const typeShort = node.memory_type.substring(0, 4).toUpperCase();
        const sigShort = Math.round(node.emotional_significance * 100);
        return `${typeShort}:${sigShort}%`;
    }

    // Event handlers
    onNodeClick(event, node) {
        event.stopPropagation();
        this.selectNode(node);

        // Dispatch custom event for external handlers
        window.dispatchEvent(new CustomEvent('nodeSelected', {
            detail: { node }
        }));
    }

    onNodeHover(event, node) {
        // Show tooltip or highlight connected edges
        this.highlightConnectedEdges(node.id);
    }

    onNodeLeave(event, node) {
        this.clearHighlights();
    }

    onEdgeClick(event, edge) {
        event.stopPropagation();

        // Dispatch custom event for external handlers
        window.dispatchEvent(new CustomEvent('edgeSelected', {
            detail: { edge }
        }));
    }

    onEdgeHover(event, edge) {
        // Highlight connected nodes
        this.highlightNodes([edge.source, edge.target]);
    }

    onEdgeLeave(event, edge) {
        this.clearHighlights();
    }

    // Interaction methods
    selectNode(node) {
        this.selectedNode = node;

        // Update visual selection
        this.container.selectAll('.node')
            .classed('selected', d => d.id === node.id);
    }

    highlightConnectedEdges(nodeId) {
        this.container.selectAll('.edge')
            .classed('highlighted', d =>
                d.source.id === nodeId || d.target.id === nodeId
            );
    }

    highlightNodes(nodeIds) {
        const idSet = new Set(nodeIds.map(n => typeof n === 'string' ? n : n.id));
        this.container.selectAll('.node')
            .classed('highlighted', d => idSet.has(d.id));
    }

    clearHighlights() {
        this.container.selectAll('.highlighted')
            .classed('highlighted', false);
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
    resetView() {
        this.svg.transition()
            .duration(750)
            .call(this.zoom.transform, d3.zoomIdentity);
    }

    fitToScreen() {
        if (this.currentData.nodes.length === 0) return;

        const bounds = this.getBounds();
        const fullWidth = this.svg.node().getBoundingClientRect().width;
        const fullHeight = this.svg.node().getBoundingClientRect().height;

        const width = bounds.maxX - bounds.minX;
        const height = bounds.maxY - bounds.minY;

        const midX = (bounds.minX + bounds.maxX) / 2;
        const midY = (bounds.minY + bounds.maxY) / 2;

        const scale = Math.min(fullWidth / width, fullHeight / height) * 0.8;

        const translate = [fullWidth / 2 - scale * midX, fullHeight / 2 - scale * midY];

        this.svg.transition()
            .duration(750)
            .call(this.zoom.transform, d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale));
    }

    getBounds() {
        const nodes = this.currentData.nodes;
        if (nodes.length === 0) return { minX: 0, minY: 0, maxX: 100, maxY: 100 };

        // Filter out undefined positions
        const validNodes = nodes.filter(d => d.x !== undefined && d.y !== undefined);
        if (validNodes.length === 0) return { minX: 0, minY: 0, maxX: 100, maxY: 100 };

        return {
            minX: d3.min(validNodes, d => d.x) - 50,
            minY: d3.min(validNodes, d => d.y) - 50,
            maxX: d3.max(validNodes, d => d.x) + 50,
            maxY: d3.max(validNodes, d => d.y) + 50
        };
    }

    // Clear the graph
    clear() {
        this.container.selectAll('*').remove();
        this.selectedNode = null;
        this.currentData = { nodes: [], edges: [] };
        this.simulation.nodes([]).force('link').links([]);
    }
}