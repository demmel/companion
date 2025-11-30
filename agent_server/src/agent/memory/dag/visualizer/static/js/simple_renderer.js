/**
 * Simplified graph renderer that definitely works
 * Uses basic D3 without complex force simulation
 */

class SimpleGraphRenderer {
    constructor(svgSelector) {
        this.svg = d3.select(svgSelector);
        this.container = this.svg.append('g').attr('class', 'graph-container');

        // Simple zoom
        this.zoom = d3.zoom()
            .scaleExtent([0.1, 3])
            .on('zoom', (event) => {
                this.container.attr('transform', event.transform);
            });
        this.svg.call(this.zoom);

        this.currentData = { nodes: [], edges: [] };
    }

    updateGraph(data) {
        console.log('SimpleGraphRenderer: updating with', data.nodes.length, 'nodes', data.edges.length, 'edges');

        this.currentData = data;

        // Clear existing
        this.container.selectAll('*').remove();

        if (data.nodes.length === 0) {
            console.log('No nodes to render');
            return;
        }

        // Use simple grid layout
        const rect = this.svg.node().getBoundingClientRect();
        const width = rect.width || 800;
        const height = rect.height || 600;

        // Calculate grid dimensions - optimize for large node counts
        const nodeCount = data.nodes.length;
        let cols, rows;

        if (nodeCount <= 16) {
            // Small grids - use square layout
            cols = Math.ceil(Math.sqrt(nodeCount));
            rows = Math.ceil(nodeCount / cols);
        } else {
            // Large grids - use wider rectangle (more columns than rows)
            cols = Math.ceil(Math.sqrt(nodeCount * 1.5));
            rows = Math.ceil(nodeCount / cols);
        }

        const minCellSize = 50; // Minimum space per node
        const cellWidth = Math.max(minCellSize, width * 0.9 / cols);
        const cellHeight = Math.max(minCellSize, height * 0.9 / rows);
        const offsetX = (width - cols * cellWidth) / 2;
        const offsetY = (height - rows * cellHeight) / 2;

        // Position nodes in grid
        data.nodes.forEach((node, i) => {
            const col = i % cols;
            const row = Math.floor(i / cols);
            node.x = offsetX + col * cellWidth + cellWidth / 2;
            node.y = offsetY + row * cellHeight + cellHeight / 2;
        });

        // Draw edges first (so they appear behind nodes)
        this.renderEdges(data.edges);

        // Draw nodes
        this.renderNodes(data.nodes);
    }

    renderEdges(edges) {
        const edgeGroups = this.container
            .selectAll('.edge')
            .data(edges, d => d.id)
            .enter()
            .append('g')
            .attr('class', 'edge');

        edgeGroups.append('line')
            .attr('x1', d => {
                const source = this.currentData.nodes.find(n => n.id === d.source);
                return source ? source.x : 0;
            })
            .attr('y1', d => {
                const source = this.currentData.nodes.find(n => n.id === d.source);
                return source ? source.y : 0;
            })
            .attr('x2', d => {
                const target = this.currentData.nodes.find(n => n.id === d.target);
                return target ? target.x : 0;
            })
            .attr('y2', d => {
                const target = this.currentData.nodes.find(n => n.id === d.target);
                return target ? target.y : 0;
            })
            .attr('stroke', d => d.color || '#666')
            .attr('stroke-width', d => d.width || 2)
            .attr('stroke-dasharray', d => {
                if (d.line_style === 'dashed') return '8,4';
                if (d.line_style === 'dotted') return '3,3';
                return 'none';
            })
            .attr('marker-end', 'url(#arrowhead)');

        // Add arrow marker
        if (!this.svg.select('#arrowhead').node()) {
            this.svg.append('defs').append('marker')
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
    }

    renderNodes(nodes) {
        const nodeGroups = this.container
            .selectAll('.node-group')
            .data(nodes, d => d.id)
            .enter()
            .append('g')
            .attr('class', 'node-group')
            .attr('transform', d => `translate(${d.x}, ${d.y})`);

        // Add circles
        nodeGroups.append('circle')
            .attr('r', 20)
            .attr('fill', d => d.color || '#4488FF')
            .attr('stroke', d => d.in_context ? '#333' : '#666')
            .attr('stroke-width', d => d.in_context ? 3 : 1)
            .attr('stroke-dasharray', d => d.in_context ? 'none' : '5,5')
            .on('click', (event, d) => {
                console.log('Node clicked:', d);
                window.dispatchEvent(new CustomEvent('nodeSelected', { detail: { node: d } }));
            });

        // Add labels
        nodeGroups.append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', '0.35em')
            .attr('font-size', '10px')
            .attr('fill', '#333')
            .text(d => {
                // Show memory type abbreviation
                return d.memory_type ? d.memory_type.substring(0, 4).toUpperCase() : 'MEM';
            });

        // Add ID labels below
        nodeGroups.append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', '35px')
            .attr('font-size', '8px')
            .attr('fill', '#666')
            .text(d => d.id ? d.id.substring(0, 8) : '');
    }

    fitToScreen() {
        if (this.currentData.nodes.length === 0) return;

        const bounds = this.getBounds();
        const rect = this.svg.node().getBoundingClientRect();

        const width = bounds.maxX - bounds.minX;
        const height = bounds.maxY - bounds.minY;
        const centerX = (bounds.minX + bounds.maxX) / 2;
        const centerY = (bounds.minY + bounds.maxY) / 2;

        const scale = Math.min(rect.width / width, rect.height / height) * 0.8;
        const translateX = rect.width / 2 - centerX * scale;
        const translateY = rect.height / 2 - centerY * scale;

        this.svg.transition()
            .duration(750)
            .call(this.zoom.transform, d3.zoomIdentity.translate(translateX, translateY).scale(scale));
    }

    getBounds() {
        const nodes = this.currentData.nodes;
        if (nodes.length === 0) return { minX: 0, minY: 0, maxX: 100, maxY: 100 };

        return {
            minX: Math.min(...nodes.map(d => d.x)) - 50,
            minY: Math.min(...nodes.map(d => d.y)) - 50,
            maxX: Math.max(...nodes.map(d => d.x)) + 50,
            maxY: Math.max(...nodes.map(d => d.y)) + 50
        };
    }

    clear() {
        this.container.selectAll('*').remove();
        this.currentData = { nodes: [], edges: [] };
    }
}