#!/usr/bin/env python3
"""
Knowledge Graph Visualization

Creates interactive visualizations for the KNN-based knowledge graph system:
- Hyperedge node network for n-ary relationships
- t-SNE/UMAP embedding space visualization
- KNN decision dashboard
"""

import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE

from agent.experiments.knowledge_graph.knowledge_graph_prototype import (
    KnowledgeExperienceGraph,
    GraphNode,
)
from agent.experiments.knowledge_graph.n_ary_relationship import (
    NaryRelationship,
)
from agent.experiments.knowledge_graph.relationship_schema_bank import (
    RelationshipSchemaBank,
)

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingPoint:
    """A point in embedding space with metadata"""

    id: str
    text: str
    embedding: np.ndarray
    type: str  # 'trigger', 'entity', 'schema'
    category: Optional[str] = None  # entity type or schema category
    confidence: Optional[float] = None


@dataclass
class KGVisualizationData:
    """All data needed for visualization"""

    entities: List[GraphNode]
    relationships: List[NaryRelationship]
    embedding_points: List[EmbeddingPoint]
    knn_decisions: List[Dict[str, Any]]


class KnowledgeGraphVisualizer:
    """Creates interactive visualizations of the knowledge graph system"""

    def __init__(self, output_path: str = "kg_visualization.html"):
        self.output_path = output_path

    def create_visualization(
        self,
        kg: KnowledgeExperienceGraph,
        schema_bank: RelationshipSchemaBank,
        trigger_embeddings: Optional[List[EmbeddingPoint]] = None,
    ) -> str:
        """Create complete interactive tabbed visualization"""

        logger.info("Creating knowledge graph visualizations...")

        # Collect all data
        viz_data = self._collect_visualization_data(kg, schema_bank, trigger_embeddings)

        # Create three separate full-screen visualizations
        network_fig = self._create_network_tab(viz_data)
        embedding_fig = self._create_embedding_tab(viz_data)
        stats_fig = self._create_statistics_tab(viz_data)

        # Create tabbed HTML interface
        html_content = self._create_tabbed_html(network_fig, embedding_fig, stats_fig)

        # Save to HTML
        with open(self.output_path, "w") as f:
            f.write(html_content)
        logger.info(f"Tabbed visualization saved to {self.output_path}")

        return self.output_path

    def _collect_visualization_data(
        self,
        kg: KnowledgeExperienceGraph,
        schema_bank: RelationshipSchemaBank,
        trigger_embeddings: Optional[List[EmbeddingPoint]],
    ) -> KGVisualizationData:
        """Collect all data needed for visualization"""

        entities = list(kg.nodes.values())
        relationships = kg.get_nary_relationships()

        # Collect embedding points
        embedding_points = []

        # Add trigger embeddings if provided
        if trigger_embeddings:
            embedding_points.extend(trigger_embeddings)

        # Add entity embeddings
        for entity in entities:
            if entity.embedding is not None and entity.get_embedding().size > 0:
                embedding_points.append(
                    EmbeddingPoint(
                        id=entity.id,
                        text=f"{entity.name}: {entity.description[:100]}...",
                        embedding=entity.get_embedding(),
                        type="entity",
                        category=entity.node_type.value,
                    )
                )

        # Add schema embeddings
        for schema_name, schema_entry in schema_bank.relationship_schemas.items():
            if schema_entry.embedding:
                embedding_points.append(
                    EmbeddingPoint(
                        id=schema_name,
                        text=f"{schema_name} ({schema_entry.category}): {schema_entry.description}",
                        embedding=np.array(schema_entry.embedding),
                        type="schema",
                        category=schema_entry.category,
                    )
                )

        return KGVisualizationData(
            entities=entities,
            relationships=relationships,
            embedding_points=embedding_points,
            knn_decisions=[],  # TODO: collect KNN decisions during processing
        )

    def _create_network_tab(self, viz_data: KGVisualizationData) -> go.Figure:
        """Create full-screen network visualization tab"""
        fig = go.Figure()
        self._add_hyperedge_network(fig, viz_data, None, None)

        fig.update_layout(
            title=dict(text="N-ary Knowledge Graph", x=0.5, y=0.98, font=dict(size=16)),
            showlegend=True,
            hovermode="closest",
            height=None,  # Let it fill available space
            margin=dict(l=0, r=0, t=40, b=0),  # Minimal margins
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        )

        return fig

    def _create_embedding_tab(self, viz_data: KGVisualizationData) -> go.Figure:
        """Create full-screen embedding space visualization tab"""
        fig = go.Figure()
        self._add_embedding_space(fig, viz_data, None, None)

        fig.update_layout(
            title=dict(
                text="Embedding Space (t-SNE)", x=0.5, y=0.98, font=dict(size=16)
            ),
            showlegend=True,
            hovermode="closest",
            height=None,  # Let it fill available space
            margin=dict(
                l=50, r=0, t=40, b=50
            ),  # Minimal margins with space for axis labels
            xaxis_title="t-SNE Dimension 1",
            yaxis_title="t-SNE Dimension 2",
        )

        return fig

    def _create_statistics_tab(self, viz_data: KGVisualizationData) -> go.Figure:
        """Create full-screen statistics dashboard tab"""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Entity Types",
                "Relationship Types",
                "Entity Resolution Stats",
                "Network Metrics",
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}],
            ],
        )

        self._add_entity_statistics(fig, viz_data, row=1, col=1)
        self._add_relationship_statistics(fig, viz_data, row=1, col=2)
        self._add_resolution_statistics(fig, viz_data, row=2, col=1)
        self._add_network_metrics(fig, viz_data, row=2, col=2)

        fig.update_layout(
            title=dict(
                text="Knowledge Graph Statistics", x=0.5, y=0.98, font=dict(size=16)
            ),
            height=None,  # Let it fill available space
            margin=dict(l=20, r=20, t=40, b=20),  # Minimal margins
            showlegend=False,
        )

        return fig

    def _add_hyperedge_network(
        self,
        fig,
        viz_data: KGVisualizationData,
        row: Optional[int] = None,
        col: Optional[int] = None,
    ):
        """Add hyperedge network visualization for n-ary relationships"""

        entity_nodes = {e.id: e for e in viz_data.entities}
        relationship_nodes = {r.id: r for r in viz_data.relationships}

        # Create node positions using spring layout approximation
        positions = self._calculate_network_positions(viz_data)

        # Add entity nodes
        entity_x = []
        entity_y = []
        entity_text = []
        entity_colors = []
        entity_sizes = []

        color_map = {
            "person": "#FF6B6B",
            "object": "#4ECDC4",
            "concept": "#45B7D1",
            "emotion": "#96CEB4",
            "experience": "#FFEAA7",
        }

        for entity in viz_data.entities:
            pos = positions[f"n_{entity.id}"]
            entity_x.append(pos[0])
            entity_y.append(pos[1])
            entity_text.append(f"{entity.name}<br>{entity.node_type.value}")
            entity_colors.append(color_map.get(entity.node_type.value, "#DDA0DD"))
            # Size by number of relationships - larger base size
            relationship_count = len(
                [
                    r
                    for r in viz_data.relationships
                    if entity.id in r.participants.values()
                ]
            )
            entity_sizes.append(
                max(
                    8, 8 + min(relationship_count * 2, 12)
                )  # Smaller nodes with limited max size
            )

        trace_kwargs = {
            "x": entity_x,
            "y": entity_y,
            "mode": "markers+text",
            "text": [t.split("<br>")[0] for t in entity_text],  # Just names
            "textposition": "middle center",
            "textfont": dict(
                size=8,  # Much smaller entity text
                color="black",
                family="Arial, sans-serif",
            ),
            "hovertext": entity_text,
            "marker": dict(
                size=entity_sizes,
                color=entity_colors,
                line=dict(
                    width=3, color="white"
                ),  # Thicker border for better visibility
                opacity=0.9,  # Higher opacity
            ),
            "name": "Entities",
            "showlegend": True,
        }

        if row is not None and col is not None:
            fig.add_trace(go.Scatter(**trace_kwargs), row=row, col=col)
        else:
            fig.add_trace(go.Scatter(**trace_kwargs))

        # Add relationship hyperedge nodes
        rel_x = []
        rel_y = []
        rel_text = []
        rel_colors = []

        # Use a single, readable color for all relationships
        relationship_color = "#4472C4"  # Professional blue

        # Add edges from entities to relationship nodes with role labels
        edge_x = []
        edge_y = []
        edge_text = []
        edge_text_x = []
        edge_text_y = []

        for i, rel in enumerate(viz_data.relationships):
            rel_pos = positions[f"r_{rel.id}"]
            rel_x.append(rel_pos[0])
            rel_y.append(rel_pos[1])

            # Create relationship node text (show relationship type)
            rel_text.append(rel.relationship_type)

            # Use consistent color for all relationships
            rel_colors.append(relationship_color)

            # Add edges from relationship to each participant with role labels
            for role, participant_id in rel.participants.items():
                part_pos = positions[f"n_{participant_id}"]
                # Add edge line
                edge_x.extend([rel_pos[0], part_pos[0], None])
                edge_y.extend([rel_pos[1], part_pos[1], None])

                # Add role label at midpoint of edge
                mid_x = (rel_pos[0] + part_pos[0]) / 2
                mid_y = (rel_pos[1] + part_pos[1]) / 2
                edge_text_x.append(mid_x)
                edge_text_y.append(mid_y)
                edge_text.append(role)

        # Add relationship nodes with text labels
        rel_trace_kwargs = {
            "x": rel_x,
            "y": rel_y,
            "mode": "markers+text",
            "text": rel_text,
            "textposition": "middle center",
            "textfont": dict(
                size=9,  # Smaller text to reduce clutter
                color="white",
                family="Arial, sans-serif",
                weight="bold",
            ),
            "marker": dict(
                size=28,  # Smaller diamond nodes
                color=rel_colors,
                symbol="diamond",
                line=dict(width=2, color="white"),
                opacity=1.0,
            ),
            "name": "Relationships",
            "showlegend": True,
        }

        if row is not None and col is not None:
            fig.add_trace(go.Scatter(**rel_trace_kwargs), row=row, col=col)
        else:
            fig.add_trace(go.Scatter(**rel_trace_kwargs))

        # Add edges
        edge_trace_kwargs = {
            "x": edge_x,
            "y": edge_y,
            "mode": "lines",
            "line": dict(width=1, color="gray"),
            "hoverinfo": "none",
            "showlegend": False,
        }

        if row is not None and col is not None:
            fig.add_trace(go.Scatter(**edge_trace_kwargs), row=row, col=col)
        else:
            fig.add_trace(go.Scatter(**edge_trace_kwargs))

        # Add semantic role labels on edges - show on hover only
        edge_label_kwargs = {
            "x": edge_text_x,
            "y": edge_text_y,
            "mode": "markers",
            "hovertext": edge_text,  # Show role on hover
            "hoverinfo": "text",
            "marker": dict(
                size=8,  # Small invisible markers
                color="rgba(0,0,0,0)",  # Transparent
                symbol="circle",
                opacity=0,
            ),
            "showlegend": False,
        }

        if row is not None and col is not None:
            fig.add_trace(go.Scatter(**edge_label_kwargs), row=row, col=col)
        else:
            fig.add_trace(go.Scatter(**edge_label_kwargs))

    def _calculate_network_positions(
        self, viz_data: KGVisualizationData
    ) -> Dict[str, np.ndarray]:
        """Calculate 2D positions using Fruchterman-Reingold force-directed algorithm"""

        entities = {f"n_{e.id}": e for e in viz_data.entities} | {
            f"r_{r.id}": r for r in viz_data.relationships
        }  # Include relationships as nodes

        n_entities = len(entities)

        if n_entities == 0:
            return {}

        # Build edge list from relationships
        edges = set()
        for rel in viz_data.relationships:
            participants = list(rel.participants.values())
            for i in range(len(participants)):
                if participants[i] in entities and rel.id in entities:
                    edges.add((participants[i], rel.id))

        return self._fruchterman_reingold_layout(entities, edges)

    def _fruchterman_reingold_layout(self, entities, edges):
        """Implement Fruchterman-Reingold force-directed algorithm"""
        n = len(entities)
        entity_list = list(entities.keys())

        # Algorithm parameters (from FR paper)
        area = n * 100  # Layout area
        k = np.sqrt(area / n)  # Optimal distance between nodes
        max_iterations = 100 * n  # Temperature steps

        # Initialize positions randomly
        positions = {}
        for entity in entity_list:
            positions[entity] = np.random.uniform(-area / 2, area / 2, 2)

        # Temperature schedule (starts high, decreases)
        initial_temp = area / 10

        for iteration in range(max_iterations):
            # Calculate temperature (decreases linearly)
            temperature = initial_temp * (1 - iteration / max_iterations)

            # Initialize displacement vectors
            displacement = {entity: np.zeros(2) for entity in entity_list}

            # Calculate repulsive forces between all pairs of nodes
            for i in range(n):
                for j in range(i + 1, n):
                    v = entity_list[i]
                    u = entity_list[j]

                    delta = positions[v] - positions[u]
                    distance = max(
                        np.linalg.norm(delta), 0.01
                    )  # Avoid division by zero

                    # Repulsive force: f_r(d) = k²/d
                    force_magnitude = k * k / distance
                    force_direction = delta / distance

                    displacement[v] += force_direction * force_magnitude
                    displacement[u] -= force_direction * force_magnitude

            # Calculate attractive forces along edges
            for edge in edges:
                v, u = edge
                if v in positions and u in positions:
                    delta = positions[v] - positions[u]
                    distance = max(np.linalg.norm(delta), 0.01)

                    # Attractive force: f_a(d) = d²/k
                    force_magnitude = distance * distance / k
                    force_direction = delta / distance

                    displacement[v] -= force_direction * force_magnitude
                    displacement[u] += force_direction * force_magnitude

        # Limit displacement by temperature and update positions
        for entity in entity_list:
            disp_magnitude = np.linalg.norm(displacement[entity])
            if disp_magnitude > 0:
                # Limit displacement to current temperature
                limited_displacement = (
                    displacement[entity]
                    * min(disp_magnitude, temperature)
                    / disp_magnitude
                )
                positions[entity] += limited_displacement

                # Keep within bounds
                positions[entity] = np.clip(positions[entity], -area, area)

        # Center and translate to positive coordinates
        if positions:
            all_pos = np.array(list(positions.values()))
            min_x, min_y = np.min(all_pos, axis=0)
            for entity in positions:
                positions[entity] -= np.array([min_x - 50, min_y - 50])

            return positions

    def _add_embedding_space(
        self,
        fig,
        viz_data: KGVisualizationData,
        row: Optional[int] = None,
        col: Optional[int] = None,
    ):
        """Add t-SNE embedding space visualization"""

        if not viz_data.embedding_points:
            return

        # Prepare embeddings for t-SNE
        embeddings = np.array([ep.embedding for ep in viz_data.embedding_points])

        if embeddings.shape[0] < 2:
            return

        # Run t-SNE
        tsne = TSNE(
            n_components=2, random_state=42, perplexity=min(30, embeddings.shape[0] - 1)
        )
        embedding_2d = tsne.fit_transform(embeddings)

        # Color by type
        type_colors = {"trigger": "#FF6B6B", "entity": "#4ECDC4", "schema": "#FFD93D"}

        for point_type in ["trigger", "entity", "schema"]:
            mask = [ep.type == point_type for ep in viz_data.embedding_points]
            if not any(mask):
                continue

            indices = [i for i, m in enumerate(mask) if m]

            embed_trace_kwargs = {
                "x": embedding_2d[indices, 0],
                "y": embedding_2d[indices, 1],
                "mode": "markers",
                "text": [viz_data.embedding_points[i].text for i in indices],
                "hovertext": [
                    f"{viz_data.embedding_points[i].type}: {viz_data.embedding_points[i].text}"
                    for i in indices
                ],
                "marker": dict(size=8, color=type_colors[point_type], opacity=0.7),
                "name": f"{point_type.title()}s",
                "showlegend": True,
            }

            if row is not None and col is not None:
                fig.add_trace(go.Scatter(**embed_trace_kwargs), row=row, col=col)
            else:
                fig.add_trace(go.Scatter(**embed_trace_kwargs))

    def _add_entity_statistics(
        self, fig, viz_data: KGVisualizationData, row: int, col: int
    ):
        """Add entity type statistics"""

        entity_types = {}
        for entity in viz_data.entities:
            entity_types[entity.node_type.value] = (
                entity_types.get(entity.node_type.value, 0) + 1
            )

        if entity_types:
            fig.add_trace(
                go.Bar(
                    x=list(entity_types.keys()),
                    y=list(entity_types.values()),
                    name="Entity Types",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    def _add_relationship_statistics(
        self, fig, viz_data: KGVisualizationData, row: int, col: int
    ):
        """Add relationship type statistics"""

        rel_types = {}
        for rel in viz_data.relationships:
            rel_type = rel.relationship_type
            rel_types[rel_type] = rel_types.get(rel_type, 0) + 1

        if rel_types:
            fig.add_trace(
                go.Bar(
                    x=list(rel_types.keys()),
                    y=list(rel_types.values()),
                    name="Relationship Types",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    def _add_resolution_statistics(
        self, fig, viz_data: KGVisualizationData, row: int, col: int
    ):
        """Add entity resolution success/failure statistics"""

        # Placeholder data - in real implementation this would come from resolution logs
        resolution_stats = {
            "Successful": len(viz_data.entities)
            * 0.8,  # Estimate based on entity count
            "Failed": len(viz_data.entities) * 0.2,
        }

        fig.add_trace(
            go.Bar(
                x=list(resolution_stats.keys()),
                y=list(resolution_stats.values()),
                name="Resolution Stats",
                showlegend=False,
                marker_color=["green", "red"],
            ),
            row=row,
            col=col,
        )

    def _add_network_metrics(
        self, fig, viz_data: KGVisualizationData, row: int, col: int
    ):
        """Add network connectivity metrics"""

        # Calculate basic network metrics
        num_entities = len(viz_data.entities)
        num_relationships = len(viz_data.relationships)

        # Calculate average participants per relationship
        total_participants = sum(
            len(rel.participants) for rel in viz_data.relationships
        )
        avg_participants = (
            total_participants / num_relationships if num_relationships > 0 else 0
        )

        metrics = {
            "Entities": num_entities,
            "Relationships": num_relationships,
            "Avg Participants": avg_participants,
        }

        fig.add_trace(
            go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                name="Network Metrics",
                showlegend=False,
                marker_color="skyblue",
            ),
            row=row,
            col=col,
        )

    def _create_tabbed_html(
        self, network_fig: go.Figure, embedding_fig: go.Figure, stats_fig: go.Figure
    ) -> str:
        """Create HTML with tabbed interface for the three visualizations"""

        # Convert figures to HTML with inline Plotly to include all necessary scripts
        network_html = network_fig.to_html(
            include_plotlyjs="inline", div_id="network-tab"
        )
        embedding_html = embedding_fig.to_html(
            include_plotlyjs="inline", div_id="embedding-tab"
        )
        stats_html = stats_fig.to_html(include_plotlyjs="inline", div_id="stats-tab")

        # Extract both div and script content from each figure
        import re

        def extract_plotly_content(html_content, div_id):
            """Extract div and complete script tags from Plotly HTML"""
            # Find the div
            div_pattern = rf'<div[^>]*id="{div_id}"[^>]*>.*?</div>'
            div_match = re.search(div_pattern, html_content, re.DOTALL)

            # Find ALL complete script tags
            script_pattern = r"<script[^>]*>.*?</script>"
            script_matches = re.findall(script_pattern, html_content, re.DOTALL)

            # Separate library and plot scripts
            plotly_lib_script = None
            plot_script = None

            for script in script_matches:
                if (
                    "plotly.js v" in script and len(script) > 1000000
                ):  # Library script is very large
                    plotly_lib_script = script  # Keep as complete script tag
                elif (
                    "Plotly.newPlot" in script and div_id in script
                ):  # Plot configuration script
                    plot_script = script  # Keep as complete script tag

            if not div_match:
                logger.warning(f"Failed to extract div for {div_id}")
                return (
                    f'<div id="{div_id}" class="plotly-graph-div" style="height:400px; width:100%;"></div>',
                    plotly_lib_script,
                    plot_script,
                )

            if not plot_script:
                logger.warning(f"Failed to extract plot script for {div_id}")

            return div_match.group(0), plotly_lib_script, plot_script

        # Extract content for each tab
        network_div, network_lib, network_plot = extract_plotly_content(
            network_html, "network-tab"
        )
        embedding_div, embedding_lib, embedding_plot = extract_plotly_content(
            embedding_html, "embedding-tab"
        )
        stats_div, stats_lib, stats_plot = extract_plotly_content(
            stats_html, "stats-tab"
        )

        # Use the first available Plotly library (they're all the same)
        plotly_library = network_lib or embedding_lib or stats_lib or ""

        # Collect all plot scripts
        plot_scripts = []
        for plot_script in [network_plot, embedding_plot, stats_plot]:
            if plot_script:
                plot_scripts.append(plot_script)

        # Combine: library once + all plot scripts
        combined_scripts = ""
        if plotly_library:
            combined_scripts += plotly_library + "\n"
        if plot_scripts:
            combined_scripts += "\n".join(plot_scripts)

        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Knowledge Graph Analysis Dashboard</title>
    <!-- Plotly.js will be included inline with each visualization -->
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        html, body {{
            height: 100vh;
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            overflow: hidden;
        }}
        .main-container {{
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        .tab-container {{
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 0 10px 10px 10px;
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        .tab-buttons {{
            display: flex;
            background-color: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
            flex-shrink: 0;
        }}
        .tab-button {{
            background: none;
            border: none;
            padding: 12px 24px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            color: #495057;
            transition: all 0.3s ease;
        }}
        .tab-button:hover {{
            background-color: #e9ecef;
        }}
        .tab-button.active {{
            background-color: #007bff;
            color: white;
        }}
        .tab-content {{
            display: none;
            flex: 1;
            padding: 0;
        }}
        .tab-content.active {{
            display: flex;
            flex-direction: column;
        }}
        .plotly-graph-div {{
            height: 100% !important;
            width: 100% !important;
        }}
    </style>
</head>
<body>
    <div class="main-container">
        <div class="tab-container">
        <div class="tab-buttons">
            <button class="tab-button active" onclick="showTab('network')">🕸️ Network Graph</button>
            <button class="tab-button" onclick="showTab('embedding')">🎯 Embedding Space</button>
            <button class="tab-button" onclick="showTab('statistics')">📊 Statistics</button>
        </div>
        
        <div id="network" class="tab-content active">
            {network_div}
        </div>
        
        <div id="embedding" class="tab-content">
            {embedding_div}
        </div>
        
        <div id="statistics" class="tab-content">
            {stats_div}
        </div>
    </div>
    
    <!-- Plotly library and visualization scripts -->
    {combined_scripts}
    
    <script>
        function showTab(tabName) {{
            // Hide all tab contents
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));
            
            // Remove active class from all buttons
            const buttons = document.querySelectorAll('.tab-button');
            buttons.forEach(button => button.classList.remove('active'));
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked button
            event.target.classList.add('active');
            
            // Trigger plotly resize for proper rendering
            setTimeout(() => {{
                if (window.Plotly && window.Plotly.Plots) {{
                    window.Plotly.Plots.resize();
                }}
            }}, 100);
        }}
        
        // Ensure plots resize when window resizes
        window.addEventListener('resize', () => {{
            if (window.Plotly && window.Plotly.Plots) {{
                window.Plotly.Plots.resize();
            }}
        }});
        
        // Trigger initial resize after page load
        window.addEventListener('load', () => {{
            setTimeout(() => {{
                if (window.Plotly && window.Plotly.Plots) {{
                    window.Plotly.Plots.resize();
                }}
            }}, 200);
        }});
    </script>
    </div> <!-- close main-container -->
</body>
</html>
        """


def create_visualization_from_files(
    kg_file: str,
    relationship_bank_file: str,
    output_file: str = "kg_visualization.html",
) -> str:
    """Create visualization from saved knowledge graph files"""

    # Load knowledge graph
    with open(kg_file, "r") as f:
        kg_data = json.load(f)

    # Load relationship bank
    with open(relationship_bank_file, "r") as f:
        bank_data = json.load(f)

    # TODO: Reconstruct objects from JSON data and create visualization
    # This would require implementing JSON deserialization for the KG objects

    logger.info(
        f"Would create visualization from {kg_file} and {relationship_bank_file}"
    )
    return output_file
