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
        """Create complete interactive visualization"""

        logger.info("Creating knowledge graph visualizations...")

        # Collect all data
        viz_data = self._collect_visualization_data(kg, schema_bank, trigger_embeddings)

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "N-ary Knowledge Graph",
                "Embedding Space (t-SNE)",
                "Entity Types",
                "Relationship Categories",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}],
            ],
        )

        # Add hyperedge network visualization
        self._add_hyperedge_network(fig, viz_data, row=1, col=1)

        # Add embedding space visualization
        self._add_embedding_space(fig, viz_data, row=1, col=2)

        # Add statistics
        self._add_entity_statistics(fig, viz_data, row=2, col=1)
        self._add_relationship_statistics(fig, viz_data, row=2, col=2)

        # Update layout
        fig.update_layout(
            title="Knowledge Graph Analysis Dashboard",
            height=800,
            showlegend=True,
            hovermode="closest",
        )

        # Hide axes for network graph (row=1, col=1) - it should look like a network, not a coordinate system
        fig.update_xaxes(
            showgrid=False, zeroline=False, showticklabels=False, title="", row=1, col=1
        )
        fig.update_yaxes(
            showgrid=False, zeroline=False, showticklabels=False, title="", row=1, col=1
        )

        # Keep axes for embedding space (row=1, col=2) - this IS a coordinate system
        fig.update_xaxes(title="t-SNE Dimension 1", row=1, col=2)
        fig.update_yaxes(title="t-SNE Dimension 2", row=1, col=2)

        # Save to HTML
        fig.write_html(self.output_path)
        logger.info(f"Visualization saved to {self.output_path}")

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

    def _add_hyperedge_network(
        self, fig, viz_data: KGVisualizationData, row: int, col: int
    ):
        """Add hyperedge network visualization for n-ary relationships"""

        # Create node positions using spring layout approximation
        entity_positions = self._calculate_network_positions(viz_data)

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
            if entity.id in entity_positions:
                pos = entity_positions[entity.id]
                entity_x.append(pos[0])
                entity_y.append(pos[1])
                entity_text.append(f"{entity.name}<br>{entity.node_type.value}")
                entity_colors.append(color_map.get(entity.node_type.value, "#DDA0DD"))
                # Size by number of relationships
                entity_sizes.append(
                    10
                    + len(
                        [
                            r
                            for r in viz_data.relationships
                            if entity.id in r.participants.values()
                        ]
                    )
                    * 3
                )

        fig.add_trace(
            go.Scatter(
                x=entity_x,
                y=entity_y,
                mode="markers+text",
                text=[t.split("<br>")[0] for t in entity_text],  # Just names
                textposition="middle center",
                hovertext=entity_text,
                marker=dict(
                    size=entity_sizes,
                    color=entity_colors,
                    line=dict(width=2, color="white"),
                    opacity=0.8,
                ),
                name="Entities",
                showlegend=True,
            ),
            row=row,
            col=col,
        )

        # Add relationship hyperedge nodes
        rel_x = []
        rel_y = []
        rel_text = []
        rel_colors = []

        rel_color_map = {
            "action": "#FF7F7F",
            "state": "#7FFF7F",
            "transfer": "#7F7FFF",
            "comparative": "#FFFF7F",
            "causal": "#FF7FFF",
        }

        # Add edges from entities to relationship nodes
        edge_x = []
        edge_y = []

        for i, rel in enumerate(viz_data.relationships):
            # Position relationship node in center of its participants
            participant_positions = [
                entity_positions[pid]
                for pid in rel.participants.values()
                if pid in entity_positions
            ]
            if participant_positions:
                rel_pos = np.mean(participant_positions, axis=0)
                rel_x.append(rel_pos[0])
                rel_y.append(rel_pos[1])

                # Create hover text with relationship details
                roles_text = ", ".join(
                    [
                        f"{role}={pid.split('_')[-1]}"
                        for role, pid in rel.participants.items()
                    ]
                )
                rel_text.append(f"{rel.relationship_type}<br>{roles_text}")

                category = rel.properties.get("category", "action")
                rel_colors.append(rel_color_map.get(category, "#DDA0DD"))

                # Add edges from relationship to each participant
                for role, participant_id in rel.participants.items():
                    if participant_id in entity_positions:
                        part_pos = entity_positions[participant_id]
                        # Add edge line
                        edge_x.extend([rel_pos[0], part_pos[0], None])
                        edge_y.extend([rel_pos[1], part_pos[1], None])

        # Add relationship nodes
        fig.add_trace(
            go.Scatter(
                x=rel_x,
                y=rel_y,
                mode="markers",
                hovertext=rel_text,
                marker=dict(
                    size=8,
                    color=rel_colors,
                    symbol="diamond",
                    line=dict(width=1, color="black"),
                    opacity=0.9,
                ),
                name="Relationships",
                showlegend=True,
            ),
            row=row,
            col=col,
        )

        # Add edges
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=1, color="gray"),
                hoverinfo="none",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    def _calculate_network_positions(
        self, viz_data: KGVisualizationData
    ) -> Dict[str, np.ndarray]:
        """Calculate 2D positions for network layout using simple force simulation"""

        entities = {e.id: e for e in viz_data.entities}
        positions = {}

        # Initialize random positions
        np.random.seed(42)
        for entity_id in entities:
            positions[entity_id] = np.random.rand(2) * 10

        # Simple spring layout simulation
        for iteration in range(100):
            forces = {entity_id: np.zeros(2) for entity_id in entities}

            # Repulsion between all nodes
            for id1 in entities:
                for id2 in entities:
                    if id1 != id2:
                        diff = positions[id1] - positions[id2]
                        dist = np.linalg.norm(diff)
                        if dist > 0:
                            forces[id1] += diff / (dist**2) * 0.1

            # Attraction between connected nodes
            for rel in viz_data.relationships:
                participants = list(rel.participants.values())
                for i in range(len(participants)):
                    for j in range(i + 1, len(participants)):
                        if (
                            participants[i] in positions
                            and participants[j] in positions
                        ):
                            diff = (
                                positions[participants[j]] - positions[participants[i]]
                            )
                            dist = np.linalg.norm(diff)
                            if dist > 0:
                                force = diff * 0.01
                                forces[participants[i]] += force
                                forces[participants[j]] -= force

            # Update positions
            for entity_id in entities:
                positions[entity_id] += forces[entity_id] * 0.1

        return positions

    def _add_embedding_space(
        self, fig, viz_data: KGVisualizationData, row: int, col: int
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

            fig.add_trace(
                go.Scatter(
                    x=embedding_2d[indices, 0],
                    y=embedding_2d[indices, 1],
                    mode="markers",
                    text=[viz_data.embedding_points[i].text for i in indices],
                    hovertext=[
                        f"{viz_data.embedding_points[i].type}: {viz_data.embedding_points[i].text}"
                        for i in indices
                    ],
                    marker=dict(size=8, color=type_colors[point_type], opacity=0.7),
                    name=f"{point_type.title()}s",
                    showlegend=True,
                ),
                row=row,
                col=col,
            )

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
