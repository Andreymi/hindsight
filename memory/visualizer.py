"""
Memory visualization module.

Provides visual representations of memory networks and search paths.
"""
import time
from typing import List, Dict, Any, Optional, Tuple
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box


class MemoryVisualizer:
    """
    Visualizes memory networks and search paths.
    """

    def __init__(self):
        """Initialize the visualizer."""
        self.console = Console()

    def visualize_memory_graph(
        self,
        units: List[Dict[str, Any]],
        links: List[Dict[str, Any]],
        output_file: str = "memory_graph.png",
        highlight_nodes: Optional[List[str]] = None,
    ):
        """
        Create a visual representation of the memory graph.

        Args:
            units: List of memory units (id, text, context, etc.)
            links: List of links (from_unit_id, to_unit_id, link_type, weight)
            output_file: Output file path for the visualization
            highlight_nodes: Optional list of node IDs to highlight
        """
        # Create directed graph
        G = nx.DiGraph()

        # Add nodes
        node_labels = {}
        for unit in units:
            unit_id = str(unit['id'])
            # Truncate text for display
            label = unit['text'][:40] + "..." if len(unit['text']) > 40 else unit['text']
            G.add_node(unit_id)
            node_labels[unit_id] = label

        # Add edges
        temporal_edges = []
        semantic_edges = []
        for link in links:
            from_id = str(link['from_unit_id'])
            to_id = str(link['to_unit_id'])
            weight = link['weight']
            link_type = link['link_type']

            if link_type == 'temporal':
                temporal_edges.append((from_id, to_id, weight))
            else:  # semantic
                semantic_edges.append((from_id, to_id, weight))

            G.add_edge(from_id, to_id, weight=weight, type=link_type)

        # Create figure
        fig, ax = plt.subplots(figsize=(20, 14))
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#0f0f1e')

        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Draw temporal edges (blue)
        if temporal_edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(e[0], e[1]) for e in temporal_edges],
                edge_color='#4ecdc4',
                alpha=0.6,
                width=2,
                arrows=True,
                arrowsize=15,
                arrowstyle='->',
                connectionstyle='arc3,rad=0.1',
                ax=ax
            )

        # Draw semantic edges (purple)
        if semantic_edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(e[0], e[1]) for e in semantic_edges],
                edge_color='#ff6b9d',
                alpha=0.6,
                width=2,
                arrows=True,
                arrowsize=15,
                arrowstyle='->',
                connectionstyle='arc3,rad=0.1',
                ax=ax
            )

        # Determine node colors
        node_colors = []
        for node in G.nodes():
            if highlight_nodes and node in highlight_nodes:
                node_colors.append('#ffd93d')  # Yellow for highlighted
            else:
                node_colors.append('#6c63ff')  # Purple for normal

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=3000,
            alpha=0.9,
            ax=ax
        )

        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            node_labels,
            font_size=8,
            font_color='white',
            font_weight='bold',
            ax=ax
        )

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='#4ecdc4', lw=2, label='Temporal Links'),
            plt.Line2D([0], [0], color='#ff6b9d', lw=2, label='Semantic Links'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#6c63ff',
                      markersize=10, label='Memory Unit', linestyle=''),
        ]
        if highlight_nodes:
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffd93d',
                          markersize=10, label='Highlighted', linestyle='')
            )

        ax.legend(handles=legend_elements, loc='upper left', facecolor='#2d2d44',
                 edgecolor='white', fontsize=10, labelcolor='white')

        # Title
        ax.set_title('Memory Network Graph\nTemporal + Semantic Architecture',
                    color='white', fontsize=16, fontweight='bold', pad=20)

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, facecolor='#0f0f1e')
        plt.close()

        self.console.print(f"[green]✓[/green] Memory graph saved to [cyan]{output_file}[/cyan]")


class LiveSearchTracer:
    """
    Live tracer for search operations showing spreading activation in real-time.
    """

    def __init__(self):
        """Initialize the live tracer."""
        self.console = Console()
        self.visited_nodes = []
        self.current_node = None
        self.search_results = []
        self.query = ""
        self.budget_used = 0
        self.budget_total = 0

    def start_search(self, query: str, budget: int):
        """
        Start a new search trace.

        Args:
            query: Search query
            budget: Thinking budget
        """
        self.query = query
        self.budget_total = budget
        self.budget_used = 0
        self.visited_nodes = []
        self.current_node = None
        self.search_results = []

    def visit_node(
        self,
        node_id: str,
        text: str,
        activation: float,
        recency: float,
        frequency: float,
        weight: float,
        is_entry_point: bool = False,
    ):
        """
        Record a node visit.

        Args:
            node_id: Node ID
            text: Node text
            activation: Activation strength
            recency: Recency weight
            frequency: Frequency weight
            weight: Combined weight
            is_entry_point: Whether this is an entry point
        """
        self.current_node = {
            'id': node_id,
            'text': text,
            'activation': activation,
            'recency': recency,
            'frequency': frequency,
            'weight': weight,
            'is_entry_point': is_entry_point,
        }
        self.visited_nodes.append(self.current_node)
        self.budget_used += 1

    def add_result(
        self,
        text: str,
        weight: float,
        activation: float,
        recency: float,
        frequency: float,
    ):
        """
        Add a search result.

        Args:
            text: Result text
            weight: Combined weight
            activation: Activation strength
            recency: Recency weight
            frequency: Frequency weight
        """
        self.search_results.append({
            'text': text,
            'weight': weight,
            'activation': activation,
            'recency': recency,
            'frequency': frequency,
        })

    def render_live(self) -> Layout:
        """
        Render the current state.

        Returns:
            Rich Layout with current state
        """
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=5)
        )

        # Header
        header_text = Text()
        header_text.append("🔍 ", style="bold cyan")
        header_text.append(f"Query: ", style="bold white")
        header_text.append(f"{self.query}", style="bold yellow")
        layout["header"].update(Panel(header_text, style="cyan"))

        # Body - split into current node and visited
        layout["body"].split_row(
            Layout(name="current", ratio=1),
            Layout(name="path", ratio=1),
        )

        # Current node
        if self.current_node:
            current_table = Table(
                title="Current Node",
                show_header=False,
                box=box.ROUNDED,
                style="green"
            )
            current_table.add_column("Key", style="cyan")
            current_table.add_column("Value", style="white")

            status = "🎯 ENTRY POINT" if self.current_node['is_entry_point'] else "🔄 EXPLORING"
            current_table.add_row("Status", status)
            current_table.add_row("Text", self.current_node['text'][:50] + "...")
            current_table.add_row(
                "Weights",
                f"A:{self.current_node['activation']:.2f} "
                f"R:{self.current_node['recency']:.2f} "
                f"F:{self.current_node['frequency']:.2f}"
            )
            current_table.add_row(
                "Combined",
                f"[bold yellow]{self.current_node['weight']:.3f}[/bold yellow]"
            )

            layout["current"].update(Panel(current_table, border_style="green"))
        else:
            layout["current"].update(Panel("Initializing...", border_style="dim"))

        # Visited path
        path_table = Table(
            title=f"Visited Nodes ({len(self.visited_nodes)})",
            box=box.SIMPLE,
            show_header=True,
            style="blue"
        )
        path_table.add_column("#", style="dim", width=4)
        path_table.add_column("Text", style="white", width=35)
        path_table.add_column("Weight", justify="right", style="yellow", width=8)
        path_table.add_column("Type", style="cyan", width=8)

        for i, node in enumerate(reversed(self.visited_nodes[-10:])):  # Last 10
            node_type = "ENTRY" if node['is_entry_point'] else "SPREAD"
            path_table.add_row(
                str(len(self.visited_nodes) - i),
                node['text'][:32] + "...",
                f"{node['weight']:.3f}",
                node_type
            )

        layout["path"].update(Panel(path_table, border_style="blue"))

        # Footer - progress bar
        progress = self.budget_used / self.budget_total if self.budget_total > 0 else 0
        bar_width = 50
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)

        footer_text = Text()
        footer_text.append(f"Progress: ", style="bold white")
        footer_text.append(bar, style="yellow")
        footer_text.append(f" {self.budget_used}/{self.budget_total}", style="bold cyan")
        footer_text.append(f" ({progress*100:.1f}%)", style="dim")

        layout["footer"].update(Panel(footer_text, style="yellow"))

        return layout

    def show_final_results(self):
        """
        Show final search results in a nice table.
        """
        self.console.print("\n")
        results_table = Table(
            title="🎯 Search Results",
            show_header=True,
            header_style="bold magenta",
            box=box.DOUBLE_EDGE,
            title_style="bold white"
        )

        results_table.add_column("Rank", style="cyan", justify="center", width=6)
        results_table.add_column("Text", style="white", width=50)
        results_table.add_column("Weight", justify="right", style="yellow", width=8)
        results_table.add_column("A", justify="right", style="green", width=6)
        results_table.add_column("R", justify="right", style="blue", width=6)
        results_table.add_column("F", justify="right", style="magenta", width=6)

        for i, result in enumerate(self.search_results, 1):
            rank_style = "bold yellow" if i <= 3 else "cyan"
            results_table.add_row(
                f"#{i}",
                result['text'][:47] + "...",
                f"{result['weight']:.3f}",
                f"{result['activation']:.2f}",
                f"{result['recency']:.2f}",
                f"{result['frequency']:.2f}",
                style=rank_style if i <= 3 else None
            )

        self.console.print(results_table)

        # Summary stats
        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="bold cyan")
        summary.add_column(style="white")

        summary.add_row("Total nodes visited:", f"{len(self.visited_nodes)}")
        summary.add_row("Budget used:", f"{self.budget_used}/{self.budget_total}")
        summary.add_row("Results found:", f"{len(self.search_results)}")

        self.console.print(Panel(summary, title="Summary", border_style="green", padding=(1, 2)))
