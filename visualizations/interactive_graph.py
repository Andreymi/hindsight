"""
Interactive HTML graph visualization of memory system.

Uses pyvis to create a smooth, interactive network graph that can be
explored in the browser. Shows all memory units and their links with weights.
"""
import psycopg2
from dotenv import load_dotenv
import os
from pyvis.network import Network
import networkx as nx

load_dotenv()


def create_interactive_graph():
    """Create an interactive HTML graph visualization."""

    # Connect to database
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    cursor = conn.cursor()

    # Get all memory units (no agent_id filter)
    cursor.execute("""
        SELECT id, text, event_date, context
        FROM memory_units
        ORDER BY event_date
    """)
    units = cursor.fetchall()

    # Get all links with weights (no agent_id filter)
    cursor.execute("""
        SELECT
            ml.from_unit_id,
            ml.to_unit_id,
            ml.link_type,
            ml.weight,
            e.canonical_name as entity_name
        FROM memory_links ml
        LEFT JOIN entities e ON ml.entity_id = e.id
        ORDER BY ml.link_type, ml.weight DESC
    """)
    links = cursor.fetchall()

    # Get entity information (no agent_id filter)
    cursor.execute("""
        SELECT ue.unit_id, e.canonical_name, e.entity_type
        FROM unit_entities ue
        JOIN entities e ON ue.entity_id = e.id
        ORDER BY ue.unit_id
    """)
    unit_entities = cursor.fetchall()

    cursor.close()
    conn.close()

    # Build entity mapping
    entity_map = {}
    for unit_id, entity_name, entity_type in unit_entities:
        if unit_id not in entity_map:
            entity_map[unit_id] = []
        entity_map[unit_id].append(f"{entity_name} ({entity_type})")

    # Create pyvis network
    net = Network(
        height="900px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#000000",
        heading="Entity-Aware Memory Graph - Interactive Visualization"
    )

    # Configure physics for smooth layout with performance optimizations
    net.set_options("""
    {
        "nodes": {
            "font": {
                "size": 14,
                "face": "Tahoma"
            },
            "borderWidth": 2,
            "borderWidthSelected": 3
        },
        "edges": {
            "smooth": {
                "enabled": false
            },
            "font": {
                "size": 10,
                "align": "middle"
            }
        },
        "physics": {
            "enabled": true,
            "stabilization": {
                "enabled": true,
                "iterations": 100,
                "updateInterval": 10
            },
            "barnesHut": {
                "gravitationalConstant": -12000,
                "centralGravity": 0.2,
                "springLength": 350,
                "springConstant": 0.02,
                "damping": 0.09,
                "avoidOverlap": 0.8
            },
            "solver": "barnesHut",
            "timestep": 0.5,
            "adaptiveTimestep": true
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true
        }
    }
    """)

    # Add nodes
    for unit_id, text, event_date, context in units:
        # Truncate text for display
        display_text = text[:50] + "..." if len(text) > 50 else text

        # Get entities
        entities = entity_map.get(unit_id, [])
        entity_str = "\\n".join(entities) if entities else "No entities"

        # Build node label and title (hover)
        label = display_text
        title = f"""
        <b>Text:</b> {text}<br>
        <b>Date:</b> {event_date.date()}<br>
        <b>Context:</b> {context}<br>
        <b>Entities:</b> {entity_str}
        """

        # Color by entity count
        if len(entities) == 0:
            color = "#e0e0e0"  # Gray
            size = 20
        elif len(entities) == 1:
            color = "#90caf9"  # Light blue
            size = 25
        else:
            color = "#42a5f5"  # Dark blue
            size = 30

        net.add_node(
            str(unit_id),
            label=label,
            title=title,
            color=color,
            size=size,
            shape="box",
            font={"color": "#000000"}
        )

    # Add edges with colors and weights
    for from_id, to_id, link_type, weight, entity_name in links:
        # Set color and style based on link type
        if link_type == 'temporal':
            color = "#00bcd4"  # Cyan
            dashes = [5, 5]
            width = 0.5
            label = f"T: {weight:.2f}"
        elif link_type == 'semantic':
            color = "#ff69b4"  # Pink
            dashes = False
            width = 0.5
            label = f"S: {weight:.2f}"
        elif link_type == 'entity':
            color = "#ffd700"  # Gold
            dashes = False
            width = 0.8
            label = f"{entity_name}: {weight:.2f}"
        else:
            color = "#999999"
            dashes = False
            width = 0.5
            label = f"{weight:.2f}"

        net.add_edge(
            str(from_id),
            str(to_id),
            value=weight * 1,  # Scale for visual thickness
            color=color,
            dashes=dashes,
            width=width,
            label=label,
            title=f"{link_type.upper()}: {weight:.3f}" + (f" (Entity: {entity_name})" if entity_name else "")
        )

    # Add legend as HTML
    legend_html = """
    <div style="position: absolute; top: 80px; left: 10px; background: white;
                padding: 15px; border: 2px solid #333; border-radius: 8px;
                font-family: Tahoma; box-shadow: 2px 2px 8px rgba(0,0,0,0.3); z-index: 1000;">
        <h3 style="margin-top: 0; border-bottom: 2px solid #333; padding-bottom: 5px;">Legend</h3>

        <h4 style="margin-bottom: 5px;">Link Types:</h4>
        <div style="margin-left: 10px;">
            <div style="margin: 5px 0;">
                <span style="display: inline-block; width: 40px; height: 1px;
                             background: #00bcd4; border-top: 1px dashed #00bcd4;
                             vertical-align: middle;"></span>
                <span style="margin-left: 10px;"><b>Temporal</b> - Time-based (cyan, dashed)</span>
            </div>
            <div style="margin: 5px 0;">
                <span style="display: inline-block; width: 40px; height: 1px;
                             background: #ff69b4; vertical-align: middle;"></span>
                <span style="margin-left: 10px;"><b>Semantic</b> - Meaning-based (pink, solid)</span>
            </div>
            <div style="margin: 5px 0;">
                <span style="display: inline-block; width: 40px; height: 1.5px;
                             background: #ffd700; vertical-align: middle;"></span>
                <span style="margin-left: 10px;"><b>Entity</b> - Same entity (gold)</span>
            </div>
        </div>

        <h4 style="margin-bottom: 5px; margin-top: 15px;">Node Colors:</h4>
        <div style="margin-left: 10px;">
            <div style="margin: 5px 0;">
                <span style="display: inline-block; width: 20px; height: 20px;
                             background: #e0e0e0; border: 1px solid #999;
                             vertical-align: middle;"></span>
                <span style="margin-left: 10px;">Gray - No entities</span>
            </div>
            <div style="margin: 5px 0;">
                <span style="display: inline-block; width: 20px; height: 20px;
                             background: #90caf9; border: 1px solid #999;
                             vertical-align: middle;"></span>
                <span style="margin-left: 10px;">Light Blue - 1 entity</span>
            </div>
            <div style="margin: 5px 0;">
                <span style="display: inline-block; width: 20px; height: 20px;
                             background: #42a5f5; border: 1px solid #999;
                             vertical-align: middle;"></span>
                <span style="margin-left: 10px;">Dark Blue - 2+ entities</span>
            </div>
        </div>

        <div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid #ccc;
                    font-size: 11px; color: #666;">
            <b>Tip:</b> Hover over nodes/edges for details<br>
            <b>Controls:</b> Drag to move, scroll to zoom
        </div>
    </div>
    """

    # Generate the HTML
    output_file = "memory_graph_interactive.html"
    net.save_graph(output_file)

    # Read the generated HTML and inject our legend
    with open(output_file, 'r') as f:
        html_content = f.read()

    # Inject legend after the opening body tag
    html_content = html_content.replace('<body>', '<body>' + legend_html)

    # Add script to disable physics after stabilization for better performance
    physics_script = """
    <script type="text/javascript">
        // Disable physics after initial stabilization for better performance
        network.on("stabilizationIterationsDone", function () {
            network.setOptions({ physics: false });
            console.log("Physics disabled - graph should be much more responsive now!");
        });
    </script>
    """
    html_content = html_content.replace('</body>', physics_script + '</body>')

    # Write back
    with open(output_file, 'w') as f:
        f.write(html_content)

    print(f"\n{'='*80}")
    print("INTERACTIVE GRAPH GENERATED")
    print(f"{'='*80}")
    print(f"\nFile: {output_file}")
    print(f"Units: {len(units)}")
    print(f"Links: {len(links)}")
    print("\nFeatures:")
    print("  • Smooth, physics-based layout")
    print("  • Interactive - drag nodes, zoom, pan")
    print("  • Hover for details on nodes and edges")
    print("  • Color-coded by link type and entity count")
    print("  • Built-in navigation controls")
    print(f"\n{'='*80}")
    print(f"✓ Open {output_file} in your browser to explore!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    create_interactive_graph()
