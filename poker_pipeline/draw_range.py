
# draw_range.py
# Poker range grid rendering

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
from io import BytesIO

def draw_range_from_node(node_dict, dpi=120) -> Image.Image:
    """
    Draw a 13x13 poker range grid based on node["hands"] data.

    Args:
        node_dict: Dictionary with "hands" key containing hand play weights.
        dpi: Dots per inch for the output image (default 120).

    Returns:
        PIL.Image object with rendered poker range grid.
    """
    # Setup grid labels
    ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    grid_labels = []

    # Build grid labels (13x13 matrix of hands)
    for i, rank1 in enumerate(ranks):
        row = []
        for j, rank2 in enumerate(ranks):
            hi = rank1 if ranks.index(rank1) < ranks.index(rank2) else rank2
            lo = rank2 if ranks.index(rank1) < ranks.index(rank2) else rank1

            if rank1 == rank2:
                row.append(f'{hi}{lo}')  # pair
            elif i < j:
                row.append(f'{hi}{lo}s')  # suited
            else:
                row.append(f'{hi}{lo}o')  # offsuit
        grid_labels.append(row)

    # Build action color map
    ACTION_COLORS = {
        0: '#FFFFFF',     # Fold → White
        1: '#FFD700',     # Call / Check → Gold
        2: '#3399FF',     # Raise → Blue (does not blend with pink)
        3: '#FF1493'      # Raise All-in → Deep Pink (high contrast with Raise)
    }

    # Prepare figure
    fig = Figure(figsize=(8, 8), dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    # Make sure full grid is visible
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 13)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw grid
    cell_size = 1.0

    for i in range(13):
        for j in range(13):
            hand_label = grid_labels[i][j]

            # Default → fold (white)
            rounded_played = [1.0, 0.0, 0.0, 0.0]

            if hand_label in node_dict["hands"]:
                played = node_dict["hands"][hand_label]["played"]

                # Round to nearest 10%
                rounded_played = [round(p * 10) / 10 for p in played]

                # Normalize to sum == 1
                total = sum(rounded_played)
                if total > 0:
                    rounded_played = [p / total for p in rounded_played]
                else:
                    rounded_played = [0.0 for _ in played]

            # Draw multi-colored rectangle (VERTICAL bars)
            x_cursor = j  # left of cell
            for k, portion in enumerate(rounded_played):
                if portion == 0:
                    continue
                width = portion * cell_size
                rect = patches.Rectangle(
                    (x_cursor, 12 - i), width, cell_size,
                    linewidth=0, edgecolor=None, facecolor=ACTION_COLORS.get(k, 'white')
                )
                ax.add_patch(rect)
                x_cursor += width

            # Draw hand label (shift text slightly up → better readability)
            ax.text(
                j + cell_size / 2, 12 - i + cell_size / 2 + 0.1,
                hand_label,
                ha='center', va='center',
                fontsize=10, fontweight='bold'
            )

    # Draw full grid lines (14x14), soft gray color
    grid_color = '#666666'  # soft gray

    for x in range(14):
        ax.plot([x, x], [0, 13], color=grid_color, linewidth=0.5, zorder=5)

    for y in range(14):
        ax.plot([0, 13], [y, y], color=grid_color, linewidth=0.5, zorder=5)

    # Draw outer border rectangle (strong black)
    outer_rect = patches.Rectangle(
        (0, 0), 13, 13,
        linewidth=1.5, edgecolor='black', facecolor='none', zorder=10
    )
    ax.add_patch(outer_rect)

    # Finalize
    ax.axis("off")
    canvas.draw()

    # Convert Figure to PIL.Image
    buf = BytesIO()
    canvas.print_png(buf)  # no bbox_inches here
    buf.seek(0)
    img_pil = Image.open(buf)
    img_pil = img_pil.convert("RGB")

    return img_pil
