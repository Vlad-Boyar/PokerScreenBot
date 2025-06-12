
# poker_pipeline/draw_table.py
# Poker table visualization

import matplotlib.patches as patches
import math
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from io import BytesIO
from PIL import Image

def get_player_order(data):
    """ Returns player order for clockwise preflop round """
    players_in_order = list(range(data["num_players"]))
    button = data["button"]

    first_to_act = (button + 1) % data["num_players"]
    player_order = players_in_order[first_to_act:] + players_in_order[:first_to_act]

    return player_order

def draw_table_from_json_pil(data, dpi=120, verbose=False) -> Image.Image:
    """
    Draw a poker table visualization from JSON data.

    Args:
        data: Dictionary with table state (format, stacks, hero_cards, etc.).
        dpi: Dots per inch for the output image (default 120).
        verbose: If True, enables debug visualization (currently unused).

    Returns:
        PIL.Image object with the rendered table.
    """

    """ Returns player order for clockwise preflop round """
    players_in_order = list(range(data["num_players"]))
    button = data["button"]

    first_to_act = (button + 1) % data["num_players"]
    player_order = players_in_order[first_to_act:] + players_in_order[:first_to_act]

    return player_order

def draw_table_from_json_pil(data, dpi=120, verbose=False) -> Image.Image:

    # === Create Figure
    fig = Figure(figsize=(10, 6), dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    num_players = data["num_players"]

    radius_x = 8
    radius_y = 4

    center_x, center_y = 0, 0

    def format_number(x):
        return str(int(x)) if x == int(x) else str(x)

    players_sorted = sorted(data["stacks"], key=lambda p: p["player"])
    positions_all = ['UTG', 'UT', 'EP', 'MP', 'HJ', 'CO', 'BU', 'SB', 'BB']
    positions_labels_full = positions_all[-num_players:]

    hero_idx = next(i for i, p in enumerate(players_sorted) if p["player"] == data["hero"])
    hero_player_num = data["hero"]

    players_rotated = players_sorted[hero_idx:] + players_sorted[:hero_idx]
    positions_labels_rotated = positions_labels_full[hero_idx:] + positions_labels_full[:hero_idx]

    angles_rotated = [math.pi * 1.5 - 2 * math.pi * i / num_players for i in range(num_players)]

    player_radius_scale = 1.25

    positions = [
        (center_x + radius_x * player_radius_scale * math.cos(a),
         center_y + radius_y * player_radius_scale * math.sin(a))
        for a in angles_rotated
    ]

    ax.set_xlim(-radius_x * 1.5, radius_x * 1.5)
    ax.set_ylim(-radius_y * 1.8, radius_y * 1.9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('white')

    ellipse = patches.Ellipse((0, 0), width=radius_x * 2 + 1.5, height=radius_y * 2 + 1.5,
                              facecolor='#3cb371', edgecolor='black', linewidth=3)
    ax.add_patch(ellipse)

    ax.text(-radius_x * 1.4, radius_y + 2.4, f'Format: {data["format"]}', ha='left', fontsize=16, fontweight='bold')

    ante_value = data["ante"] if data["ante"] is not None else 0
    chip_radius_x = 0.3
    chip_radius_y = 0.2
    chip_spacing = 0.2

    if ante_value != 0:
        chip_positions_y = [-chip_spacing, 0, chip_spacing]

        for y_offset in chip_positions_y:
            chip = patches.Ellipse((0, y_offset), width=chip_radius_x * 2, height=chip_radius_y * 2,
                                edgecolor='black', facecolor='skyblue', linewidth=2)
            ax.add_patch(chip)

        ax.text(0.5, 0, f'{format_number(ante_value)}', ha='left', va='center',
                fontsize=14, fontweight='bold', color='black')

    if data["hero_present"] and data["hero_cards"]:
        card_width = 1.3
        card_height = 2.0
        spacing = 0.1
        total_width = 2 * card_width + spacing

        start_x = -total_width / 2 + card_width / 2
        y_pos = -radius_y - 1

        suit_colors = {'s': 'black', 'h': 'red', 'c': 'green', 'd': 'blue'}

        for i, card in enumerate(data["hero_cards"]):
            rank = card[:-1]
            suit = card[-1]
            color = suit_colors.get(suit, 'gray')

            rect_x = start_x + i * (card_width + spacing)

            rect = patches.Rectangle((rect_x - card_width / 2, y_pos - card_height / 2),
                                     card_width, card_height,
                                     linewidth=2, edgecolor='black', facecolor=color)
            ax.add_patch(rect)

            ax.text(rect_x, y_pos, rank.upper(), ha='center', va='center', fontsize=18, fontweight='bold', color='white')

    # === Player loop with EXACT solve_and_format logic

    player_order = get_player_order(data)
    hero_idx_in_order = player_order.index(hero_player_num)

    players_in_game = set()
    current_bet_level = 1.0
    need_another_round = False
    raiser_idx = None
    break_outer_loop = False

    while True:
        for player_idx_in_order, player_idx in enumerate(player_order):
            if player_idx == hero_player_num:
                break_outer_loop = True
                break

            stack_entry = next((s for s in data["stacks"] if s["player"] == player_idx), None)
            player_bet = stack_entry.get("bet", 0)
            player_stack = stack_entry.get("stack", 0)
            player_total_stack = player_stack + player_bet

            if player_total_stack == 0:
                continue

            players_in_game.add(player_idx)

            if player_idx == data["button"]:
                player_start_bet = 0.5
            elif (player_idx + 1) % data["num_players"] == data["button"]:
                player_start_bet = 1.0
            else:
                player_start_bet = 0.0

            if player_bet > current_bet_level:
                current_bet_level = player_bet
                need_another_round = True
                raiser_idx = player_idx

            if need_another_round and player_idx == raiser_idx:
                need_another_round = False

        if break_outer_loop:
            break

    y_offset_hero_box = -1.3

    button_num = data["button"]
    button_pos = None

    for idx, (player_data, pos) in enumerate(zip(players_rotated, positions)):
        x, y = pos
        player_num = player_data["player"]

        player_idx_in_order = player_order.index(player_num)

        player_label = positions_labels_rotated[idx]

        is_hero = (hero_player_num == player_num)
        is_button = (button_num == player_num)

        if is_hero:
            player_label += ' (H)'

        face_color = 'lightyellow'
        if is_hero:
            face_color = 'lightblue'

        if player_data["stack"] == 0:
            stack_info = 'All-In'
            stack_color = 'red'
            stack_fontweight = 'bold'
        else:
            stack_info = f'{format_number(player_data["stack"])} BB'
            stack_color = 'black'
            stack_fontweight = 'bold'

        y_draw = y + (y_offset_hero_box if is_hero else 0)

        box_width = 3.5
        box_height = 1.3

        rect = patches.Rectangle((x - box_width / 2, y_draw - box_height / 2),
                                box_width, box_height,
                                linewidth=2, edgecolor='black', facecolor=face_color)
        ax.add_patch(rect)

        ax.text(x, y_draw + 0.25, player_label, ha='center', va='center',
                fontsize=12, fontweight='bold', color='black')

        ax.text(x, y_draw - 0.25, stack_info, ha='center', va='center',
                fontsize=12, fontweight=stack_fontweight, color=stack_color)

        # Logic for drawing player cards based on player_order logic
        if player_idx_in_order > hero_idx_in_order:
            draw_player_cards = True
        elif player_idx_in_order < hero_idx_in_order:
            draw_player_cards = (player_num in players_in_game)
        else:
            draw_player_cards = False  # Hero

        if draw_player_cards:
            card_width = 1.1
            card_height = 1.6
            spacing = 0.1
            card_y = y_draw + box_height / 2 + card_height / 2

            left_card_x = x - (card_width / 2 + spacing / 2)
            right_card_x = x + (card_width / 2 + spacing / 2)

            for card_x in [left_card_x, right_card_x]:
                rect_card = patches.Rectangle((card_x - card_width / 2, card_y - card_height / 2),
                                            card_width, card_height,
                                            linewidth=2, edgecolor='black', facecolor='lightcoral')
                ax.add_patch(rect_card)

        # === Player bet
        if player_data["bet"] > 0:
            bet_value = player_data["bet"]

            bet_offset = 0.6
            bet_x = center_x + (x - center_x) * bet_offset
            bet_y = center_y + (y - center_y) * bet_offset

            chip_positions_y = [-chip_spacing, 0, chip_spacing]

            for y_offset in chip_positions_y:
                chip = patches.Ellipse((bet_x, bet_y + y_offset), width=chip_radius_x * 2, height=chip_radius_y * 2,
                                       edgecolor='black', facecolor='#ff6666', linewidth=2)
                ax.add_patch(chip)

            ax.text(bet_x + 0.5, bet_y, f'{format_number(bet_value)}', ha='left', va='center',
                    fontsize=14, fontweight='bold', color='black')

        # Draw dealer button (D)
        if is_button:
            circle_radius = 0.4
            button_offset = 0.7
            angle_to_player = angles_rotated[idx]

            angle_shift = math.radians(5)
            button_angle = angle_to_player - angle_shift

            button_radius_x = (x - center_x) * button_offset
            button_radius_y = (y - center_y) * button_offset

            button_x = center_x + math.cos(button_angle) * math.hypot(button_radius_x, button_radius_y)
            button_y = center_y + math.sin(button_angle) * math.hypot(button_radius_x, button_radius_y)

            button_pos = (button_x, button_y, circle_radius)

    if button_pos is not None:
        button_x, button_y, circle_radius = button_pos

        circle = patches.Circle((button_x, button_y),
                                radius=circle_radius, edgecolor='black', facecolor='yellow', linewidth=2)
        ax.add_patch(circle)

        ax.text(button_x, button_y, 'D', ha='center', va='center',
                fontsize=12, fontweight='bold', color='black')

    # === Finalize Figure
    ax.axis("off")
    canvas.draw()

    buf = BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    img_pil = Image.open(buf)
    img_pil = img_pil.convert("RGB")

    return img_pil