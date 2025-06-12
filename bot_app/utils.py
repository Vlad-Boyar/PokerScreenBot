
# bot_app/utils.py

import logging
import os

def compute_effective_stack(json_input):
    players_sorted = sorted(json_input["stacks"], key=lambda p: p["player"])
    num_players = json_input["num_players"]
    hero_player_num = json_input["hero"]
    button_num = json_input["button"]

    hero_entry = next(s for s in json_input["stacks"] if s["player"] == hero_player_num)
    hero_stack_bb = hero_entry["stack"] + hero_entry.get("bet", 0)

    player_order = [p["player"] for p in players_sorted]
    hero_idx_in_order = player_order.index(hero_player_num)

    player_order_reordered = player_order[hero_idx_in_order+1:] + player_order[:hero_idx_in_order+1]

    hero_expected_pos = (button_num + 2) % num_players
    if hero_expected_pos == 0:
        hero_expected_pos = num_players

    is_special_sb_skip = (hero_player_num == hero_expected_pos)

    if is_special_sb_skip:
        special_sb_player_num = (button_num + 1) % num_players
        if special_sb_player_num == 0:
            special_sb_player_num = num_players
    else:
        special_sb_player_num = None

    active_opponent_stacks = []
    hero_reached = False
    sb_skipped_player_num = None

    for player_num in player_order_reordered:
        if player_num == hero_player_num:
            hero_reached = True
            continue

        player_data = next(p for p in players_sorted if p["player"] == player_num)

        if not hero_reached:
            if player_data["bet"] > 0:
                if (is_special_sb_skip and player_num == special_sb_player_num and player_data["bet"] == 0.5):
                    sb_skipped_player_num = player_num
                    continue
                else:
                    stack_bb = player_data["stack"] + player_data.get("bet", 0)
                    active_opponent_stacks.append(stack_bb)
        else:
            if player_num != sb_skipped_player_num:
                stack_bb = player_data["stack"] + player_data.get("bet", 0)
                active_opponent_stacks.append(stack_bb)

    if not active_opponent_stacks:
        raise ValueError("No active opponents found!")

    avg_opponents_stack_bb = sum(active_opponent_stacks) / len(active_opponent_stacks)

    logging.info(f"🧮 Hero stack: {hero_stack_bb} BB, Avg opponents stack: {avg_opponents_stack_bb:.2f} BB")

    effective_stack_bb = min(hero_stack_bb, avg_opponents_stack_bb)
    return round(effective_stack_bb)

def pick_closest_nodes_folder(base_folder, target_bb):
    available_folders = [
        name for name in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, name)) and name.isdigit()
    ]

    if not available_folders:
        raise ValueError(f"No BB folders found in {base_folder}!")

    available_bbs = sorted([int(name) for name in available_folders])

    closest_bb = min(available_bbs, key=lambda bb: abs(bb - target_bb))

    logging.info(f"📂 Selected nodes folder: {closest_bb} (target {target_bb})")

    return os.path.join(base_folder, str(closest_bb), "nodes")
