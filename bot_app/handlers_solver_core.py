
# bot_app/handlers_solver_core.py

import logging
import os
from poker_pipeline.solver_utils import convert_hero_cards_to_key, load_node, pick_action_in_node
from bot_app.utils import compute_effective_stack, pick_closest_nodes_folder  # ideally move to utils.py

SOLVER_DB_PATH = "./solver_db"

def get_player_group_and_nodes_folder(format_str, num_players, effective_stack_bb):
    if format_str != "spin":
        return None, f"⚠️ Format {format_str.upper()} not supported."
    if num_players == 2:
        player_group = "HU"
    elif num_players == 3:
        player_group = "3max"
    else:
        return None, f"⚠️ Unsupported number of players: {num_players}"

    base_folder = os.path.join(SOLVER_DB_PATH, format_str, player_group)
    nodes_folder = pick_closest_nodes_folder(base_folder, effective_stack_bb)
    return nodes_folder, None

def solve_and_format(last_final_json):
    """
    Simulates preflop hand based on current JSON state.
    Emulates player actions through the tree and returns recommended action probabilities for Hero.
    """
    if not last_final_json or last_final_json["format"] != "spin":
        return "⚠️ We do not support this format yet.", None

    num_players = last_final_json["num_players"]
    effective_stack_bb = compute_effective_stack(last_final_json)

    nodes_folder, err_msg = get_player_group_and_nodes_folder(last_final_json["format"], num_players, effective_stack_bb)
    if err_msg:
        return err_msg, None

    logging.info(f"📂 Selected nodes folder: {os.path.basename(nodes_folder)} (target {effective_stack_bb})")

    current_node = load_node(nodes_folder, 0)
    players_in_order = list(range(last_final_json["num_players"]))
    hero_player_num = last_final_json["hero"]
    current_bet_level = 1.0
    need_another_round = False
    raiser_idx = None
    break_outer_loop = False
    hero_bet = 0

    while True:
        for player_idx in players_in_order:
            if player_idx == hero_player_num:
                hero_entry = next((s for s in last_final_json["stacks"] if s["player"] == hero_player_num), None)
                hero_bet = hero_entry.get("bet", 0)

                if hero_player_num == last_final_json["button"]:
                    hero_start_bet = 0.0
                elif hero_player_num == (last_final_json["button"] + 1) % last_final_json["num_players"]:
                    hero_start_bet = 0.5
                elif hero_player_num == (last_final_json["button"] + 2) % last_final_json["num_players"]:
                    hero_start_bet = 1.0
                else:
                    raise ValueError(f"❌ Unexpected hero position {hero_player_num} vs button {last_final_json['button']}")

                if not need_another_round:
                    if abs(hero_bet - hero_start_bet) < 1e-6:
                        break_outer_loop = True
                        break
                else:
                    if hero_bet < current_bet_level - 1e-6:
                        break_outer_loop = True
                        break

            if need_another_round and player_idx == raiser_idx:
                need_another_round = False
                continue

            stack_entry = next((s for s in last_final_json["stacks"] if s["player"] == player_idx), None)
            player_bet = stack_entry.get("bet", 0)
            player_stack = stack_entry.get("stack", 0)
            player_total_stack = player_stack + player_bet

            if player_idx == last_final_json["button"]:
                player_start_bet = 0.5
            elif (player_idx + 1) % last_final_json["num_players"] == last_final_json["button"]:
                player_start_bet = 1.0
            else:
                player_start_bet = 0.0

            if player_total_stack == 0:
                fake_action_type = 'C'
                fake_action_amount = player_bet * 10000
            elif player_bet == player_start_bet and current_bet_level == player_start_bet:
                fake_action_type = 'F'
                fake_action_amount = 0
            elif player_bet == current_bet_level:
                fake_action_type = 'C'
                fake_action_amount = player_bet * 10000
            elif player_bet > current_bet_level:
                fake_action_type = 'R'
                fake_action_amount = player_bet * 10000
                current_bet_level = player_bet
                need_another_round = True
                raiser_idx = player_idx
            else:
                fake_action_type = 'F'
                fake_action_amount = 0

            if fake_action_type == 'R':
                r_actions = [a for a in current_node["actions"] if a["type"] == 'R']
                if r_actions:
                    r_action = min(r_actions, key=lambda a: abs(a["amount"] - fake_action_amount))
                    action = r_action
                else:
                    c_action = next((a for a in current_node["actions"] if a["type"] == 'C'), None)
                    if c_action:
                        action = c_action
                    else:
                        action = next(a for a in current_node["actions"] if a["type"] == 'F')
            elif fake_action_type == 'F':
                action = next(a for a in current_node["actions"] if a["type"] == 'F')
            elif fake_action_type == 'C':
                action = next(a for a in current_node["actions"] if a["type"] == 'C')
            else:
                raise ValueError(f"❌ Unknown emulated action type: {fake_action_type}")

            next_node_id = action["node"]
            current_node = load_node(nodes_folder, next_node_id)

        if break_outer_loop:
            break

    hero_entry = next(s for s in last_final_json["stacks"] if s["player"] == hero_player_num)
    hero_stack_bb = hero_entry["stack"] + hero_entry.get("bet", 0)

    reply_text = ""

    action_labels = [a["type"] for a in current_node["actions"]]
    weights = current_node["hands"][convert_hero_cards_to_key(last_final_json["hero_cards"])]["played"]

    for action, weight, a in zip(action_labels, weights, current_node["actions"]):
        weight_percent = round(weight * 100, 1)
        if weight_percent < 1.0:
            continue

        if action == 'F':
            reply_text += f"Fold: {weight_percent}%\n"
        elif action == 'C':
            reply_text += f"Call: {weight_percent}%\n"
        elif action == 'X':
            reply_text += f"Check: {weight_percent}%\n"
        elif action == 'R':
            amount_bb = a["amount"] / 10000
            if amount_bb >= 0.5 * hero_stack_bb:
                reply_text += f"All-In: {weight_percent}%\n"
            else:
                reply_text += f"Raise: {weight_percent}%\n"

    if reply_text.strip() == "":
        reply_text = "No significant actions (>1%)"

    return reply_text, current_node

def solve_traverse_and_format(json_input):
    """
    Returns full range summary based on the input JSON.
    """
    from poker_pipeline.rag_solver import traverse_tree
    if not json_input:
        return "⚠️ No data."

    num_players = json_input["num_players"]
    effective_stack_bb = compute_effective_stack(json_input)

    nodes_folder, err_msg = get_player_group_and_nodes_folder(json_input["format"], num_players, effective_stack_bb)
    if err_msg:
        return err_msg

    result = traverse_tree(json_input, nodes_folder)
    reply_text = "🎯 Range:"
    for action, weight in result.items():
        reply_text += f"- {action}: {weight}%\n"
    return reply_text
