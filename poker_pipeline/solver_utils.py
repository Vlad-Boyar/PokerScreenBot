
"""
Solver utilities.

Provides functions to traverse preflop decision trees and extract recommended ranges.
"""

import json
import os
import numpy as np

def load_node(nodes_folder, node_id):
    """
    Load a solver node from the given folder.

    Args:
        nodes_folder: Path to the folder containing node JSON files.
        node_id: ID of the node to load.

    Returns:
        Dictionary with node data.
    """
    with open(os.path.join(nodes_folder, f"{node_id}.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def pick_action_in_node(node, action_type, action_amount=None):
    """
    Pick an action of a given type in the node.

    Args:
        node: Dictionary with node data.
        action_type: Action type ('F', 'C', 'R').
        action_amount: Optional amount for Raise action.

    Returns:
        Action dictionary.
    """
    if action_type == 'R':
        r_actions = [a for a in node["actions"] if a["type"] == 'R']
        if r_actions:
            if action_amount is None:
                return r_actions[0]
            r_action = min(r_actions, key=lambda a: abs(a["amount"] - action_amount))
            return r_action
        else:
            c_action = next((a for a in node["actions"] if a["type"] == 'C'), None)
            if c_action:
                return c_action
            else:
                return next(a for a in node["actions"] if a["type"] == 'F')

    elif action_type == 'F':
        return next(a for a in node["actions"] if a["type"] == 'F')
    elif action_type == 'C':
        return next(a for a in node["actions"] if a["type"] == 'C')
    else:
        raise ValueError(f"Unknown action type: {action_type}")

def convert_hero_cards_to_key(hero_cards):
    """
    Convert hero's cards to a key string used in node["hands"].

    Args:
        hero_cards: List of 2 cards as strings (e.g. ['Ac', 'Ks']).

    Returns:
        String key (e.g. 'AKo', 'AKs', 'QQ').
    """
    ranks_order = "23456789TJQKA"
    rank1 = hero_cards[0][0].replace("T", "10")
    rank2 = hero_cards[1][0].replace("T", "10")

    r1 = hero_cards[0][0] if hero_cards[0][0] != "T" else "T"
    r2 = hero_cards[1][0] if hero_cards[1][0] != "T" else "T"

    s1 = hero_cards[0][1]
    s2 = hero_cards[1][1]

    if r1 == r2:
        return f"{r1}{r2}"
    else:
        ranks_sorted = sorted([r1, r2], key=lambda x: ranks_order.index(x), reverse=True)
        suited = s1 == s2
        return f"{ranks_sorted[0]}{ranks_sorted[1]}{'s' if suited else 'o'}"

def traverse_tree(json_input, nodes_folder):
    """
    Traverse the solver tree according to current game state.

    Args:
        json_input: Dictionary with current game state.
        nodes_folder: Path to solver nodes folder.

    Returns:
        Dictionary mapping action labels to their weights in %.
    """
    num_players = json_input["num_players"]
    button_num = json_input["button"]
    hero_player_num = json_input["hero"]

    current_node = load_node(nodes_folder, 0)

    players_sorted = sorted(json_input["stacks"], key=lambda p: p["player"])
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

    hero_reached = False
    sb_skipped_player_num = None

    for player_num in player_order_reordered:
        if player_num == hero_player_num:
            hero_reached = True
            continue

        player_data = next(p for p in players_sorted if p["player"] == player_num)
        player_bet = player_data.get("bet", 0)
        player_stack = player_data["stack"]
        player_total_stack = player_stack + player_bet

        if not hero_reached:
            if player_bet > 0:
                if (is_special_sb_skip and player_num == special_sb_player_num and player_bet == 0.5):
                    sb_skipped_player_num = player_num
                    continue
                else:
                    fake_action_type = 'C' if player_bet == 1.0 else 'R'
                    fake_action_amount = player_bet * 10000 if fake_action_type == 'R' else None

                    action = pick_action_in_node(current_node, fake_action_type, fake_action_amount)
                    next_node_id = action["node"]
                    current_node = load_node(nodes_folder, next_node_id)

    hero_cards_key = convert_hero_cards_to_key(json_input["hero_cards"])
    hero_hand_entry = current_node["hands"].get(hero_cards_key)

    if hero_hand_entry is None:
        raise ValueError(f"Hero hand {hero_cards_key} not found in node hands.")

    played_weights = hero_hand_entry["played"]
    action_labels = [a["type"] for a in current_node["actions"]]

    result = {}
    for action, weight in zip(action_labels, played_weights):
        weight_percent = round(weight * 100, 1)
        result[action] = weight_percent

    return result
