from dota2_predictor.data_service.build_db import get_saved_matches
import numpy as np
import dota2_predictor.data_service.consts as consts
import torch
from sklearn.preprocessing import StandardScaler



role_map = {
    "Carry": 0,
    "Support": 1,
    "Nuker": 2,
    "Disabler": 3,
    "Durable": 4,
    "Escape": 5,
    "Pusher": 6,
    "Initiator": 7,
}



attributes = ["str","agi","int"]
attack_type = ["Melee","Ranged"]


stats_keys = [
    "base_health",
    "base_health_regen",
    "base_mana",
    "base_mana_regen",
    "base_armor",
    "base_mr",
    "base_attack_min",
    "base_attack_max",
    "base_str",
    "base_agi",
    "base_int",
    "str_gain",
    "agi_gain",
    "int_gain",
    "attack_range",
    "projectile_speed",
    "attack_rate",
    "base_attack_time",
    "attack_point",
    "move_speed",
    "day_vision",
    "night_vision",
]


def get_hero_table():
    one_hots = []
    all_hero_stats = []
    ids = []
    for hero in consts.HEROES_LIST:
        p_attr = [1 if hero["primary_attr"] == a else 0 for a in attributes]
        a_type = [1 if hero["attack_type"] == a else 0 for a in attack_type]
        roles = [0]*8
        for i in hero["roles"]:
            roles[role_map[i]] = 1

        stats = [hero[i] for i in stats_keys]
        stats.append(hero["pub_pick"]/hero["pub_win"])
        
        one_hots.append(p_attr+a_type+roles)
        all_hero_stats.append(stats)
        ids.append(hero["id"])
    
    all_hero_stats = np.array(all_hero_stats)
    normalizer = StandardScaler()
    all_hero_stats = normalizer.fit_transform(all_hero_stats)

    hero_table = dict([(id,one_hots[i]+all_hero_stats[i].tolist())for i,id in enumerate(ids)])
    return hero_table
    


def extract_features(entries,hero_table):
    features = []
    for i in entries:
        match = []
        for h in i["radiant_team"]:
            match.extend(hero_table[h]+[0])
        for h in i["dire_team"]:
            match.extend(hero_table[h]+[1])
        features.append(match)
    return torch.tensor(features,dtype=torch.float32)

