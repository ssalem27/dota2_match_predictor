from dota2_predictor.data_service.build_db import get_saved_matches
import numpy as np
import dota2_predictor.data_service.consts as consts
import torch
from dota2_predictor.models.embedder import HeroEmbeddings, TeamComp
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



attribute_map = {
    "agi":0,
    "str":1,
    "int":2,
    "all":3
}

a_type_map = {
    "Melee": 0,
    "Ranged": 1
}

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
    stats_list = [[i[s] for s in stats_keys] for i in consts.HEROES_LIST]
    roles_list = [h["roles"] for h in consts.HEROES_LIST]

    one_hot_roles = []
    for i in roles_list:
        roles = [0,0,0,0,0,0,0,0]
        for r in i:
            roles[role_map[r]]=1 
        one_hot_roles.append(roles)
    
    normalizer = StandardScaler(copy=False)
    stats_list = normalizer.fit_transform(stats_list)
    return dict([
        (
            i["id"], 
                [attribute_map[i["primary_attr"]], 
                a_type_map[i["attack_type"]],
                one_hot_roles[z],
                stats_list[z]]
            )
        for z,i in enumerate(consts.HEROES_LIST)
    ])

def extract_features(entries,hero_table):
    features = [[],[],[],[]]
    for i in entries:
        for h in i["radiant_team"]:
            features[0].append(hero_table[h][0])
            features[1].append(hero_table[h][1])
            features[2].extend(hero_table[h][2])
            features[3].extend(hero_table[h][3])
        for h in i["dire_team"]:
            features[0].append(hero_table[h][0])
            features[1].append(hero_table[h][1])
            features[2].extend(hero_table[h][2])
            features[3].extend(hero_table[h][3])
    return [torch.tensor(features[0]).float(),torch.tensor(features[1]).float(),torch.tensor(features[2],dtype=torch.float32).view(-1,8),torch.tensor(features[3],dtype=torch.float32).view(-1,22)]
