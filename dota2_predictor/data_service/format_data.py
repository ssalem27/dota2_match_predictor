from dota2_predictor.data_service.build_db import get_saved_matches
import numpy as np
import dota2_predictor.data_service.consts as consts
import torch


agi_heroes = list(filter(lambda x: x["primary_attr"]=="agi",consts.HEROES_LIST))
str_heroes = list(filter(lambda x: x["primary_attr"]=="str",consts.HEROES_LIST))
int_heroes = list(filter(lambda x: x["primary_attr"]=="int",consts.HEROES_LIST))
attr_all_heroes = list(filter(lambda x: x["primary_attr"]=="all",consts.HEROES_LIST))
heroes = agi_heroes+str_heroes+int_heroes+attr_all_heroes

roles = []
for i in consts.HEROES_LIST:
    roles+= i["roles"]

roles = set(roles)

hero_table = dict([(hero["id"],(i,hero)) for i,hero in enumerate(heroes)])
role_table = dict([(role,i) for i,role in enumerate(roles)])

radiant_heroes_offset = 0
radiant_roles_offset = len(heroes)
dire_heroes_offset = len(heroes)+len(roles)
dire_roles_offset = 2*len(heroes)+len(roles)

def extract_features(entries):
    features = np.zeros(shape=(len(entries),(2*len(hero_table)+2*len(role_table))))
    for i,entry in enumerate(entries):
        feature = np.zeros(shape=(2*len(hero_table)+2*len(role_table)))
        for j in entry["radiant_team"]:
            feature[radiant_heroes_offset+hero_table[j][0]] = 1
            for r in hero_table[j][1]["roles"]:
                feature[radiant_roles_offset+role_table[r]]+=1
        for j in entry["dire_team"]:
            feature[dire_heroes_offset+hero_table[j][0]] = 1
            for r in hero_table[j][1]["roles"]:
                feature[dire_roles_offset+role_table[r]]+=1
        features[i]=feature
    return np.float32(features)

def format_for_torch(entries):
    return torch.from_numpy(entries)

        
        
        