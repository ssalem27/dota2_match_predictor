from data_service.build_db import get_saved_matches
import numpy as np
import data_service.consts as consts

hero_table = dict([(hero["id"],i) for i,hero in enumerate(consts.HEROES_LIST)])

def extract_features(entries):
    features = np.zeros(shape=(len(entries),2*len(hero_table)))
    for i,entry in enumerate(entries):
        feature = np.zeros(shape=2*len(hero_table))
        for j in entry["radiant_team"]:
            feature[hero_table[j]] = 1
        for j in entry["dire_team"]:
            feature[hero_table[j]+len(hero_table)] = 1
        features[i]=feature
    return features
        
        
        