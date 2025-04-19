from data_service.build_db import get_saved_matches
import numpy as np
import consts



def extract_features():
    entries = get_saved_matches()
    features = []
    labels = []

    for entry in entries:
        
        