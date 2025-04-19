import requests
import os
import dotenv

dotenv.load_dotenv()

HEROES_ENDPOINT = "heroes"
MATCHES_ENPOINT ="publicMatches"
BASE_URL = os.environ["DOTA_API_BASE_URL"]

def get_heroes():
    heroes = requests.request("GET",BASE_URL+HEROES_ENDPOINT)
    return heroes.json()

def get_matches(): 
    #return a 
    matches = requests.request("GET",BASE_URL+MATCHES_ENPOINT)
    return matches.json()