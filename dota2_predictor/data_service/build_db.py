import pymongo, os, dotenv
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.mongo_client import MongoClient
from pymongo.operations import UpdateOne
from dota2_predictor.data_service.retrieve_data import get_matches
import numpy as np
import certifi


dotenv.load_dotenv()
certifi.where()

CLIENT_URL = os.environ["CLIENT_URL"]
DB_NAME = os.environ["DB_NAME"]
COLLECTION = os.environ["COLLECTION"]

mongodb: MongoClient = pymongo.MongoClient(CLIENT_URL)
db: Database = mongodb.get_database(DB_NAME)
collection: Collection = db.get_collection(COLLECTION)


def save_matches_list(matches: list[dict]):
    return collection.bulk_write(
        [
            UpdateOne(
                filter={"match_id": match["match_id"]},
                update={
                    "$set": {
                        "match_sew_num": match["match_seq_num"],
                        "radiant_win": match["radiant_win"],
                        "start_time": match["start_time"],
                        "duration": match["duration"],
                        "lobby_type": match["lobby_type"],
                        "game_mode": match["game_mode"],
                        "avg_rank_tier": match["avg_rank_tier"],
                        "num_rank_tier": match["num_rank_tier"],
                        "cluster": match["cluster"],
                        "radiant_team": match["radiant_team"],
                        "dire_team": match["dire_team"],
                    }
                },
                upsert=True,
            )
            for match in matches
        ]
    )

def get_saved_matches():
    entries = list(collection.find(projection={"_id":0,"dire_team":1,"radiant_team":1, "radiant_win":1}))
    return entries, np.array([int(data["radiant_win"]) for data in entries])