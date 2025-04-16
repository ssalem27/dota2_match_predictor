from data_service.build_db import save_matches_list
from data_service.build_db import get_matches
import time
import traceback

while True:
    try:
        print("get matches")
        ret = save_matches_list(get_matches())
        print(f"upserted:{ret.upserted_count}")
        time.sleep(100)
        print("end sleep")
    except Exception as e:
        print(traceback.format_exc())
        raise e