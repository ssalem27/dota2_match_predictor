from data_service.build_db import *
from models.logistic_regression import LGModel
import time
import traceback
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss

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

# features,labels = get_saved_matches()
# features_train, features_val, labels_train, labels_val = train_test_split(
#     features, labels, test_size=0.2, random_state=42, stratify=labels
# )
# model = LGModel()
# model.logistic_regression(features_train,labels_train,alpha=0.01,epochs=10000,batch_size=100)

# print(zero_one_loss(labels_val,model.predict(features_val)))