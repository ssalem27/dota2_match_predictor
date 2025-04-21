import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss,zero_one_loss
import numpy as np

class LGModel():

    def logistic_regression(self,features,gold_labels,alpha,epochs,batch_size):
        #logistic regression with gradient descent using minibatching
        model = SGDClassifier(loss='log_loss',penalty='l2',alpha=alpha,learning_rate="optimal")
        log_losses = []
        loss_01 = []
        for i in range(epochs):
            print(f"cur_epoch = {i}")
            features,gold_labels = sklearn.utils.shuffle(features,gold_labels,random_state=i)
            for j in range(0,len(features),batch_size):
                batch_features = features[j:j+batch_size]
                batch_labels = gold_labels[j:j+batch_size]

                if i+j == 0:
                    model.partial_fit(batch_features,batch_labels,classes=np.array([0,1]))
                else:
                    model.partial_fit(batch_features,batch_labels)
            probs = model.predict_proba(features)[:, 1]
            log_losses.append(log_loss(gold_labels, probs))
            loss_01.append(zero_one_loss(gold_labels,np.array([1 if prob>=0.5 else 0 for prob in probs]),normalize=False))

        self.model = model
        return log_losses,loss_01

    def predict(self,features):
        return self.model.predict(features)


