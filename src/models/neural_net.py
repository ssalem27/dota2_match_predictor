import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm
import sklearn.utils



class NNModel(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.hidden = nn.Linear(input_dim,3*input_dim)
        self.relu = nn.ReLU()
        self.output = nn.Linear(3*input_dim,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,feature):
        feature = self.relu(self.hidden(feature))
        feature = self.sigmoid(self.output(feature))
        return feature

    def train_model(self,features,labels,test_feature,test_labels,eta, epochs,batch_size):
        class_counts = torch.bincount(labels.long())
        class_weights = 1.0 / class_counts.float()
        weight_tensor = class_weights[labels.long()]
        loss_fn = nn.BCELoss(weight=weight_tensor)
        optimizer = optim.Adam(self.parameters(),lr = eta)
        batch_start = torch.arange(0,len(features),batch_size)

        best_acc = -np.inf
        best_weights = None

        logloss = []
        loss_01 = []

        for i in range(epochs):
            self.train()

            with tqdm.tqdm(batch_start,unit="batch",mininterval=0,disable=False) as progress:
                progress.set_description(f"Epoch {i}")
                for start in progress:
                    feature_batch = features[start:start+batch_size]
                    labels_batch = labels[start:start+batch_size]
                    preds = self(feature_batch)
                    loss = loss_fn(preds,labels_batch.float())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    accuracy = (preds.round() == labels_batch).float().mean()

                    lgloss = -torch.mean(labels_batch * torch.log(preds + 1e-9) + (1 - labels_batch) * torch.log(1 - preds + 1e-9))
                    logloss.append(lgloss)
                    loss_01.append(accuracy)

                    progress.set_postfix(loss=float(loss),accuracy=float(accuracy))
            
            self.eval()
            curr_val_feature, curr_val_labels = sklearn.utils.shuffle(test_feature,test_labels,random_state = i)
            num_to_val = np.random.randint(len(curr_val_feature))
            curr_val_feature = curr_val_feature[:num_to_val]
            curr_val_labels = curr_val_labels[:num_to_val]

            preds = self(curr_val_feature)
            accuracy = (preds.round() == curr_val_labels).float().mean()
            accuracy = float(accuracy)
            if accuracy > best_acc:
                best_acc = accuracy
                best_weights = copy.deepcopy(self.state_dict())
        
        self.load_state_dict(best_weights)
        return best_acc,logloss,loss_01
