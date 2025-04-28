import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm
import sklearn.utils




class NNModel(nn.Module):
    def __init__(self,match_dim,lin_dim):
        super().__init__()
        self.hidden1 = nn.Linear(2*37*5,lin_dim)
        self.bmorm1 = nn.BatchNorm1d(lin_dim)
        self.relu = nn.ReLU()
        self.radiant_attn = nn.Linear(37,5)
        self.dire_attn = nn.Linear(37,5)
        # self.dropout = nn.Dropout(p=0.3)
        # self.hidden2 = nn.Linear(lin_dim,lin_dim//2)
        # self.bmorm2 = nn.BatchNorm1d(lin_dim//2)
        # self.dropout2 = nn.Dropout(p=0.3)
        self.output = nn.Linear(lin_dim,1)

        
    def forward(self,feature):
        feature = feature.view(feature.shape[0],10,37)
        radiant = feature[:,:5,:]
        dire = feature[:,5:,:]
        radiant_score = self.radiant_attn(radiant)
        radiant_w = torch.softmax(radiant_score,dim=1)
        dire_score = self.dire_attn(dire)
        dire_w = torch.softmax(dire_score,dim=1)
        radiant = (radiant.unsqueeze(3)*radiant_w.unsqueeze(2)).sum(dim=1).flatten(start_dim=1)
        dire = (dire.unsqueeze(3)*dire_w.unsqueeze(2)).sum(dim=1).flatten(start_dim=1)
        feature = torch.cat([radiant,dire],dim=1)
        feature = self.hidden1(feature)
        feature = self.bmorm1(feature)
        feature = self.relu(feature)
        feature = self.output(feature)
        return feature.squeeze(1)

    def train_model(self,features,labels,test_feature,test_labels,eta, epochs,batch_size,decay,early_stop):
        count_no_change = 0
        labels_int = labels.view(-1).long()
        class_counts = torch.bincount(labels_int)
        weight = torch.tensor([class_counts[0].float()/class_counts[1].float()])
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters(),lr = eta,weight_decay=decay)
        # sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
        batch_start = torch.arange(0,len(labels),batch_size)

        best_loss = -np.inf
        best_weights = None

        logloss = []
        acc_list = []
        train_loss = []
        train_accuracy = []
        for i in range(epochs):
            self.train()
            avg_train_loss = []
            avg_train_acc = []
            with tqdm.tqdm(batch_start,unit="batch",mininterval=0,disable=False) as progress:
                progress.set_description(f"Epoch {i}")
                for start in progress:
                    end = start+batch_size
                    batch_features = features[start:end]            
                    labels_batch = labels[start:end]
                    logits = self(batch_features)
                    labels_batch_smoothed = labels_batch.float()*(1.0-0.05)+0.5*0.05
                    loss = loss_fn(logits,labels_batch_smoothed)
                    preds = torch.sigmoid(logits)
                    avg_train_loss.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    accuracy = (preds.round() == labels_batch).float().mean()
                    avg_train_acc.append(float(accuracy))
                    progress.set_postfix(loss=float(loss),accuracy=float(accuracy))
            
            self.eval()
            with torch.no_grad():
                test_logtis = self(test_feature)
                test_preds = torch.sigmoid(test_logtis)
                test_accuracy = (test_preds.round() == test_labels).float().mean().item()
                test_loss = loss_fn(test_logtis,test_labels.float()).item()
                # sched.step()
            logloss.append(test_loss)
            acc_list.append(float(test_accuracy))
            train_accuracy.append(np.average(avg_train_acc))
            train_loss.append(np.average(avg_train_loss))
            if test_accuracy > best_loss:
                best_loss = test_accuracy
                best_weights = copy.deepcopy(self.state_dict())
                count_no_change = 0
            else:
                count_no_change += 1
                if(count_no_change>=early_stop):
                    break
        
        self.load_state_dict(best_weights)
        return best_loss,logloss,acc_list,train_accuracy,train_loss
