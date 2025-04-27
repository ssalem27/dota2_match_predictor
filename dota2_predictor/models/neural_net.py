import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm
import sklearn.utils

from dota2_predictor.models.embedder import HeroEmbeddings



class NNModel(nn.Module):
    def __init__(self,hero_dim,lin_dim):
        super().__init__()
        self.h_out_dim = 3*hero_dim+22+1
        self.hero = HeroEmbeddings(hero_dim)
        self.hidden1 = nn.Linear(10*self.h_out_dim,lin_dim)
        self.bmorm1 = nn.BatchNorm1d(lin_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.hidden2 = nn.Linear(lin_dim,lin_dim//2)
        self.bmorm2 = nn.BatchNorm1d(lin_dim//2)
        self.dropout2 = nn.Dropout(p=0.3)
        self.output = nn.Linear(lin_dim//2,1)

        
    def forward(self,p_attrs,a_types,role_i,float_stats):
        heros = self.hero(p_attrs,a_types,role_i,float_stats)
        batch_size = heros.shape[0]//10
        heros = heros.view(batch_size,10,self.h_out_dim)
        matches = heros.view(batch_size, -1)
        feature = self.hidden1(matches)
        feature = self.bmorm1(feature)
        feature = self.relu(feature)
        feature = self.dropout(feature)
        feature = self.hidden2(feature)
        feature = self.bmorm2(feature)
        feature = self.relu(feature)
        feature = self.dropout(feature)
        feature = self.output(feature)
        return feature.squeeze(1)

    def train_model(self,features,labels,test_feature,test_labels,eta, epochs,batch_size,decay,early_stop):
        count_no_change = 0
        labels_int = labels.view(-1).long()
        class_counts = torch.bincount(labels_int)
        weight = torch.tensor([class_counts[0].float()/class_counts[1].float()])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
        optimizer = optim.Adam(self.parameters(),lr = eta,weight_decay=decay)
        # sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
        batch_start = torch.arange(0,len(labels),batch_size)

        best_loss = np.inf
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
                    batch_pattr = features[0][start*10:end*10]
                    batch_atype = features[1][start*10:end*10]
                    roles = features[2][start*10:end*10]
                    batch_stats = features[3][start*10:end*10]              
                    labels_batch = labels[start:start+batch_size]
                    logits = self(batch_pattr,batch_atype,roles,batch_stats)
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
                test_pattr = test_feature[0]
                test_atype = test_feature[1]
                test_roles = test_feature[2]
                test_stats = test_feature[3]
                test_logtis = self(test_pattr,test_atype,test_roles,test_stats)
                test_preds = torch.sigmoid(test_logtis)
                test_accuracy = (test_preds.round() == test_labels).float().mean().item()
                test_loss = loss_fn(test_logtis,test_labels.float()).item()
                # sched.step()
            logloss.append(test_loss)
            acc_list.append(float(test_accuracy))
            train_accuracy.append(np.average(avg_train_acc))
            train_loss.append(np.average(avg_train_loss))
            if test_loss < best_loss:
                best_loss = test_loss
                best_weights = copy.deepcopy(self.state_dict())
                count_no_change = 0
            else:
                count_no_change += 1
                if(count_no_change>=early_stop):
                    break
        
        self.load_state_dict(best_weights)
        return best_loss,logloss,acc_list,train_accuracy,train_loss
