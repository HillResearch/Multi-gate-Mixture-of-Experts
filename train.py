# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 16:25:05 2023

@author: hwang147
"""

import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import os
import numpy as np
from MMOE import MMoEModel
from Dataset import Dataset_patient

from sklearn.metrics import accuracy_score




def train(model, optimizer, data_loader, criterion, log_interval=100):
    model.train()
    total_loss = 0
    loader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (x,y, labels) in enumerate(loader):
        x,y, labels = x.float(), y.float(), labels.float()
        y = model(torch.cat([x,y], 1))
        loss_list = [criterion(y[i], labels[:, i].float()) for i in range(labels.size(1))]
        loss = 0
        for item in loss_list:
            loss += item
        loss /= len(loss_list)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            loader.set_postfix(loss=total_loss / log_interval)
            total_loss = 0
    
def test(model, data_loader, task_num):
    model.eval()
    labels_dict, predicts_dict, loss_dict = {}, {}, {}
    for i in range(task_num):
        labels_dict[i], predicts_dict[i], loss_dict[i] = list(), list(), list()
    with torch.no_grad():
        for categorical_fields, numerical_fields, labels in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            categorical_fields, numerical_fields, labels = categorical_fields.float(), numerical_fields.float(), labels.float()
            y = model(torch.cat([categorical_fields, numerical_fields],1))
            for i in range(task_num):
                labels_dict[i].extend(labels[:, i].tolist())
                predicts_dict[i].extend(y[i].tolist())
                loss_dict[i].extend(torch.nn.functional.binary_cross_entropy(y[i], labels[:, i].float(), reduction='none').tolist())
    auc_results, loss_results = list(), list()
    for i in range(task_num):
        auc_results.append(roc_auc_score(labels_dict[i], predicts_dict[i]))
        loss_results.append(np.array(loss_dict[i]).mean())
    return auc_results, loss_results

def test_result(model,data_loader,task_num):
    model.eval()
    
    with torch.no_grad():
        y_pred = [[] for i in range(task_num)]
        y_true = [[] for i in range(task_num)]
        for categorical_fields, numerical_fields, labels in data_loader:
            categorical_fields, numerical_fields, labels = categorical_fields.float(), numerical_fields.float(), labels.float()
            y = model(torch.cat([categorical_fields, numerical_fields],1))
            labels = labels.T.tolist()
            for i in range(task_num):
                y[i] = y[i].tolist()
                for j in range(len(categorical_fields)):
                    if y[i][j]>=0.5:
                    
                        y_pred[i].append(1)
                    else:
                        y_pred[i].append(0)
                    y_true[i].append(labels[i][j])
    accuracy = []
    for i in range(task_num):
        accuracy.append(accuracy_score(y_true[i],y_pred[i]))
    return accuracy
                    
    
            
    

def main(root_path,
         data_path,
         size,
         task_num,
         expert_num,
         epoch,
         learning_rate,
         batch_size,
         embed_dim,
         weight_decay,
         device,
         input_dim,
         tower_mlp_dims,
         bottom_mlp_dims,
         dropout):
    device = torch.device(device)
    Data = Dataset_patient(root_path,data_path,size)
    training_data = DataLoader(Data, batch_size=batch_size)
    model = MMoEModel(input_dim, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, expert_num, dropout)

    
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    #early_stopper = EarlyStopper(num_trials=2, save_path=save_path)
    for epoch_i in range(epoch):
        
        
        train(model, optimizer, training_data, criterion)
        auc, loss = test(model, training_data, task_num)
        print('epoch:', epoch_i, 'test: auc:', auc)
        for i in range(task_num):
            print('task {}, AUC {}, Log-loss {}'.format(i, auc[i], loss[i]))
        #if not early_stopper.is_continuable(model, np.array(auc).mean()):
            #print(f'test: best auc: {early_stopper.best_accuracy}')
            #break

    #model.load_state_dict(torch.load(save_path))
    auc, loss = test(model, training_data, task_num)
    #print('final accuracy:',auc,loss)
    
    accuracy = test_result(model,training_data,task_num)
    print('training accuracy:',accuracy)
    
    # f = open('{}_{}.txt'.format(model_name, dataset_name), 'a', encoding = 'utf-8')
    # f.write('learning rate: {}\n'.format(learning_rate))
    # for i in range(task_num):
    #     print('task {}, AUC {}, Log-loss {}'.format(i, auc[i], loss[i]))
    #     f.write('task {}, AUC {}, Log-loss {}\n'.format(i, auc[i], loss[i]))
    # print('\n')
    # f.write('\n')
    # f.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default= r'D:\Hill\Data')
    parser.add_argument('--data_path', default=['basic','diagnosis','result','treatment'])
    
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--task_num', type=int, default=3)
    parser.add_argument('--expert_num', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--input_dim', type = int, default= 834)
    parser.add_argument('--dropout',type = float,default = 0.2)
    parser.add_argument('--bottom_mlp_dims', default = (512, 256))
    parser.add_argument('--tower_mlp_dims', default = (128,64))
    parser.add_argument('--size', default = [0,4000])


    args = parser.parse_args()
    
    main(args.root_path,
         args.data_path,
         args.size,
         args.task_num,
         args.expert_num,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.embed_dim,
         args.weight_decay,
         args.device,
         args.input_dim,
         args.tower_mlp_dims,
         args.bottom_mlp_dims,
         args.dropout)

