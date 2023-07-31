# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 10:34:16 2023

@author: hwang147
"""
import torch
import numpy as np

class EmbeddingLayer(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)
class MMoEModel(torch.nn.Module):
    """
    A pytorch implementation of MMoE Model.

    Reference:
        Ma, Jiaqi, et al. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. KDD 2018.
    """

    def __init__(self, input_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, expert_num, dropout):
        super().__init__()
        
        self.numerical_layer = torch.nn.Linear(input_num, embed_dim)
        #self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim
        self.task_num = task_num
        self.expert_num = expert_num

        self.expert = torch.nn.ModuleList([MultiLayerPerceptron(embed_dim, bottom_mlp_dims, dropout, output_layer=False) for i in range(expert_num)])
        
        self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(embed_dim, expert_num), torch.nn.Softmax(dim=1)) for i in range(task_num)])
        
        
        self.tower = torch.nn.ModuleList([MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout) for i in range(task_num)])
    
    
    def forward(self, numerical_x):
        """
        :param 
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        
        numerical_emb = self.numerical_layer(numerical_x)
        #emb = torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim)
        gate_value = [self.gate[i](numerical_emb).unsqueeze(1) for i in range(self.task_num)]
        fea = torch.cat([self.expert[i](numerical_emb).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.task_num)]
        
        results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num)]
        return results
