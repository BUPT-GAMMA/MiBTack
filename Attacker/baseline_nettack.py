import torch
import numpy as np
import sys
sys.path.append("..")
from Preprocess.preprocessing import *
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from tqdm import tqdm
import argparse
import random
import copy
import scipy
import os 
from GNNs.sgc import SGC
from GNNs.appnp import APPNP   
from nettack_model import Nettack
from torch import optim
from torch.nn.parameter import Parameter
from tqdm import tqdm
from functools import partial
import math
from typing import Optional
from torch import nn, Tensor
from torch.autograd import grad

for dataset in ['cora', 'citeseer', 'polblogs','pubmed']:
    for GNN_name in ['GCN','SGC', 'APPNP']:
        print("GNN_name:",GNN_name, " dataset:", dataset)
        target_or_test = 'target'
        init_boundary = False
        explore = True
        tanh = 1.0
        device = 3

        norm = 0
        config = Config(dataset)
        config.device = device
        seed = 19
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


        data = Data(dataset, dataset+'_ft_norm')
        nclass = data.labels.max().item() + 1
        # Setup Surrogate model
        if GNN_name == 'SGC':
            surrogate = SGC(nfeat=data.features.shape[1], nclass=data.labels.max().item() + 1, device=device)
            surrogate = surrogate.to(device)
            surrogate.fit(data.features_norm.todense(), data.adj.todense(), \
                    data.labels, data.idx_train, data.idx_val, normalize=True, verbose=False)
                    
            gcn = GCN(nfeat=data.features.shape[1], nclass=data.labels.max().item()+1,
                                        nhid=config.eval_hidden, dropout=config.eval_dropout, weight_decay=config.eval_weight_decay,
                                       device=config.device).to(config.device) 
            gcn = gcn.to(device)
            gcn.fit(data.features_norm, data.adj, data.labels, data.idx_train, data.idx_val, normalize=True)
            gcn.eval()
        if GNN_name == 'GCN':
            # Setup Surrogate model
            surrogate = GCN(nfeat=data.features.shape[1], nclass=data.labels.max().item()+1,
                                        nhid=config.eval_hidden, dropout=config.eval_dropout, weight_decay=config.eval_weight_decay,
                                       device=config.device).to(config.device) 
            surrogate = surrogate.to(device)
            surrogate.fit(data.features_norm, data.adj, data.labels, data.idx_train, data.idx_val, normalize=True)
        if GNN_name == 'APPNP':
            # Setup Surrogate model
            surrogate = APPNP(nfeat=data.features.shape[1], nclass=data.labels.max().item()+1, device=config.device).to(config.device) 
            surrogate = surrogate.to(device)
            surrogate.fit(data.features_norm, data.adj, data.labels, data.idx_train, data.idx_val, normalize=True)
            
            gcn = GCN(nfeat=data.features.shape[1], nclass=data.labels.max().item()+1,
                                        nhid=config.eval_hidden, dropout=config.eval_dropout, weight_decay=config.eval_weight_decay,
                                       device=config.device).to(config.device) 
            gcn = gcn.to(device)
            gcn.fit(data.features_norm, data.adj, data.labels, data.idx_train, data.idx_val, normalize=True)
            gcn.eval()
        surrogate.eval()
        surrogate.test(data.idx_test)

        if target_or_test == 'test':
            tar_all = data.idx_test.tolist()
        elif target_or_test == 'target':
            tar_all = []
            for tar in data.target_sets:
                for target_node in tar:
                    tar_all.append(target_node)
                    
        seed = 19
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        surrogate.eval()
        confidence = torch.FloatTensor([0.01]).to(device)
        cnt = 0
        num = 0
        bound = []
        evas_each_acc = []
        δ_init = []
        best_δ_dict = {}
        margin = []
        acc_list = []
        for target_node in tqdm(tar_all):
            logits = surrogate.predict(data.features_norm, data.adj)
            confidence = torch.exp(logits[target_node].detach().clone())
            predict = torch.argmax(confidence).item()
            if predict != data.labels[target_node]:
                b = 0
                bound.append(b)
                best_δ_dict[target_node] = None
                margin.append(None)
                acc_list.append(1)
        #         print("@@@ target:",target_node,b)
            else:
                if GNN_name != 'GCN':
                    model = Nettack(gcn, GNN_name, nnodes=data.adj.shape[0], attack_structure=True, attack_features=False, device=device).to(device)
                else:
                    model = Nettack(surrogate, GNN_name, nnodes=data.adj.shape[0], attack_structure=True, attack_features=False, device=device).to(device)
                b = model.attack(surrogate, data.features_norm, data.adj, data.labels, target_node)
                best_δ_dict[target_node] = model.modified_adj - data.adj
                bound.append(b)

                logits = surrogate.predict(data.features_norm, model.modified_adj)
                confidence = torch.exp(logits[target_node].detach().clone())
                predict = torch.argmax(confidence).item()
                if predict != data.labels[target_node]:
                    acc_list.append(1)
                else:
                    acc_list.append(0)
                confidence_label = confidence[data.labels[target_node]].item()
                confidence[data.labels[target_node]] = -10
                confidence_non_label = confidence[torch.argmax(confidence).item()].item()
                margin.append(confidence_label - confidence_non_label)

        assert len(bound) == 250
        assert len(acc_list) == 250
        print("bound:",sum(bound)/len(bound))