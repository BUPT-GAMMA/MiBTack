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
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm
from functools import partial
import math
from typing import Optional
from torch import nn, Tensor
from torch.autograd import grad

for GNN_name in ['GCN']:#['GCN','SGC']:
    for dataset in ['pubmed']:#'cora', 'citeseer','polblogs', 
        for target_atk in ['all','best_tar']: #'all' specify_tar
            print("GNN_name:",GNN_name, " dataset:", dataset, " target_atk:", target_atk)
            target_or_test = 'target'
            device = 4
            verbose = False
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
            nnodes = data.n_nodes
            batch_view = lambda tensor: tensor.view(batch_size, *[1] * (nnodes - 1))
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            surrogate.eval()
            adj, features, labels = utils.to_tensor(copy.deepcopy(data.adj), data.features_norm, data.labels, device=device)
            adj_norm = utils.normalize_adj_tensor(adj,True)


            bound = []
            acc_list = []
            for target_node in tar_all:
                output = surrogate(features, adj_norm)
                logits = torch.exp(output[target_node].detach().clone())
                logits[labels[target_node]] = 0.0
                target_class = torch.argmax(logits).item()
                mod_adj = copy.deepcopy(adj)
                b = 0
                if target_atk == 'all':
                    adv_pool = np.hstack((torch.where(labels!=labels[target_node])[0].cpu().numpy(), data.adj[target_node].nonzero()[1]))
                elif target_atk == 'best_tar':
                    adv_pool = np.hstack((torch.where(labels==target_class)[0].cpu().numpy(), data.adj[target_node].nonzero()[1]))
                while(True):
                    adj_norm = utils.normalize_adj_tensor(mod_adj, True)
                    output = surrogate(features, adj_norm)
                    logits = torch.exp(output[target_node].detach().clone())
            #         print(logits)
                    predict = torch.argmax(logits).item()
            #         print(logits,predict,target_class)
                    if predict != labels[target_node]:
                        if verbose:
                            print(target_node,b)
                        bound.append(b)
                        acc_list.append(1)
                        break
                    if adv_pool.shape[0] == 0:
                        acc_list.append(0)
                        break
                    adv = np.random.choice(adv_pool)
                    index_adv = torch.LongTensor([[target_node, adv],[adv, target_node]])   #row, col
                    values_adv = torch.FloatTensor([-2*mod_adj[target_node, adv]+1, -2*mod_adj[adv, target_node] + 1])    #data
                    adj_tmp = torch.sparse.FloatTensor(index_adv, values_adv, torch.Size([nnodes, nnodes])).to(device)   #torch.Size()
                    mod_adj = mod_adj + adj_tmp
                    adv_pool = np.delete(adv_pool, np.where(adv_pool==adv)[0][0])
                    b += 1
            print(bound)
            print("ASR:",sum(acc_list)/len(acc_list)*100,", bound:",sum(bound)/len(bound))  