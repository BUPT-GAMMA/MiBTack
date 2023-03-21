import torch
from deeprobust.graph.targeted_attack import BaseAttack
from torch.nn.parameter import Parameter
from copy import deepcopy
from deeprobust.graph import utils
import torch.nn.functional as F
import scipy.sparse as sp
from functools import partial
import math
from typing import Optional
from torch import nn, Tensor
import torch
import numpy as np
import sys
sys.path.append("..")
from Preprocess.preprocessing import *
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from GNNs.sgc import SGC
from GNNs.appnp import APPNP
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from tqdm import tqdm
import argparse
import random
import copy
import scipy
import pickle as pkl

for GNN_name in ['GCN']:#, 'SGC', 'APPNP']:
    # for dataset in ['cora', 'citeseer', 'polblogs']:#, 'pubmed']: #
    for dataset in ['cora']: #
        print("*"*20, GNN_name, dataset,"*"*20)
        target_or_test = 'target'
        verbose = False

        config = Config(dataset)
        config.device = 1
        device = config.device

        seed = 19
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        data = Data(dataset, dataset+'_ft_norm')
        nnodes=data.adj.shape[0]
        print(data.adj.shape, data.adj.nonzero()[0].shape[0]/2, data.features_norm.shape, data.labels.max(), data.idx_train.shape[0], data.idx_val.shape[0], data.idx_test.shape[0])

        data.idx_unlabeled = np.union1d(data.idx_val, data.idx_test)

        # Setup Surrogate model
        if GNN_name == 'SGC':
            surrogate = SGC(nfeat=data.features.shape[1], nclass=data.labels.max().item() + 1, device=device)
            surrogate = surrogate.to(device)
            surrogate.fit(data.features_norm.todense(), data.adj.todense(), \
                    data.labels, data.idx_train, data.idx_val, normalize=True, verbose=False)
            surrogate.test(data.idx_test)
        if GNN_name == 'GCN':
            # Setup Surrogate model
            surrogate = GCN(nfeat=data.features.shape[1], nclass=data.labels.max().item()+1,
                                        nhid=config.eval_hidden, dropout=config.eval_dropout, weight_decay=config.eval_weight_decay,
                                       device=config.device).to(config.device) 
            surrogate.fit(data.features_norm, data.adj, data.labels, data.idx_train, data.idx_val, normalize=True)
            surrogate.test(data.idx_test)
        if GNN_name == 'APPNP':
            # Setup Surrogate model
            surrogate = APPNP(nfeat=data.features.shape[1], nclass=data.labels.max().item()+1, device=config.device).to(config.device) 
            surrogate = surrogate.to(device)
            surrogate.fit(data.features_norm, data.adj, data.labels, data.idx_train, data.idx_val, normalize=True)
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
        n_class = data.labels.max().item()+1
        start_state = {}
        ori_adj, features, labels = utils.to_tensor(copy.deepcopy(data.adj), data.features_norm, data.labels, device=device)
        surrogate.eval()
        for target_node in tar_all:
            dif_max = -1000
            for ci in range(n_class):
                if ci == labels[target_node]:
                    continue
                modified_adj = copy.deepcopy(ori_adj.to_dense())
                modified_adj.requires_grad = True
                adj_norm = utils.normalize_adj_tensor(modified_adj)
                output = surrogate(features, adj_norm)
                logits = torch.exp(output[target_node].detach().clone())
                if verbose:
                    print(logits.detach().cpu().numpy())
                loss = -(output[target_node,labels[target_node]] - output[target_node, ci])
                grad = torch.autograd.grad(loss, modified_adj)[0]
                # bidirection
                grad = (grad[target_node] + grad[:, target_node]) * (-2*modified_adj[target_node] + 1)
                grad[target_node] = -10
                grad_argmax = torch.argmax(grad)

                value = -2*modified_adj[target_node][grad_argmax] + 1
                modified_adj.data[target_node][grad_argmax] += value
                modified_adj.data[grad_argmax][target_node] += value

                #记录 logits差值
                adj_norm_test = utils.normalize_adj_tensor(modified_adj)
                output = surrogate(features, adj_norm_test)
                if verbose:
                    print(ci,value,grad_argmax,labels[grad_argmax])
                logits = torch.exp(output[target_node].detach().clone())
                if verbose:
                    print(logits.detach().cpu().numpy())
                dif = logits[ci].item() - logits[labels[target_node]].item()
                if dif > dif_max:
                    dif_max = dif
                    start_state[target_node] = grad_argmax.item()
                if verbose:
                    print("*"*20,ci,dif,dif_max)
                   
        with open('explore/start_'+GNN_name+'_'+dataset+'_'+target_or_test+'_.pkl', 'wb') as f:
            pkl.dump(start_state,f)