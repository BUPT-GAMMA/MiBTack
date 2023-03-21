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
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from GNNs.sgc import SGC
from GNNs.appnp import APPNP
from tqdm import tqdm
import argparse
import random
import copy
import scipy

def difference_of_logits(logits: torch.Tensor, labels: torch.Tensor, labels_infhot: Optional[torch.Tensor] = None) -> torch.Tensor:
    class_logits = logits[labels]
    other_logits = (logits - labels_infhot).max(0).values
    return class_logits - other_logits

def get_output(model, adj, feat_norm, target_node):
    modified_adj, modified_features, labels = utils.to_tensor(adj, feat_norm, labels, device=self.device)

class FGA_mini_L0(BaseAttack):
    
    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, device='cpu'):

        super(FGA_mini_L0, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, device=device)


        assert not self.attack_features, "not support attacking features"

        if self.attack_features:
            self.feature_changes = Parameter(torch.FloatTensor(feature_shape))
            self.feature_changes.data.fill_(0)
        self.loss_type = None

    def attack(self, ori_features, ori_adj, labels, idx_train, target_node, confidence, verbose=False, **kwargs):
        modified_adj, modified_features, labels = utils.to_tensor(ori_adj.todense(), ori_features.todense(), labels, device=self.device)

        self.surrogate.eval()
        if verbose == True:
            print('number of pertubations: %s' % n_perturbations)

#         pseudo_labels = self.surrogate.predict().detach().argmax(1)
#         pseudo_labels[idx_train] = labels[idx_train]
        
        pseudo_labels = labels
        modified_adj.requires_grad = True
        # print(modified_adj.requires_grad)
        for i in range(1000):
            adj_norm = utils.normalize_adj_tensor(modified_adj)
            output = self.surrogate(modified_features, adj_norm)
            if i == 0:
                labels_infhot = torch.zeros_like(output[target_node].detach())
                labels_infhot[labels[target_node]] = float('inf')
                logit_diff_func = partial(difference_of_logits, labels=labels[target_node], labels_infhot=labels_infhot)
            logits = torch.exp(output[target_node].detach().clone())
#             print(logits)
            real = logits[labels[target_node]].clone()
            logits[labels[target_node]] = -float('inf')
            other = (logits).max() #非label index的，最大logits的结果
            # print("prob of other, real:", target_node, other, real)
            isadv = (other-real) > confidence # other和real logit的差值是否大于40
            if isadv:
                modified_adj = modified_adj.detach().cpu().numpy()
                modified_adj = sp.csr_matrix(modified_adj)
                self.check_adj(modified_adj)
                self.modified_adj = modified_adj
                return i           
            loss = -logit_diff_func(logits=output[target_node])
            grad = torch.autograd.grad(loss, modified_adj)[0]
            # bidirection
            grad = (grad[target_node] + grad[:, target_node]) * (-2*modified_adj[target_node] + 1)
            grad[target_node] = -10
            grad_argmax = torch.argmax(grad)

            value = -2*modified_adj[target_node][grad_argmax] + 1
            modified_adj.data[target_node][grad_argmax] += value
            modified_adj.data[grad_argmax][target_node] += value
            # print(value,grad_argmax)

        modified_adj = modified_adj.detach().cpu().numpy()
        modified_adj = sp.csr_matrix(modified_adj)
        self.check_adj(modified_adj)
        self.modified_adj = modified_adj
        return None

def select_nodes(gnnmodel, data):
    '''
    selecting nodes as reported in nettack paper:
    (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    (ii) the 10 nodes with lowest margin (but still correctly classified) and
    (iii) 20 more nodes randomly
    '''

    gnnmodel.eval()
    output = gnnmodel.predict()

    margin_dict = {}
    for idx in data.idx_test:
        margin = classification_margin(output[idx], data.labels[idx])
        if margin < 0: # only keep the nodes correctly classified
            continue
        margin_dict[idx] = margin
    sorted_margins = sorted(margin_dict.items(), key=lambda x:x[1], reverse=True)
    high = [x for x, y in sorted_margins[: 10]]
    low = [x for x, y in sorted_margins[-10: ]]
    other = [x for x, y in sorted_margins[10: -10]]
    other = np.random.choice(other, 20, replace=False).tolist()

    return high + low + other


for GNN_name in ['GCN']:#['GCN', 'SGC']:
    for dataset in ['cora']:#['cora', 'citeseer', 'polblogs']:
        print("*"*20, GNN_name, dataset)
        target_or_test = 'target'

        config = Config(dataset)
        config.device = 3
        device = config.device

        seed = 19
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        data = Data(dataset, dataset+'_ft_norm')

        # print(data.adj.shape, data.adj.nonzero()[0].shape[0]/2, data.features_norm.shape, data.labels.max(), data.idx_train.shape[0], data.idx_val.shape[0], data.idx_test.shape[0])

        data.idx_unlabeled = np.union1d(data.idx_val, data.idx_test)

            
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
        surrogate.test(data.idx_test)

        # Setup Attack Model
        model = FGA_mini_L0(surrogate, nnodes=data.adj.shape[0], device=device)
        model = model.to(device)
        model.loss_type = loss_type


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
        confidence = torch.FloatTensor([0.0]).to(device)
        cnt = 0
        num = 0
        bound = []
        evas_each_acc = []
        δ_init = []
        for target_node in tqdm(tar_all):
        # for target_node in [47]:

            model.modified_adj = copy.deepcopy(data.adj)
            b = model.attack(data.features_norm, copy.deepcopy(data.adj), data.labels, data.idx_train, target_node, confidence)
            bound.append(b)

            data.modified_adj = model.modified_adj
            if b == 0:
                δ_init.append(None)
            else:
                δ_init.append(np.array(data.modified_adj.todense()[target_node]-data.adj[target_node]))

        print("bound:",bound)
        print("L0:",sum(bound)/len(bound))

        # with open('FGM_init_'+dataset+'_'+loss_type+'.pkl','wb') as f:
        #     pkl.dump([bound,δ_init],f)