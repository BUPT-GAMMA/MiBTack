import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
import numpy as np
import sys
sys.path.append("..")
from Preprocess.preprocessing import *
import random


class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass, device):
        self.lr=0.2
        self.weight_decay=5e-6
        self.nfeat = nfeat
        self.nclass = nclass
        self.train_iters = 100
        self.hidden_sizes = []

        super(SGC, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.layer = nn.Linear(self.nfeat, self.nclass).to(device)
        self.initialize()
     
    def initialize(self):
        stdv = 1. / math.sqrt(self.layer.weight.size(1))
        self.layer.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, x, adj_norm):
        if x.data.is_sparse:
            x = x.to_dense()
        if adj_norm.data.is_sparse:
            return F.log_softmax(self.layer(torch.spmm(adj_norm, torch.spmm(adj_norm, x))), dim=1)
        else:
            return F.log_softmax(self.layer(torch.mm(adj_norm, torch.mm(adj_norm, x))), dim=1)


    def fit(self, features, adj, labels, idx_train, idx_val=None, normalize=True, initialize=True, verbose=False, patience=30, **kwargs):
        """Train the GAT model, when idx_val is not None, pick the best model
        according to the validation loss.
        Parameters
        ----------
        pyg_data :
            pytorch geometric dataset object
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """

        if initialize:
            self.initialize()
            
        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            self.adj_norm = adj

        self.adj_norm = adj_norm
        self.features = features
        self.labels = labels

        if verbose:
            print('=== training GAT model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 30
        

        for i in range(self.train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(features, adj_norm)

            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(features, adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)

    def test(self, idx_test):
        """Evaluate GAT performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test

    def predict(self, features=None, adj=None):
        """By default, the inputs should be unnormalized data

        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.


        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(features, adj_norm)



if __name__ == "__main__":
    device = 3
    
    dataset = 'polblogs'

    config = Config(dataset)
    config.device = 0
    device = config.device

    seed = 19
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data = Data(dataset, dataset+'_ft_norm')
    
    sgc = SGC(nfeat=data.features.shape[1], nclass=data.labels.max().item() + 1, device=device)
    sgc = sgc.to(device)
    sgc.fit(data.features_norm.todense(), data.adj.todense(), \
            data.labels, data.idx_train, data.idx_val, normalize=True, verbose=True)
    sgc.test(data.idx_test)
