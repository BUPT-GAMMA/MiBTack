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

class SparseDropout(nn.Module):
    def __init__(self, p):
        super(SparseDropout, self).__init__()
        self.p = p

    def forward(self, input):
        input_coal = input.coalesce()
        drop_val = F.dropout(input_coal._values(), self.p, self.training)
        return torch.sparse.FloatTensor(input_coal._indices(), drop_val, input.shape)

class MixedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MixedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Our fan_in is interpreted by PyTorch as fan_out (swapped dimensions)
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.bias is None:
            if input.is_sparse:
                res = torch.sparse.mm(input, self.weight)
            else:
                res = input.matmul(self.weight)
        else:
            if input.is_sparse:
                res = torch.sparse.addmm(self.bias.expand(input.shape[0], -1), input, self.weight)
            else:
                res = torch.addmm(self.bias, input, self.weight)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias is not None)
        
class MixedDropout(nn.Module):
    def __init__(self, p):
        super(MixedDropout, self).__init__()
        self.dense_dropout = nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, input):
        if input.is_sparse:
            return self.sparse_dropout(input)
        else:
            return self.dense_dropout(input)
            
            
class APPNP(nn.Module):
    def __init__(self, nfeat, nclass, device):
        super(APPNP, self).__init__()
        self.lr=0.01
        self.drop_prob = 0.5
        self.weight_decay=5e-6
        self.nfeat = nfeat
        self.nclass = nclass
        self.train_iters = 200
        self.alpha = 0.1
        self.niter = 10
        self.hidden_sizes = []#不能删，attack用
        self.hiddenunits = [64]
        bias = False
        
        # dropout
        if self.drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(self.drop_prob)
        self.act_fn = nn.ReLU()
        assert device is not None, "Please specify 'device'!"
        
        # 线性变换
        fcs = [MixedLinear(nfeat, self.hiddenunits[0], bias=bias)] #wx+b
        for i in range(1, len(self.hiddenunits)):
            fcs.append(nn.Linear(self.hiddenunits[i - 1], self.hiddenunits[i], bias=bias)) 
        fcs.append(nn.Linear(self.hiddenunits[-1], nclass, bias=bias))# len(hiddenunits)层的wx+b
        self.fcs = nn.ModuleList(fcs)
        
        self.device = device
        self.reg_params = list(self.fcs[0].parameters())
        
    def _transform_features(self, features):#多层MLP
        layer_inner = self.act_fn(self.fcs[0](self.dropout(features))) 
        for fc in self.fcs[1:-1]:
            layer_inner = self.act_fn(fc(layer_inner))
        res = self.fcs[-1](self.dropout(layer_inner))
        return res

    def initialize(self):
        for fc in self.fcs:
            fc.reset_parameters()
        
    def forward(self, x, adj):
        x = self._transform_features(x)
        z = x
        adj = self.dropout(adj)#??
        for _ in range(self.niter):
            z = adj @ z + self.alpha * x
        return F.log_softmax(z, dim=-1)
        
    def fit(self, features, adj, labels, idx_train, idx_val=None, normalize=True, initialize=True, verbose=False, patience=30, **kwargs):
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
    dataset = 'cora'
    config = Config(dataset)
    config.device = 0
    device = config.device
    seed = 19
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data = Data(dataset, dataset+'_ft_norm')

    appnp = APPNP(nfeat=data.features.shape[1], nclass=data.labels.max().item() + 1, device=device)
    appnp = appnp.to(device)
    appnp.fit(data.features_norm.todense(), data.adj.todense(), \
            data.labels, data.idx_train, data.idx_val, normalize=True, verbose=True)
    appnp.test(data.idx_test)