import time
import argparse
import numpy as np
import sys
from torch.autograd import Variable
sys.path.append("/home/zmm/advFair/FairGNN/src/")
import torch
from .utils import load_pokec_new
from deeprobust.graph import utils 
import pickle as pkl
import argparse
from utils import feature_norm
import pickle
import scipy

class Result(object):
    def __init__(self, config, data): 
        self.config = config
        self.data = data
        self.log = ""
        self.hyper_scale_acc = None
        self.description = ""
        self.path = 'result/'

class Config(object):
    def __init__(self,dataset):    
        # data
        self.dataset = dataset
        self.device='cpu'
                     
        #attack
        self.budget = 2
        self.atk_epochs = 200
        self.atk_loss_type = 'CE' # CE CW
        self.n_perturbations = None
        
        # evaluation
        self.eval_lr = 0.01
        self.eval_num_gcn_ave = 5  
        self.eval_fastmode=False             
        self.eval_epochs_gcn= 200
        self.eval_weight_decay=5e-4
        self.eval_dropout=0.5
        self.eval_hidden=16
        self.normalize=True

class Data(object):
    def __init__(self,dataset,norm):
        with open('/home/zmm/advUnno/dataset/'+dataset+'.pkl','rb') as f:
            self.adj, self.features, self.labels, self.idx_train, self.idx_val, self.idx_test, self.target_sets, self.sens = pkl.load(f)
        self.dataset = dataset
        self.n_nodes = self.adj.shape[0]
        self.preds = None
        if None == norm:
            self.features_norm = scipy.sparse.csr_matrix(self.features)
        elif 'cora_ft_norm' == norm:
            self.features_norm = utils.normalize_feature(scipy.sparse.csr_matrix(self.features))
        elif 'citeseer_ft_norm' == norm:
            self.features_norm = utils.normalize_feature(scipy.sparse.csr_matrix(self.features))
        elif 'polblogs_ft_norm' == norm:
            self.features_norm = utils.normalize_feature(scipy.sparse.csr_matrix(self.features))
        elif 'pubmed_ft_norm' == norm:
            self.features_norm = utils.normalize_feature(scipy.sparse.csr_matrix(self.features))
        elif 'nba_ft_norm' == norm:
            self.features_norm = scipy.sparse.csr_matrix(feature_norm(self.features))
            