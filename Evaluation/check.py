import sys
sys.path.append("..")
from Preprocess.preprocessing import *
import dgl
from test_gcn import eval_gcn
import pickle as pkl
from deeprobust.graph import utils 
import scipy.sparse as sp

def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2*(features - min_values).div(max_values-min_values) - 1

log = ""
# for dataset in ['nba','cora','citeseer','pokecz','pokecn','cora-ml']:
for dataset in ['pokecz_sub']:#,'pokecz','pokecn']:
    # for hidden in [16,32,64,128]:#,32,64,128,256]:
        # print("*"*10+dataset+"*"*10+str(hidden))
        # config = Config(dataset)
        # config.device = 0
        # config.eval_hidden = hidden
        # config.eval_weight_decay = 1e-5
        # config.eval_lr = 0.001
        # device = config.device
        
        # data = Data()
        # with open('/home/zmm/advUnno/dataset/'+dataset+'.pkl','rb') as f:
            # data.adj, data.features, data.labels, data.idx_train, data.idx_val, data.idx_test, data.target_sets, data.sens = pkl.load(f)
        # print(data.adj.shape, data.adj.nonzero()[0].shape[0]/2, data.features.shape, data.labels.max(), data.idx_train.shape[0], data.idx_val.shape[0], data.idx_test.shape[0])
        
        # data.features = feature_norm(data.features)
        # log += dataset +'\n'+ eval_gcn(data.adj, data.features, config, data)
    for eval_lr in [0.001, 0.01]:#,32,64,128,256]:
        for norm in ['nba_ft_norm', 'cora_ft_norm', None]:
            print("*"*10+dataset+"*"*10+str(eval_lr),norm)
            config = Config(dataset)
            config.device = 0
            config.eval_hidden = 16
            config.eval_weight_decay = 1e-5
            config.eval_lr = eval_lr
            device = config.device
            config.eval_num_gcn_ave = 2
            data = Data(dataset,norm)
            log += dataset +'\n'+ eval_gcn(data.adj, data.features_norm, config, data)
with open('clean_results.txt','w') as f:
    f.write(log)