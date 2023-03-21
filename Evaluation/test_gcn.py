from sklearn.metrics import accuracy_score,roc_auc_score,f1_score
from deeprobust.graph import utils 
from deeprobust.graph.defense import GCN as GCN_vic
import torch
import numpy as np
import scipy.sparse as sp

def fair_metric(output,idx,labels,sens):
    val_y = labels[idx].cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()]==0 #所有节点中sens为0的
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()]==1
    idx_s0_y1 = np.bitwise_and(idx_s0,val_y==1) # 属性为0 且 分类为1的
    idx_s1_y1 = np.bitwise_and(idx_s1,val_y==1) # 属性为1 且 分类为1的
    pred_y = (output[idx].squeeze()>0).type_as(labels).cpu().numpy()
    parity = abs(sum(pred_y[idx_s0])/sum(idx_s0)-sum(pred_y[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred_y[idx_s0_y1])/sum(idx_s0_y1)-sum(pred_y[idx_s1_y1])/sum(idx_s1_y1))
    return parity,equality    

def eval_gcn(modified_adj, features, config, data):
    '''
    输入必须是normalize的，self-loop的，其他无所谓
    '''
    acc_list = []
#     roc_list = []
    acc_target = []
#     roc_target = []
    for _ in range(config.eval_num_gcn_ave):
        test_gcn = GCN_vic(nfeat=features.shape[1], nclass=data.labels.max().item()+1,
                            nhid=config.eval_hidden, dropout=config.eval_dropout, weight_decay=config.eval_weight_decay,
                           device=config.device).to(config.device) 
        test_gcn.fit(features, modified_adj, data.labels, data.idx_train, data.idx_val, normalize=config.normalize)
        test_gcn.eval()
        output = test_gcn.predict()
        for tar in data.target_sets:
            acc_target.append(utils.accuracy(output[tar], data.labels[tar]).item())
        acc_list.append(utils.accuracy(output[data.idx_test], data.labels[data.idx_test]).item())
    log = ""
    for l in [acc_list,acc_target]:
        arr = np.array(l)*100
        arr_mean = np.mean(arr)
        arr_std = np.std(arr)
        print(arr_mean,"\t",arr_std)  
        log += str(arr_mean) +"\t"+ str(arr_std) +"\n"
    return log

def eval_edges(modified_adj, adj, data):
    '''
    更详细地分析净流入流出
    '''
    delta_adj = sp.csr_matrix(modified_adj.numpy())-adj
    row,col = delta_adj.nonzero()
    mod_same_label = []
    mod_same_sens = []
    mod_dif_label = []
    mod_dif_sens = []
    for i,j in zip(row,col):
        if data.labels[i] == data.labels[j]:
            mod_same_label.append(delta_adj[i,j])
        else:
            mod_dif_label.append(delta_adj[i,j])
        if data.sens[i] == data.sens[j]:
            mod_same_sens.append(delta_adj[i,j])
        else:
            mod_dif_sens.append(delta_adj[i,j])
    mod_same_label = np.array(mod_same_label)
    mod_same_sens = np.array(mod_same_sens)
    mod_dif_label = np.array(mod_dif_label)
    mod_dif_sens = np.array(mod_dif_sens)
    print("更详细地分析净流入流出:")
    print("加了label intra的边：",mod_same_label[mod_same_label>0].shape[0],\
         "加了label inter的边：",mod_dif_label[mod_dif_label>0].shape[0],\
         "加了sens intra的边：",mod_same_sens[mod_same_sens>0].shape[0],\
         "加了sens inter的边：",mod_dif_sens[mod_dif_sens>0].shape[0])
    print("删了label intra的边：",mod_same_label[mod_same_label<0].shape[0],\
         "删了label inter的边：",mod_dif_label[mod_dif_label<0].shape[0],\
         "删了sens intra的边：",mod_same_sens[mod_same_sens<0].shape[0],\
         "删了sens inter的边：",mod_dif_sens[mod_dif_sens<0].shape[0])