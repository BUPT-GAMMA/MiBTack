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
import pickle as pkl
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


def get_modified_adj(complementary, adj_changes, adj, target_node, device):
    if complementary is None:
        complementary = torch.ones_like(adj_changes[0], device=device)
        complementary[target_node] = 0.0
        complementary = (complementary - adj[target_node]) - adj[target_node]
        complementary = complementary.unsqueeze(0)
    adj[target_node, :] = adj[target_node, :] + adj_changes * complementary
    adj[:, target_node] = adj[:, target_node] + torch.squeeze((adj_changes * complementary).t())
    return adj


def difference_of_logits(logits: torch.Tensor, labels: torch.Tensor,
                         labels_infhot: Optional[torch.Tensor] = None) -> torch.Tensor:
    if labels_infhot is None:
        labels_infhot = torch.zeros_like(logits.detach()).scatter(1, labels.unsqueeze(1), float('inf'))
    #     print(logits,labels,labels.unsqueeze(1))
    class_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    other_logits = (logits - labels_infhot).max(1).values
    return class_logits - other_logits


def l0_projection_0(δ: Tensor, ε: Tensor) -> Tensor:
    δ_abs = δ.flatten(1).abs()
    sorted_indices = δ_abs.argsort(dim=1, descending=True).gather(1, (ε.long().unsqueeze(1) - 1).clamp_min(0))
    thresholds = δ_abs.gather(1, sorted_indices)
    return torch.clamp(torch.where((δ_abs >= thresholds).view_as(δ), δ, torch.zeros(1, device=δ.device)), min=0, max=1)


target_or_test = 'target'  # attack target nodes(#250) or  all the test nodes
explore = True  # Use initialization strategy or not
tanh = 1.0
decay = True
device = 0  # GPU
gamma = 0.0
for GNN_name in ['GCN', 'SGC', 'APPNP']:
    for dataset in ['cora', 'citeseer','polblogs']:
        for explore in [True]:
            print("GNN_name:", GNN_name, " dataset:", dataset, " explore:", explore, " decay:", decay)
            norm = 0
            batch_size = 1
            config = Config(dataset)
            config.device = device

            seed = 19
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            data = Data(dataset, dataset + '_ft_norm')

            data.idx_unlabeled = np.union1d(data.idx_val, data.idx_test)

            # Setup Surrogate model
            if GNN_name == 'SGC':
                surrogate = SGC(nfeat=data.features.shape[1], nclass=data.labels.max().item() + 1, device=device)
                surrogate = surrogate.to(device)
                surrogate.fit(data.features_norm.todense(), data.adj.todense(), \
                              data.labels, data.idx_train, data.idx_val, normalize=True, verbose=False)
            if GNN_name == 'GCN':
                # Setup Surrogate model
                surrogate = GCN(nfeat=data.features.shape[1], nclass=data.labels.max().item() + 1,
                                nhid=config.eval_hidden, dropout=config.eval_dropout,
                                weight_decay=config.eval_weight_decay,
                                device=config.device).to(config.device)
                surrogate = surrogate.to(device)
                surrogate.fit(data.features_norm.todense(), data.adj, data.labels, data.idx_train, data.idx_val, normalize=True)
            if GNN_name == 'APPNP':
                # Setup Surrogate model
                surrogate = APPNP(nfeat=data.features.shape[1], nclass=data.labels.max().item() + 1,
                                  device=config.device).to(config.device)
                surrogate = surrogate.to(device)
                surrogate.fit(data.features_norm.todense(), data.adj, data.labels, data.idx_train, data.idx_val, normalize=True)
            surrogate.test(data.idx_test)

            if explore:
                with open('explore/start_' + GNN_name + '_' + dataset + '_' + target_or_test + '_.pkl', 'rb') as f:
                    starts = pkl.load(f)
            if target_or_test == 'test':
                tar_all = data.idx_test.tolist()
            elif target_or_test == 'target':
                tar_all = []
                for tar in data.target_sets:
                    for target_node in tar:
                        tar_all.append(target_node)

            seed = 19
            verbose = False
            nnodes = data.n_nodes
            batch_view = lambda tensor: tensor.view(batch_size, *[1] * (nnodes - 1))
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            surrogate.eval()

            bound = []
            acc_list = []
            ptb_list = []
            order = 0
            log = ""
            epoch_list = []
            import time

            start_time = time.time()
            best_δ_dict = {}
            margin = {}
            for target_node in tqdm(tar_all):
                # for target_node in [ 1554]:
                if verbose:
                    print("*" * 50, target_node)
                α_init = 1.0
                α_final = α_init / 100
                γ_init = 0.05
                γ_final = 0.001
                labels = data.labels[target_node].unsqueeze(0)  # [c1]
                ori_adj, ori_features, labels = utils.to_tensor(data.adj, data.features_norm, labels, device=device)
                adj_changes_org = Parameter(torch.FloatTensor(int(nnodes))).unsqueeze(0).to(device)
                adj_changes_org[0].data.fill_(0.0001)
                if explore:
                    adj_changes_org[0, starts[target_node]] = 0.001
                adj_changes = adj_changes_org * adj_changes_org
                complementary = None
                ε = torch.ones(batch_size, device=device)
                worst_norm = torch.max(adj_changes, 1 - adj_changes).norm(p=norm, dim=1)
                best_norm = worst_norm.clone()

                best_δ = None
                adv_found = False

                logits = surrogate(ori_features, utils.normalize_adj_tensor(ori_adj.to_dense()))[target_node]
                p_y = math.exp(logits[data.labels[target_node]].item())
                logits[data.labels[target_node]] = -1000
                pred_max_c = logits.argmax()
                p_c = math.exp(logits[pred_max_c].item())
                if verbose:
                    print("lalalla", math.exp(logits[pred_max_c].item()), math.exp(logits[labels[0]].item()))
                if p_c - p_y > gamma:
                    best_norm = 0
                    bound.append(0)
                    acc_list.append(True)
                    ptb_list.append(None)
                    best_δ_dict[target_node] = None
                    margin[target_node] = p_c - p_y
                    if verbose:
                        print("@@@check:  target:", target_node, "success:", success, "bound:", best_norm)
                    continue

                if explore:
                    index_adv = torch.LongTensor(
                        [[target_node, starts[target_node]], [starts[target_node], target_node]])
                    values_adv = torch.FloatTensor([-2 * ori_adj[target_node, starts[target_node]] + 1,
                                                    -2 * ori_adj[starts[target_node], target_node] + 1])
                    adj_tmp = torch.sparse.FloatTensor(index_adv, values_adv, torch.Size([nnodes, nnodes])).to(device)
                    logits = surrogate(ori_features, utils.normalize_adj_tensor(ori_adj + adj_tmp, True))[
                        target_node]
                    p_y = math.exp(logits[data.labels[target_node]].item())
                    logits[data.labels[target_node]] = -1000
                    pred_max_c = logits.argmax()
                    p_c = math.exp(logits[pred_max_c].item())
                    if verbose:
                        print("lalalla", math.exp(logits[pred_max_c].item()), math.exp(logits[labels[0]].item()))
                    if p_c - p_y > gamma:
                        best_norm = 1.0
                        bound.append(1.0)
                        acc_list.append(True)
                        ptb_list.append((target_node, starts[target_node]))
                        best_δ = adj_tmp
                        best_δ_dict[target_node] = best_δ
                        margin[target_node] = p_c - p_y
                        if verbose:
                            print("@@@check:  target:", target_node, "success:", success, "bound:", best_norm)
                        continue
                t = 0
                epochs = 800
                patience = 800
                L0_list = []
                while (True):
                    if adv_found:
                        patience = patience - 1
                    if patience == 0:
                        break

                    α = α_init
                    γ = γ_init
                    if adv_found and decay:
                        cosine = (1 + math.cos(math.pi * (epochs - patience) / epochs)) / 2
                        α = α_final + (α_init - α_final) * cosine
                        γ = γ_final + (γ_init - γ_final) * cosine
                        α = α_final + (α_init - α_final) * cosine
                        γ = γ_final + (γ_init - γ_final) * cosine

                    δ_norm = adj_changes.data.norm(p=norm, dim=1)
                    L0_list.append(δ_norm)
                    modified_adj = get_modified_adj(complementary, adj_changes, ori_adj.to_dense(), target_node, device)

                    adj_norm = utils.normalize_adj_tensor(modified_adj)
                    logits = surrogate(ori_features, adj_norm)[target_node].unsqueeze(0)
                    pred_labels = logits.argmax(dim=1)
                    if t == 0:
                        labels_infhot = torch.zeros_like(logits.detach()).scatter(1, labels.unsqueeze(1), float('inf'))
                        logit_diff_func = partial(difference_of_logits, labels=labels, labels_infhot=labels_infhot)
                    loss = -logit_diff_func(logits=logits)
                    #             loss = F.nll_loss(logits, labels)
                    δ_grad = grad(loss.sum(), adj_changes_org, only_inputs=True)[
                        0]  # δ, adv_inputs, logits, logit_diffs, loss

                    if t > 0:
                        cols = adj_changes.data.nonzero()[:, 1].tolist()
                        rows = [target_node] * len(cols)
                        ori_values = (-2 * ori_adj[target_node].to_dense()[cols].cpu().numpy() + 1).tolist()
                        index_adv = torch.LongTensor([rows, cols])
                        values_adv = torch.FloatTensor(ori_values)
                        adj_tmp = torch.sparse.FloatTensor(index_adv, values_adv, torch.Size([nnodes, nnodes])).to(
                            device)
                        adj_tmp = adj_tmp + adj_tmp.t()
                        logits_hard = surrogate(ori_features, utils.normalize_adj_tensor(ori_adj + adj_tmp, True))[
                            target_node]
                        p_y = math.exp(logits_hard[data.labels[target_node]].item())
                        logits_hard[data.labels[target_node]] = -1000
                        pred_max_c_hard = logits_hard.argmax()
                        p_c = math.exp(logits_hard[pred_max_c_hard].item())
                        is_adv = torch.BoolTensor([p_c - p_y > gamma]).to(device)
                        #             is_adv = pred_labels != labels
                        is_smaller = δ_norm < best_norm
                        is_both = is_adv & is_smaller # 又成功 又有更小的扰动
                        adv_found = (adv_found + is_adv) > 0
                        best_norm = torch.where(is_both, δ_norm, best_norm)
                        best_δ = adj_tmp.data if is_both[0] else best_δ
                        if is_both[0]:
                            margin[target_node] = p_c - p_y

                        if δ_norm < 2 and is_adv == True:
                            if verbose:
                                print("early stop since δ_norm:", δ_norm, " is_adv:", is_adv)
                            break
                        if verbose:
                            print(str(target_node) + ":" + "~" * 20 + " δ step:" + "~" * 20)
                            print("current ε:", ε)
                            print("current α(step size for δ):", α)
                            print("current γ(step size for ε):", γ)
                            print("current δ_norm, best_norm:", δ_norm.item(), best_norm.item())
                            print("current soft pred_labels and logits: &adv:", pred_labels.item(),
                                  math.exp(logits[0, pred_labels].item()), " &org:", labels.item(),
                                  math.exp(logits[0, labels[0]].cpu().item()))
                            print("current hard pred_labels and logits: &adv:", pred_labels_hard.item(),
                                  math.exp(logits_hard[0, pred_labels_hard].item()), " &org:", labels.item(),
                                  math.exp(logits_hard[0, labels[0]].cpu().item()))
                            print("(whether update)is_adv & is_smaller:", is_adv.item(), is_smaller.item())

                        if norm == 0:
                            ε = torch.where(is_adv,
                                            torch.min(torch.min(ε - 1, (ε * (1 - γ)).long().float()), best_norm),
                                            torch.max(ε + 1, (ε * (1 + γ)).long().float()))
                            ε.clamp_min_(0)
                        else:
                            distance_to_boundary = loss.detach().abs() / δ_grad.flatten(1).norm(p=dual,
                                                                                                dim=1).clamp_min(1e-12)
                            ε = torch.where(is_adv,
                                            torch.min(ε * (1 - γ), best_norm),
                                            torch.where(adv_found, ε * (1 + γ), δ_norm + distance_to_boundary))

                        # clip ε
                        ε = torch.min(ε, worst_norm)

                    # normalize gradient
                    if verbose:
                        print("~" * 10 + " ε step:" + "~" * 10)
                        print("δ_grad before norm:", δ_grad, δ_grad[0, starts[target_node]])
                    grad_l2_norms = δ_grad.flatten(1).norm(p=2, dim=1).clamp_min(1e-12)
                    δ_grad.div_(grad_l2_norms.view(batch_size, 1))
                    if verbose:
                        print("δ_grad after norm:", δ_grad)

                    # gradient ascent step
                    adj_changes_org.data.add_(δ_grad, alpha=α)
                    adj_changes = torch.tanh(tanh * adj_changes_org * adj_changes_org)

                    # project
                    adj_changes.data = l0_projection_0(adj_changes, ε)
                    #             adj_changes[:,adj_changes.data.nonzero()[:,1]]=1#!!!!!!
                    if verbose:
                        print("△A：", adj_changes.nonzero(), adj_changes[0, adj_changes.nonzero()[:, 1]])
                    t += 1

                bound.append(best_norm.item())
                epoch_list.append(t)
                best_δ_dict[target_node] = best_δ
            print(sum(bound))


