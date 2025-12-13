import os
import time
import math
import json
import random
import numpy as np
import argparse
import torch
import torch.nn as nn

from tqdm import tqdm
from gnn_models_sag import ppi_model
from gnn_data import GNN_DATA
from utils import Metrictor_PPI, print_file
from torch_geometric.loader import LinkNeighborLoader
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='HIGH-PPI_model_training')

parser.add_argument('--ppi_path', default=None, type=str,
                    help="ppi path")
parser.add_argument('--pseq_path', default=None, type=str,
                    help="protein sequence path")
parser.add_argument('--vec_path', default='./protein_info/vec5_CTC.txt', type=str,
                    help='protein sequence vector path')
parser.add_argument('--p_feat_matrix', default=None, type=str,
                    help="protein feature matrix")
parser.add_argument('--p_adj_matrix', default=None, type=str,
                    help="protein adjacency matrix")
parser.add_argument('--index_path', default=None, type=str,
                    help='training and test PPI index')
parser.add_argument('--model_path', default=None, type=str,
                    help="path for trained model")
parser.add_argument('--batch_size', default=None, type=int,
                    help="Batch size")

def slice_and_remap_internal_edges(node_ids, x_num_index, edge_num_index, p_edge_all, p_x_all_sliced, device):
    
    node_ids_d = node_ids.to(device).long()
    x_num_index_d = x_num_index.to(device).long()
    
    num_nodes_expected = p_x_all_sliced.size(0)
    
    residue_offsets = torch.cat([
        torch.tensor([0], dtype=torch.long, device=device), 
        x_num_index_d.cumsum(dim=0)
    ])
    
    protein_ids = torch.searchsorted(residue_offsets[1:], node_ids_d, right=False)
    unique_protein_ids = torch.unique(protein_ids)
    
    edge_offsets = torch.cat([
        torch.tensor([0], dtype=torch.long, device=device), 
        edge_num_index.to(device).long().cumsum(dim=0)
    ])
    
    current_edge_indices = []
    for pid in unique_protein_ids:
        start = edge_offsets[pid]
        end = edge_offsets[pid + 1]
        current_edge_indices.append(p_edge_all[:, start:end])
    
    if len(current_edge_indices) > 0:
        current_edge_index = torch.cat(current_edge_indices, dim=1).long().to(device)
    else:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    
    global_to_local = {global_id: local_id for local_id, global_id in enumerate(node_ids_d.tolist())}
    
    valid_edges = []
    for i in range(current_edge_index.size(1)):
        src_global = current_edge_index[0, i].item()
        dst_global = current_edge_index[1, i].item()
        
        if src_global in global_to_local and dst_global in global_to_local:
            valid_edges.append([global_to_local[src_global], global_to_local[dst_global]])
    
    if len(valid_edges) > 0:
        remapped_edge_index = torch.tensor(valid_edges, dtype=torch.long, device=device).t()
    else:
        remapped_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    
    return remapped_edge_index.contiguous()

def multi2big_x(x_ori):
    x_cat = torch.zeros(1, 7)
    x_num_index = torch.zeros(1553)
    for i in range(1553):
        x_now = torch.tensor(x_ori[i])
        x_num_index[i] = torch.tensor(x_now.size(0))
        x_cat = torch.cat((x_cat, x_now), 0)
    return x_cat[1:, :], x_num_index

def multi2big_batch(x_num_index):
    num_sum = x_num_index.sum()
    num_sum = num_sum.int()
    batch = torch.zeros(num_sum)
    count = 1
    for i in range(1,1553):
        zj1 = x_num_index[:i]
        zj11 = zj1.sum()
        zj11 = zj11.int()
        zj22 = zj11 + x_num_index[i]
        zj22 = zj22.int()
        size1 = x_num_index[i]
        size1 = size1.int()
        tc = count * torch.ones(size1)
        batch[zj11:zj22] = tc
        test = batch[zj11:zj22]
        count = count + 1
    batch = batch.int()
    return batch

def multi2big_edge(edge_ori, num_index):
    edge_cat = torch.zeros(2, 1)
    edge_num_index = torch.zeros(1553)
    for i in range(1553):
        edge_index_p = edge_ori[i]
        edge_index_p = np.asarray(edge_index_p)
        edge_index_p = torch.tensor(edge_index_p.T, dtype=torch.long)
        edge_num_index[i] = torch.tensor(edge_index_p.size(1))
        if i == 0:
            offset = 0
        else:
            zj = (num_index[:i])
            offset = zj.sum().item()
        edge_cat = torch.cat((edge_cat, edge_index_p + offset), 1)
    return edge_cat[:, 1:], edge_num_index


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def test(model, graph, test_mask, train_loader, device,batch, p_x_all, p_edge_all, x_num_index, edge_num_index, batch_size):
    valid_pre_result_list = []
    valid_label_list = []

    model.eval()

    #batch_size = 8

    valid_steps = len(train_loader)
    
    valid_pre_result_list = []
    valid_label_list = []
    true_prob_list = []
    with torch.no_grad():
        for step, sampled_data in enumerate(train_loader):
            sampled_data=sampled_data.to(device)
            node_ids=sampled_data.n_id.to('cpu')
            batch_sliced=batch[node_ids]
            p_x_all_sliced=p_x_all[node_ids,:]

            batch_sliced=batch_sliced.to(device)
            p_x_all_sliced=p_x_all_sliced.to(device)
            p_edge_all_sliced = slice_and_remap_internal_edges(node_ids, x_num_index, edge_num_index, p_edge_all, p_x_all_sliced, device)
            num_nodes_in_batch = p_x_all_sliced.size(0)

            edge_index_cpu = p_edge_all_sliced.cpu()

            # output = model(graph.x, graph.edge_index, valid_edge_id)
            output = model(batch_sliced, p_x_all_sliced, p_edge_all_sliced, sampled_data.edge_index, sampled_data.edge_label_index)
            label = sampled_data.edge_label.type(torch.FloatTensor).to(device)


            m = nn.Sigmoid()
            pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

            valid_pre_result_list.append(pre_result.cpu().data)
            valid_label_list.append(label.cpu().data)
            true_prob_list.append(m(output).cpu().data)

    valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
    valid_label_list = torch.cat(valid_label_list, dim=0)
    true_prob_list = torch.cat(true_prob_list, dim = 0)
    metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list, true_prob_list)

    metrics.show_result()

    print('recall: {}, precision: {}, F1: {}, AUPRC: {}'.format(metrics.Recall, metrics.Precision, \
        metrics.F1, metrics.Aupr))
    print(valid_pre_result_list)
    print(valid_label_list)

def main():

    args = parser.parse_args()

    ppi_data = GNN_DATA(ppi_path=args.ppi_path)
    ppi_data.get_feature_origin(pseq_path=args.pseq_path,
                                vec_path=args.vec_path)


    ppi_data.generate_data()

    graph = ppi_data.data
    temp = graph.edge_index.transpose(0, 1).numpy()
    ppi_list = []

    for edge in temp:
        ppi_list.append(list(edge))

    truth_edge_num = len(ppi_list) // 2
    # fake_edge_num = len(ppi_data.fake_edge) // 2
    fake_edge_num = 0
    index_path = args.index_path
    with open(index_path, 'r') as f:
        index_dict = json.load(f)
        f.close()
    graph.train_mask = index_dict['train_index']

    graph.val_mask = index_dict['valid_index']

    print("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))
    graph.edge_index=graph.edge_index.contiguous()

    if hasattr(graph,'x') and graph.x is not None:
        graph.x =graph.x.contiguous()
    graph=graph.to('cpu')

    node_vision_dict = {}
    for index in graph.train_mask:
        ppi = ppi_list[index]
        if ppi[0] not in node_vision_dict.keys():
            node_vision_dict[ppi[0]] = 1
        if ppi[1] not in node_vision_dict.keys():
            node_vision_dict[ppi[1]] = 1

    for index in graph.val_mask:
        ppi = ppi_list[index]
        if ppi[0] not in node_vision_dict.keys():
            node_vision_dict[ppi[0]] = 0
        if ppi[1] not in node_vision_dict.keys():
            node_vision_dict[ppi[1]] = 0
    
    vision_num = 0
    unvision_num = 0
    for node in node_vision_dict:
        if node_vision_dict[node] == 1:
            vision_num += 1
        elif node_vision_dict[node] == 0:
            unvision_num += 1
    print("vision node num: {}, unvision node num: {}".format(vision_num, unvision_num))

    test1_mask = []
    test2_mask = []
    test3_mask = []

    for index in graph.val_mask:
        ppi = ppi_list[index]
        temp = node_vision_dict[ppi[0]] + node_vision_dict[ppi[1]]
        if temp == 2:
            test1_mask.append(index)
        elif temp == 1:
            test2_mask.append(index)
        elif temp == 0:
            test3_mask.append(index)
    print("test1 edge num: {}, test2 edge num: {}, test3 edge num: {}".format(len(test1_mask), len(test2_mask), len(test3_mask)))

    graph.test1_mask = test1_mask
    graph.test2_mask = test2_mask
    graph.test3_mask = test3_mask

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    p_x_all_list_of_np = torch.load(args.p_feat_matrix, weights_only=False)
    p_x_all_tensors = []
    for p in p_x_all_list_of_np:
        
        tensor_p = torch.from_numpy(p).float()
        p_x_all_tensors.append(tensor_p)

    x_num_index = torch.tensor([p.size(0) for p in p_x_all_tensors], dtype=torch.long)
    #print(f"Final check on x_num_index sum: {x_num_index.sum().item()}")
    
    p_x_all = torch.cat(p_x_all_tensors, dim=0)
    # p_edge_all = np.load('/apdcephfs/share_1364275/kaithgao/edge_list_12.npy', allow_pickle=True)
    p_edge_all = np.load(args.p_adj_matrix, allow_pickle=True)
    #p_x_all, x_num_index = multi2big_x(p_x_all)
    # p_x_all = p_x_all[:,torch.arange(p_x_all.size(1))!=6] 
    batch = multi2big_batch(x_num_index)+1
    p_edge_all, edge_num_index = multi2big_edge(p_edge_all, x_num_index)
    
    #graph.val_mask = ppi_data.ppi_split_dict['valid_index']
    valid_edge_index=graph.edge_index[:,torch.tensor(graph.val_mask, dtype=torch.long)]
    valid_labels=graph.edge_attr_1[graph.val_mask].type(torch.float32)
    
    NUM_NEIGHBOURS=[20,20]
    valid_loader=LinkNeighborLoader(graph, edge_label_index=valid_edge_index, edge_label=valid_labels, num_neighbors=NUM_NEIGHBOURS, batch_size=args.batch_size, shuffle= False,num_workers=4)



    
    model = ppi_model()
    model.to(device)
    model_path = args.model_path
    model.load_state_dict(torch.load(model_path, weights_only=False)['state_dict'])

    test(model, graph, graph.val_mask, valid_loader,device,batch, p_x_all, p_edge_all,x_num_index,edge_num_index,args.batch_size)
    
if __name__ == "__main__":
    main()
