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
parser.add_argument('--index_path', default=None, type=str,
                    help='training and test PPI index')
parser.add_argument('--model_path', default=None, type=str,
                    help="path for trained model")
parser.add_argument('--batch_size', default=None, type=int,
                    help="Batch size")
parser.add_argument('--precomputed_esm', default='/users/cnaraya2/ppi/high-ppi/protein_info/SHS27k_embedding/embedding.pt', type=str, help='path to pre-computed ESM embeddings')

def get_unique_proteins_from_edges(edge_index):
    all_proteins = torch.cat([edge_index[0], edge_index[1]])
    unique_proteins = torch.unique(all_proteins)
    return unique_proteins


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def test(model, graph, test_mask, valid_loader, device, esm_embeddings, batch_size):
    valid_pre_result_list = []
    valid_label_list = []
    true_prob_list = []
    valid_loss_sum = 0.0

    model.eval()
    valid_steps = len(valid_loader)

    with torch.no_grad():
        for step, sampled_data in enumerate(valid_loader):
            sampled_data = sampled_data.to(device)
            
            unique_proteins = get_unique_proteins_from_edges(sampled_data.edge_index)
            batch_esm_embeddings = esm_embeddings[unique_proteins]
            
            output = model(
                edge_index=sampled_data.edge_index,
                train_edge_id=sampled_data.edge_label_index,
                esm_embeddings=batch_esm_embeddings
            )
            
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

    valid_edge_index=graph.edge_index[:,torch.tensor(graph.val_mask, dtype=torch.long)]
    valid_labels=graph.edge_attr_1[graph.val_mask].type(torch.float32)
    
    NUM_NEIGHBOURS=[20,20]
    valid_loader=LinkNeighborLoader(graph, edge_label_index=valid_edge_index, edge_label=valid_labels, num_neighbors=NUM_NEIGHBOURS, batch_size=args.batch_size, shuffle= False,num_workers=4)

    esm_embeddings = torch.load(args.precomputed_esm).to(device)
    print(f"Loaded embeddings shape: {esm_embeddings.shape}")

    
    model = ppi_model()
    model.to(device)
    model_path = args.model_path
    model.load_state_dict(torch.load(model_path, weights_only=False)['state_dict'])

    test(model, graph, graph.val_mask, valid_loader,device,esm_embeddings,args.batch_size)
    
if __name__ == "__main__":
    main()
