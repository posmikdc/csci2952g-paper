import os
import time
import math
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
from gnn_data import GNN_DATA
from gnn_models_sag import ppi_model
from utils import Metrictor_PPI, print_file
from tensorboardX import SummaryWriter
from torch_geometric.loader import LinkNeighborLoader

torch.cuda.empty_cache()
parser = argparse.ArgumentParser(description='HIGH-PPI_model_training')

parser.add_argument('--ppi_path', default=None, type=str, help="ppi path")
parser.add_argument('--pseq_path', default=None, type=str, help="protein sequence path")
parser.add_argument('--vec_path', default='./protein_info/vec5_CTC.txt', type=str, help='protein sequence vector path')
parser.add_argument('--split', default=None, type=str, help='split method, random, bfs or dfs')
parser.add_argument('--save_path', default=None, type=str, help="save folder")
parser.add_argument('--epoch_num', default=None, type=int, help='train epoch number')
parser.add_argument('--batch_size', default=None, type=int, help='batch size')
parser.add_argument('--precomputed_esm', default='/users/cnaraya2/ppi/high-ppi/protein_info/SHS27k_embedding/embedding.pt', type=str, help='path to pre-computed ESM embeddings')

seed_num = 2
np.random.seed(seed_num)
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)


def get_unique_proteins_from_edges(edge_index):
    all_proteins = torch.cat([edge_index[0], edge_index[1]])
    unique_proteins = torch.unique(all_proteins)
    return unique_proteins


def train(model, train_loader, valid_loader, graph, ppi_list, loss_fn, optimizer, device,
          result_file_path, summary_writer, save_path, esm_embeddings,
          batch_size=8, epochs=1000, scheduler=None):
    global_step = 0
    global_best_valid_f1 = 0.0
    global_best_valid_f1_epoch = 0
    
    for epoch in range(epochs):
        recall_sum = 0.0
        precision_sum = 0.0
        f1_sum = 0.0
        loss_sum = 0.0

        model.train()
        steps = len(train_loader)

        for step, sampled_data in enumerate(train_loader):
            sampled_data = sampled_data.to(device)
            
            unique_proteins = get_unique_proteins_from_edges(sampled_data.edge_index)
            batch_esm_embeddings = esm_embeddings[unique_proteins]
            
            output = model(
                edge_index=sampled_data.edge_index,
                train_edge_id=sampled_data.edge_label_index,
                esm_embeddings=batch_esm_embeddings
            )

            label = sampled_data.edge_label.type(torch.FloatTensor).to(device)
            loss = loss_fn(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m = nn.Sigmoid()
            pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

            metrics = Metrictor_PPI(pre_result.cpu().data, label.cpu().data, m(output).cpu().data)
            metrics.show_result()

            recall_sum += metrics.Recall
            precision_sum += metrics.Precision
            f1_sum += metrics.F1
            loss_sum += loss.item()

            summary_writer.add_scalar('train/loss', loss.item(), global_step)
            summary_writer.add_scalar('train/precision', metrics.Precision, global_step)
            summary_writer.add_scalar('train/recall', metrics.Recall, global_step)
            summary_writer.add_scalar('train/F1', metrics.F1, global_step)

            global_step += 1
            
            if step % 10 == 0:
                print_file("epoch: {}, step: {}/{}, Train: loss: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}"
                           .format(epoch, step, steps, loss.item(), metrics.Precision, metrics.Recall, metrics.F1))

        torch.save({'epoch': epoch, 'state_dict': model.state_dict()},
                   os.path.join(save_path, 'gnn_model_train.ckpt'))

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
                loss = loss_fn(output, label)
                valid_loss_sum += loss.item()

                m = nn.Sigmoid()
                pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

                valid_pre_result_list.append(pre_result.cpu().data)
                valid_label_list.append(label.cpu().data)
                true_prob_list.append(m(output).cpu().data)

        valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
        valid_label_list = torch.cat(valid_label_list, dim=0)
        true_prob_list = torch.cat(true_prob_list, dim=0)

        metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list, true_prob_list)
        metrics.show_result()

        recall = recall_sum / steps
        precision = precision_sum / steps
        f1 = f1_sum / steps
        loss = loss_sum / steps
        valid_loss = valid_loss_sum / valid_steps

        if scheduler != None:
            scheduler.step(loss)
            print_file("epoch: {}, now learning rate: {}".format(epoch, scheduler.optimizer.param_groups[0]['lr']),
                       save_file_path=result_file_path)

        if global_best_valid_f1 < metrics.F1:
            global_best_valid_f1 = metrics.F1
            global_best_valid_f1_epoch = epoch

            torch.save({'epoch': epoch, 'state_dict': model.state_dict()},
                       os.path.join(save_path, 'gnn_model_valid_best.ckpt'))

        summary_writer.add_scalar('valid/precision', metrics.Precision, global_step)
        summary_writer.add_scalar('valid/recall', metrics.Recall, global_step)
        summary_writer.add_scalar('valid/F1', metrics.F1, global_step)
        summary_writer.add_scalar('valid/loss', valid_loss, global_step)

        print_file(
            "epoch: {}, Training_avg: loss: {:.4f}, recall: {:.4f}, precision: {:.4f}, F1: {:.4f}, "
            "Validation_avg: loss: {:.4f}, recall: {:.4f}, precision: {:.4f}, F1: {:.4f}, "
            "Best valid_f1: {:.4f}, in epoch {}"
            .format(epoch, loss, recall, precision, f1, valid_loss, metrics.Recall, metrics.Precision, metrics.F1,
                    global_best_valid_f1, global_best_valid_f1_epoch), save_file_path=result_file_path)


def main():
    args = parser.parse_args()
    
    ppi_data = GNN_DATA(ppi_path=args.ppi_path)
    ppi_data.get_feature_origin(pseq_path=args.pseq_path, vec_path=args.vec_path)
    ppi_data.generate_data()
    ppi_data.split_dataset(train_valid_index_path='./train_val_split_data/train_val_split_1.json', 
                           random_new=True, mode=args.split)
    
    graph = ppi_data.data
    ppi_list = ppi_data.ppi_list

    graph.train_mask = ppi_data.ppi_split_dict['train_index']
    graph.val_mask = ppi_data.ppi_split_dict['valid_index']
    graph.edge_index = graph.edge_index.contiguous()

    if hasattr(graph, 'x') and graph.x is not None:
        graph.x = graph.x.contiguous()
    graph = graph.to('cpu')

    train_edge_index = graph.edge_index[:, graph.train_mask]
    train_labels = graph.edge_attr_1[graph.train_mask].type(torch.float32)

    NUM_NEIGHBOURS = [20, 20]
    train_loader = LinkNeighborLoader(
        graph, 
        edge_label_index=train_edge_index, 
        edge_label=train_labels, 
        num_neighbors=NUM_NEIGHBOURS, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4
    )
    
    valid_edge_index = graph.edge_index[:, graph.val_mask]
    valid_labels = graph.edge_attr_1[graph.val_mask].type(torch.float32)
    valid_loader = LinkNeighborLoader(
        graph, 
        edge_label_index=valid_edge_index, 
        edge_label=valid_labels, 
        num_neighbors=NUM_NEIGHBOURS, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4
    )

    
    print("train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    
    esm_embeddings = torch.load(args.precomputed_esm).to(device)
    print(f"Loaded embeddings shape: {esm_embeddings.shape}")

    model = ppi_model()
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    POS_WEIGHT = torch.tensor([10.0]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(POS_WEIGHT).to(device)

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(save_path, "esm_gat_emb-128_b-64_e-500")
    os.mkdir(save_path)
    
    result_file_path = os.path.join(save_path, "valid_results.txt")
    summary_writer = SummaryWriter(save_path)

    
    print(f"Results will be saved to: {save_path}")

    train(
        model, train_loader, valid_loader, graph, ppi_list, 
        loss_fn, optimizer, device, result_file_path, summary_writer, save_path, 
        esm_embeddings, batch_size=args.batch_size, epochs=args.epoch_num, scheduler=scheduler
    )

    summary_writer.close()
    


if __name__ == "__main__":
    main()