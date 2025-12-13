import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
# from pygcn.layers import GraphConvolution
# from dgl.nn import GraphConv, EdgeWeightNorm
from torch_geometric.nn import GINConv, JumpingKnowledge, global_mean_pool, SAGEConv, GCNConv
from torch_geometric.nn.pool import SAGPooling
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATConv




class GIN(torch.nn.Module):
    def __init__(self,  hidden=512, train_eps=True, class_num=7):  #gat3-e-100-batch-16-h-8-128
        super(GIN, self).__init__()
        self.train_eps = train_eps
        self.gin_conv1 = GINConv(
            nn.Sequential(
                nn.Linear(128, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                # nn.Linear(hidden, hidden),
                # nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        self.gin_conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                # nn.Linear(hidden, hidden),
                # nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        self.gin_conv3 = GINConv(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )

        self.lin1 = nn.Linear(hidden, hidden)
        self.fc1 = nn.Linear(2 * hidden, 7) #clasifier for concat
        self.fc2 = nn.Linear(hidden, 7)   #classifier for inner product



    def reset_parameters(self):

        self.fc1.reset_parameters()

        self.gin_conv1.reset_parameters()
        self.gin_conv2.reset_parameters()
        # self.gin_conv3.reset_parameters()
        self.lin1.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


    def forward(self, x, edge_index, train_edge_id, p=0.5):

        x = self.gin_conv1(x, edge_index)
        x = self.gin_conv2(x, edge_index)
        # x = self.gin_conv3(x, edge_index)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        node_id = edge_index[:, train_edge_id]
        x1 = x[node_id[0]]
        x2 = x[node_id[1]]
        #x1 = x[train_edge_id[0]]
        #x2 = x[train_edge_id[1]]
        # x = torch.cat([x1, x2], dim=1)
        # x = self.fc1(x)
        x = torch.mul(x1, x2)
        x = self.fc2(x)
        

        return x

class GAT_BGNN(nn.Module):
    def __init__(self):
        super(GAT_BGNN,self).__init__()
        in_channels=7
        hidden_channels=256
        heads=8

        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.4)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.4)
        

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.lin1.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self,x,edge_index,batch=None):
        x=F.dropout(x,p=0.4, training=self.training)
        x=self.conv1(x, edge_index)
        x=F.elu(x)
        x=F.dropout(x,p=0.4, training=self.training)
        x=self.conv2(x, edge_index)
        return x


class GAT_TGNN(nn.Module): #last run-e-30h-8-128
    def __init__(self):
        super(GAT_TGNN,self).__init__()
        in_channels=128
        hidden_channels=512
        heads=8

        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.4)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.4)
        #self.conv3 = GATConv(hidden_channels, hidden_channels, heads=2, concat=False, dropout=0.2)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc1 = nn.Linear(2 * hidden_channels, 7) 
        self.fc2 = nn.Linear(hidden_channels, 7)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        
        self.lin1.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()



    def forward(self, x, edge_index, train_edge_id, p=0.4):
        x=F.dropout(x,p, training=self.training)
        x=self.conv1(x, edge_index)
        x=F.elu(x)
        x=F.dropout(x,p, training=self.training)
        x=self.conv2(x, edge_index)
        
        x = F.relu(self.lin1(x))
        x1 = x[train_edge_id[0]]
        x2 = x[train_edge_id[1]]
        x = torch.mul(x1, x2)
        x = self.fc2(x)
        return x




class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        hidden = 128
        self.conv1 = GCNConv(7, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.conv4 = GCNConv(hidden, hidden)
  
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.bn4 = nn.BatchNorm1d(hidden)

        self.sag1 = SAGPooling(hidden,0.5)
        self.sag2 = SAGPooling(hidden,0.5)
        self.sag3 = SAGPooling(hidden,0.5)
        self.sag4 = SAGPooling(hidden,0.5)

        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)

        self.dropout = nn.Dropout(0.5)
        for param in self.parameters():
            print(type(param), param.size())


    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.fc1(x)
        x = F.relu(x) 
        x = self.bn1(x)
        y = self.sag1(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1] 

        x = self.conv2(x, edge_index)
        x = self.fc2(x)
        x = F.relu(x) 
        x = self.bn2(x)
        y = self.sag2(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]  
        
        x = self.conv3(x, edge_index)
        x = self.fc3(x)
        x = F.relu(x) 
        x = self.bn3(x)
        y = self.sag3(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        x = self.conv4(x, edge_index)
        x = self.fc4(x)
        x = F.relu(x) 
        x = self.bn4(x)
        y = self.sag4(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        # y = self.sag4(x, edge_index, batch = batch)

        return global_mean_pool(y[0], y[3])
        # return y[0]
class ESM2(nn.Module):
    def __init__(self):
        super(ESM2, self).__init__()
        #self.esm=torch.load('/users/cnaraya2/ppi/HIGH-PPI/protein_info/SHS27k embedding/embedding.pt').to(device)
        input_dim=128
        self.fc1=nn.Linear(480, input_dim)
        self.fc2=nn.Linear(input_dim, input_dim)
        self.bn1=nn.BatchNorm1d(input_dim)
        self.bn2=nn.BatchNorm1d(input_dim)
        self.drop=nn.Dropout(0.4)

    def forward(self,esm):
        x=self.fc1(esm)
        x=self.bn1(x)
        x=F.relu(x)
        x=self.drop(x)
        x=self.fc2(x)
        x=self.bn2(x)
        x=F.relu(x)
        return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class ppi_model(nn.Module):
    def __init__(self):
        super(ppi_model,self).__init__()
        self.BGNN =GCN()
        self.TGNN =GIN()
    
    def forward(self, batch, p_x_all, p_edge_all, edge_index, train_edge_id, p=0.5):
        edge_index = edge_index.to(device)
        batch = batch.to(torch.int64).to(device)
        x = p_x_all.to(torch.float32).to(device)
        edge = p_edge_all.to(torch.long).to(device)
        embs = self.BGNN(x, edge, batch)
        #print(embs.shape)
        #esm=torch.load('/users/cnaraya2/ppi/HIGH-PPI/protein_info/SHS27k embedding/embedding.pt').to(device)
        #embs=self.BGNN(esm_embeddings)
        final = self.TGNN(embs, edge_index, train_edge_id, p=0.5)
        return final
    '''
    def forward(self, edge_index, train_edge_id,esm_embeddings, p=0.5):
    
        edge_index = edge_index.to(device)
        embs=self.BGNN(esm_embeddings)
        final = self.TGNN(embs, edge_index, train_edge_id, p=0.5)
        return final
    '''