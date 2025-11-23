'''

'''
import numpy as np

import torch
import torch.nn as nn
from torch.nn import ParameterList, Parameter
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.conv import GATConv
from torch_scatter import scatter_add
import math


num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)
        self.edge_embedding3 = torch.nn.Embedding(2, emb_dim)  # is_in_ring (0 or 1)
        self.edge_embedding4 = torch.nn.Embedding(2, emb_dim)  # is_conjugated (0 or 1)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding4.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 4)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1]) + \
                         self.edge_embedding3(edge_attr[:, 2]) + self.edge_embedding4(edge_attr[:, 3])

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin", extra_feature_dim=0):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.extra_feature_dim = extra_feature_dim

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)
        self.x_embedding3 = torch.nn.Embedding(11, emb_dim)  # degree (0-10)
        self.x_embedding4 = torch.nn.Embedding(11, emb_dim)  # formal_charge (-5 to +5, index 0-10)
        self.x_embedding5 = torch.nn.Embedding(9, emb_dim)   # num_h (0-8)
        self.x_embedding6 = torch.nn.Embedding(7, emb_dim)   # hybridization (7 types)
        self.x_embedding7 = torch.nn.Embedding(2, emb_dim)   # is_aromatic (0 or 1)
        
        # Linear projection for extra continuous features
        if self.extra_feature_dim > 0:
            self.feat_bn = torch.nn.BatchNorm1d(self.extra_feature_dim)
            self.feat_proj = torch.nn.Linear(self.extra_feature_dim, emb_dim)
            torch.nn.init.xavier_uniform_(self.feat_proj.weight.data)
            self.feat_proj.bias.data.zero_()

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding5.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding6.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding7.weight.data)
        self._initialize_weights()

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINConv(emb_dim, aggr="add"))


        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        if JK == 'w_sum':
            initial_scalar_parameters = [0.0] * (num_layer + 1)
            self.para = ParameterList(
                [
                    Parameter(
                        torch.FloatTensor([initial_scalar_parameters[i]]), requires_grad=True
                    )
                    for i in range(num_layer + 1)
                ])

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")
            
        # Split input x into categorical (first 7) and continuous (rest)
        # x is FloatTensor now, so we need to cast the first 7 columns to Long for embedding lookup
        x_cat = x[:, :7].long()
        
        x_emb = self.x_embedding1(x_cat[:, 0]) + self.x_embedding2(x_cat[:, 1]) + self.x_embedding3(x_cat[:, 2]) + \
            self.x_embedding4(x_cat[:, 3]) + self.x_embedding5(x_cat[:, 4]) + self.x_embedding6(x_cat[:, 5]) + \
            self.x_embedding7(x_cat[:, 6])
            
        # Add projected extra features if they exist
        if self.extra_feature_dim > 0:
            x_cont = x[:, 7:]
            x_cont = self.feat_bn(x_cont)
            x_emb = x_emb + self.feat_proj(x_cont)

        h_list = [x_emb]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)
        else:
            raise ValueError("unmatched argument.")


        return node_representation

    def _initialize_weights(self):
    # https://github.com/facebookresearch/deepcluster/blob/main/models/alexnet.py
    # nn.Embedding is initialized by xavier_uniform_ from the original gnn code    
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class GNN_graphpred(torch.nn.Module):

    def __init__(self, args):
        super(GNN_graphpred, self).__init__()
        self.num_layer = args.num_layer
        self.drop_ratio = args.dropout_ratio
        self.JK = args.JK
        self.emb_dim = args.emb_dim
        self.num_tasks = args.num_tasks
        self.graph_pooling = args.graph_pooling
        self.gnn_type = args.gnn_type

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(self.num_layer, self.emb_dim, self.JK, self.drop_ratio, gnn_type=self.gnn_type)

        # Different kind of graph pooling
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * self.emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(self.emb_dim, 1))
        elif self.graph_pooling[:-1] == "set2set":
            set2set_iter = int(self.graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * self.emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(self.emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        if self.graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        # self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')), strict=False)

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        return self.graph_pred_linear(self.pool(node_representation[-1], batch))

    def get_graph_rep(self, *argv ):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")
        node_representation = self.gnn(x, edge_index, edge_attr)
        return self.pool(node_representation, batch)
 

class gate(torch.nn.Module):  
    def __init__(self, emb_dim, gate_dim=300):
        super(gate, self).__init__()
        self.linear1 = nn.Linear(emb_dim, gate_dim)
        self.batchnorm = nn.BatchNorm1d(gate_dim)
        self.linear2 = nn.Linear(gate_dim, gate_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        gate_emb = self.linear2(x)
        return gate_emb  


class expert(torch.nn.Module): 
    def __init__(self, channel, num_tasks):
        super(expert, self).__init__()
        self.clf = nn.Linear(channel, num_tasks)

    def forward(self, x):
        x = self.clf(x)
        return x    


class GNN_topexpert(torch.nn.Module): # expert 를 parallel 하게

    def __init__(self, args, criterion):
        super(GNN_topexpert, self).__init__()
        self.num_layer = args.num_layer
        self.drop_ratio = args.dropout_ratio
        self.JK = args.JK
        self.emb_dim = args.emb_dim
        self.num_tasks = args.num_tasks
        self.num_classes = args.num_classes
        self.graph_pooling = args.graph_pooling
        self.gnn_type = args.gnn_type
        self.extra_feature_dim = getattr(args, 'extra_feature_dim', 0)

        self.gate = gate(args.emb_dim, args.gate_dim)
        self.cluster = nn.Parameter(torch.Tensor(args.num_experts, args.gate_dim))
        torch.nn.init.xavier_normal_(self.cluster.data)
        ## optimal transport
        self.scf_emb = nn.Parameter(torch.Tensor(args.num_tr_scf, args.gate_dim))
        torch.nn.init.xavier_normal_(self.scf_emb.data)
        self.cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    
        self.T = 10 
        self.criterion = criterion
        self.num_experts = args.num_experts

        # Binary (num_classes=2): num_tasks * num_experts (기존과 동일)
        # Multi-class (num_classes>2): num_classes * num_tasks * num_experts
        # Regression (num_classes=1): num_tasks * num_experts (기존과 동일)
        if self.num_classes > 2:
            # Multi-class classification
            output_dim = self.num_classes * self.num_tasks * self.num_experts
        else:
            # Binary classification or regression
            output_dim = self.num_tasks * self.num_experts
            
        self.experts_w = nn.Parameter(torch.empty(self.emb_dim, output_dim))
        self.experts_b =  nn.Parameter(torch.empty(output_dim))
        self.reset_experts()

        self.gate_pool = global_add_pool
        
    
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(self.num_layer, self.emb_dim, self.JK, self.drop_ratio, gnn_type=self.gnn_type, extra_feature_dim=self.extra_feature_dim)
        
        # Different kind of graph pooling
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * self.emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(self.emb_dim, 1))
        elif self.graph_pooling[:-1] == "set2set":
            set2set_iter = int(self.graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * self.emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(self.emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")


    def reset_experts(self):
        torch.nn.init.kaiming_uniform_(self.experts_w, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.emb_dim) 
        torch.nn.init.uniform_(self.experts_b, -bound, bound)
        

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')), strict=False)


    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        node_rep = self.gnn(x, edge_index, edge_attr)
        gnn_out = self.pool(node_rep, batch)
        gate_input = self.gate_pool(node_rep, batch)

        ## multi-head mlps
        gnn_out = torch.unsqueeze(gnn_out, -1) # N x emb_dim x 1
        
        if self.num_classes > 2:
            # Multi-class: output num_classes * tasks * experts
            output_size = self.num_classes * self.num_tasks * self.num_experts
            gnn_out = gnn_out.repeat(1, 1, output_size) # N x emb_dim x (num_classes * tasks * experts)
            clf_logit = torch.sum(gnn_out * self.experts_w, dim=1) + self.experts_b # N x (num_classes * tasks * experts)
            clf_logit = clf_logit.view(-1, self.num_classes, self.num_tasks, self.num_experts) # N x num_classes x tasks x num_experts
        else:
            # Binary classification or regression
            output_size = self.num_tasks * self.num_experts
            gnn_out = gnn_out.repeat(1, 1, output_size) # N x emb_dim x (tasks * experts)
            clf_logit = torch.sum(gnn_out * self.experts_w, dim=1) + self.experts_b # N x (tasks * experts)
            clf_logit = clf_logit.view(-1, self.num_tasks, self.num_experts) # N x tasks x num_experts
        
        
        z = self.gate(gate_input)
        q = self.get_q(z)

        return clf_logit, z, q # N x tasks x heads (binary/reg) or N x num_classes x tasks x heads (multi-class)


    def assign_head(self, q):  
        
        q_idx = torch.argmax(q, dim=-1)  # N x 1     
        if self.training:       
            g = F.gumbel_softmax((q + 1e-10).log(), tau=10, hard=False, dim=-1)       
            g = torch.unsqueeze(g, 1)
            g = g.repeat(1, self.num_tasks, 1) # N x tasks x heads
            return g,  q_idx # N x tasks x heads // N // N 
        else:
            q = torch.unsqueeze(q, 1)
            q = q.repeat(1, self.num_tasks, 1) # N x tasks x heads
            return q, q_idx # N x tasks x heads // N // N 

    def clf_loss(self, clf_outs, labels, assign, task_type='classification', num_classes=2): 

        if task_type == 'classification' and num_classes > 2:
            # Multi-class: clf_outs shape: N x num_classes x tasks x num_experts
            # labels shape: N x tasks (with integer class indices 0, 1, 2, ...)
            N, C, T, E = clf_outs.shape
            
            # Validity check for multi-class (all non-negative integers are valid)
            is_valid = labels >= 0
            is_valid_tensor = is_valid.unsqueeze(-1).repeat(1, 1, E)  # N x tasks x experts
            
            # Calculate loss for each task and expert
            loss_tensor = torch.zeros(N, T, E).to(clf_outs.device)
            
            for t in range(T):
                labels_t = labels[:, t].long()  # N
                for e in range(E):
                    # clf_outs[:, :, t, e] is N x num_classes (logits)
                    # labels_t is N (class indices)
                    loss_tensor[:, t, e] = F.cross_entropy(
                        clf_outs[:, :, t, e], 
                        labels_t, 
                        reduction='none'
                    )
        else:
            # Binary classification or regression
            # clf_outs shape: N x tasks x num_experts
            
            if task_type == 'classification':
                # Binary: labels are 0/1. 
                # Previous logic (labels**2 > 0) assumed -1/1 labels where 0 was missing.
                # Now we assume 0/1 are valid. We assume -1 (or < 0) is missing if any.
                is_valid = labels >= 0
            else:
                # Regression
                is_valid = torch.ones_like(labels, dtype=torch.bool)

            is_valid_tensor = is_valid.unsqueeze(-1).repeat(1, 1, self.num_experts)
            
            labels_expanded = labels.unsqueeze(-1).repeat(1, 1, self.num_experts)  # N x tasks x experts
            
            if task_type == 'classification':
                # Binary: labels are 0/1, so use them directly
                loss_tensor = self.criterion(clf_outs, labels_expanded)
            else:
                # Regression: use labels as-is (NO CONVERSION)
                loss_tensor = self.criterion(clf_outs, labels_expanded)

        #### modify loss --> assign 0 to invalid labels    
        loss_tensor_valid = torch.where(is_valid_tensor, loss_tensor, torch.zeros_like(loss_tensor))

        #### modify loss based on assign index 
        loss_mat = torch.sum(assign * loss_tensor_valid, dim=0)  # tasks x experts
        num_valid_mat = torch.sum(assign * is_valid_tensor.long(), dim=0)  # tasks x experts
        
        return loss_mat, (num_valid_mat + 1e-10)  # tasks x experts

    def get_q(self, z):

        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster, 2), 2) )
        q = q.pow((1.0 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def target_distribution(self, q):
        weight = q**2 / q.sum(0)
        p = (weight.t() / weight.sum(1)).t()
        return p

    def alignment_loss(self, scf_idx, q):
        e = self.scf_emb[scf_idx] 
        e = e.unsqueeze(dim=-1)
        mu = torch.transpose(self.cluster, 1, 0).unsqueeze(dim=0)
        loss = torch.mean(torch.sum(q * (1 - self.cos_similarity(e, mu)), dim=1))
        return loss
    

if __name__ == "__main__":
    pass
