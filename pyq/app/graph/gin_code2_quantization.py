import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import ModuleList, BatchNorm1d, Sequential, Embedding

from tqdm import tqdm

from torch_geometric.loader import DataLoader
from torch_geometric.nn import (GlobalAttention, MessagePassing, Set2Set, global_add_pool, global_max_pool,
                                global_mean_pool, Aggregation)
from torch_geometric.utils import degree
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from torchvision import transforms
from pyq.core.parser import TorchModelParser
from pyq.core.quantization.observer import UniformRangeObserver
from pyq.core.quantization.initializer import MinMaxInitializer, MinMaxOffsetInitializer
from pyq.core.quantization.functional import STEQuantizeFunction, STEOffsetQuantizeFunction
from pyq.core.quantization.wrapper import ActivationQuantizerWrapper, TransformationLayerQuantizerWrapper


# skip warnings
warnings.filterwarnings("ignore")

import torch
from collections import Counter
import numpy as np
import torch


class ASTNodeEncoder(torch.nn.Module):
    '''
        Input:
            x: default node feature. the first and second column represents node type and node attributes.
            depth: The depth of the node in the AST.

        Output:
            emb_dim-dimensional vector

    '''

    def __init__(self, emb_dim, num_nodetypes, num_nodeattributes, max_depth):
        super(ASTNodeEncoder, self).__init__()

        self.max_depth = max_depth

        self.type_encoder = torch.nn.Embedding(num_nodetypes, emb_dim)
        self.attribute_encoder = torch.nn.Embedding(num_nodeattributes, emb_dim)
        self.depth_encoder = torch.nn.Embedding(self.max_depth + 1, emb_dim)

    def forward(self, x, depth):
        depth[depth > self.max_depth] = self.max_depth
        return self.type_encoder(x[:, 0]) + self.attribute_encoder(x[:, 1]) + self.depth_encoder(depth)


def get_vocab_mapping(seq_list, num_vocab):
    '''
        Input:
            seq_list: a list of sequences
            num_vocab: vocabulary size
        Output:
            vocab2idx:
                A dictionary that maps vocabulary into integer index.
                Additioanlly, we also index '__UNK__' and '__EOS__'
                '__UNK__' : out-of-vocabulary term
                '__EOS__' : end-of-sentence

            idx2vocab:
                A list that maps idx to actual vocabulary.

    '''

    vocab_cnt = {}
    vocab_list = []
    for seq in seq_list:
        for w in seq:
            if w in vocab_cnt:
                vocab_cnt[w] += 1
            else:
                vocab_cnt[w] = 1
                vocab_list.append(w)

    cnt_list = np.array([vocab_cnt[w] for w in vocab_list])
    topvocab = np.argsort(-cnt_list, kind='stable')[:num_vocab]

    print('Coverage of top {} vocabulary:'.format(num_vocab))
    print(float(np.sum(cnt_list[topvocab])) / np.sum(cnt_list))

    vocab2idx = {vocab_list[vocab_idx]: idx for idx, vocab_idx in enumerate(topvocab)}
    idx2vocab = [vocab_list[vocab_idx] for vocab_idx in topvocab]

    # print(topvocab)
    # print([vocab_list[v] for v in topvocab[:10]])
    # print([vocab_list[v] for v in topvocab[-10:]])

    vocab2idx['__UNK__'] = num_vocab
    idx2vocab.append('__UNK__')

    vocab2idx['__EOS__'] = num_vocab + 1
    idx2vocab.append('__EOS__')

    # test the correspondence between vocab2idx and idx2vocab
    for idx, vocab in enumerate(idx2vocab):
        assert (idx == vocab2idx[vocab])

    # test that the idx of '__EOS__' is len(idx2vocab) - 1.
    # This fact will be used in decode_arr_to_seq, when finding __EOS__
    assert (vocab2idx['__EOS__'] == len(idx2vocab) - 1)

    return vocab2idx, idx2vocab


def augment_edge(data):
    '''
        Input:
            data: PyG data object
        Output:
            data (edges are augmented in the following ways):
                data.edge_index: Added next-token edge. The inverse edges were also added.
                data.edge_attr (torch.Long):
                    data.edge_attr[:,0]: whether it is AST edge (0) for next-token edge (1)
                    data.edge_attr[:,1]: whether it is original direction (0) or inverse direction (1)
    '''

    ##### AST edge
    edge_index_ast = data.edge_index
    edge_attr_ast = torch.zeros((edge_index_ast.size(1), 2))

    ##### Inverse AST edge
    edge_index_ast_inverse = torch.stack([edge_index_ast[1], edge_index_ast[0]], dim=0)
    edge_attr_ast_inverse = torch.cat(
        [torch.zeros(edge_index_ast_inverse.size(1), 1), torch.ones(edge_index_ast_inverse.size(1), 1)], dim=1)

    ##### Next-token edge

    ## Obtain attributed nodes and get their indices in dfs order
    # attributed_node_idx = torch.where(data.node_is_attributed.view(-1,) == 1)[0]
    # attributed_node_idx_in_dfs_order = attributed_node_idx[torch.argsort(data.node_dfs_order[attributed_node_idx].view(-1,))]

    ## Since the nodes are already sorted in dfs ordering in our case, we can just do the following.
    attributed_node_idx_in_dfs_order = torch.where(data.node_is_attributed.view(-1, ) == 1)[0]

    ## build next token edge
    # Given: attributed_node_idx_in_dfs_order
    #        [1, 3, 4, 5, 8, 9, 12]
    # Output:
    #    [[1, 3, 4, 5, 8, 9]
    #     [3, 4, 5, 8, 9, 12]
    edge_index_nextoken = torch.stack([attributed_node_idx_in_dfs_order[:-1], attributed_node_idx_in_dfs_order[1:]],
                                      dim=0)
    edge_attr_nextoken = torch.cat(
        [torch.ones(edge_index_nextoken.size(1), 1), torch.zeros(edge_index_nextoken.size(1), 1)], dim=1)

    ##### Inverse next-token edge
    edge_index_nextoken_inverse = torch.stack([edge_index_nextoken[1], edge_index_nextoken[0]], dim=0)
    edge_attr_nextoken_inverse = torch.ones((edge_index_nextoken.size(1), 2))

    data.edge_index = torch.cat(
        [edge_index_ast, edge_index_ast_inverse, edge_index_nextoken, edge_index_nextoken_inverse], dim=1)
    data.edge_attr = torch.cat([edge_attr_ast, edge_attr_ast_inverse, edge_attr_nextoken, edge_attr_nextoken_inverse],
                               dim=0)

    return data


def encode_y_to_arr(data, vocab2idx, max_seq_len):
    '''
    Input:
        data: PyG graph object
        output: add y_arr to data
    '''

    # PyG >= 1.5.0
    seq = data.y

    # PyG = 1.4.3
    # seq = data.y[0]

    data.y_arr = encode_seq_to_arr(seq, vocab2idx, max_seq_len)

    return data


def encode_seq_to_arr(seq, vocab2idx, max_seq_len):
    '''
    Input:
        seq: A list of words
        output: add y_arr (torch.Tensor)
    '''

    augmented_seq = seq[:max_seq_len] + ['__EOS__'] * max(0, max_seq_len - len(seq))
    return torch.tensor([[vocab2idx[w] if w in vocab2idx else vocab2idx['__UNK__'] for w in augmented_seq]],
                        dtype=torch.long)


def decode_arr_to_seq(arr, idx2vocab):
    '''
        Input: torch 1d array: y_arr
        Output: a sequence of words.
    '''

    eos_idx_list = torch.nonzero(arr == len(idx2vocab) - 1,
                                 as_tuple=False)  # find the position of __EOS__ (the last vocab in idx2vocab)
    if len(eos_idx_list) > 0:
        clippted_arr = arr[: torch.min(eos_idx_list)]  # find the smallest __EOS__
    else:
        clippted_arr = arr

    return list(map(lambda x: idx2vocab[x], clippted_arr.cpu()))


def test():
    seq_list = [['a', 'b'], ['a', 'b', 'c', 'df', 'f', '2edea', 'a'], ['eraea', 'a', 'c'], ['d'],
                ['4rq4f', 'f', 'a', 'a', 'g']]
    vocab2idx, idx2vocab = get_vocab_mapping(seq_list, 4)
    print(vocab2idx)
    print(idx2vocab)
    print()
    assert (len(vocab2idx) == len(idx2vocab))

    for vocab, idx in vocab2idx.items():
        assert (idx2vocab[idx] == vocab)

    for seq in seq_list:
        print(seq)
        arr = encode_seq_to_arr(seq, vocab2idx, max_seq_len=4)[0]
        # Test the effect of predicting __EOS__
        # arr[2] = vocab2idx['__EOS__']
        print(arr)
        seq_dec = decode_arr_to_seq(arr, idx2vocab)

        print(arr)
        print(seq_dec)
        print('')


# GIN convolution along the graph structure
### GIN convolution along the graph structure
### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        # edge_attr is two dimensional after augment_edge transformation
        self.edge_encoder = torch.nn.Linear(2, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)

        # edge_attr is two dimensional after augment_edge transformation
        self.edge_encoder = torch.nn.Linear(2, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out



### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, node_encoder, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = node_encoder

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, node_depth, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.node_depth, batched_data.batch


        ### computing input node embedding

        h_list = [self.node_encoder(x, node_depth.view(-1,))]
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation


### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, node_encoder, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = node_encoder

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), \
                                                    torch.nn.Linear(2*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))


    def forward(self, batched_data):

        x, edge_index, edge_attr, node_depth, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.node_depth, batched_data.batch

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        h_list = [self.node_encoder(x, node_depth.view(-1,))]
        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation


class GNN(torch.nn.Module):

    def __init__(self, num_vocab, max_seq_len, node_encoder, num_layer = 5, emb_dim = 300,
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_vocab = num_vocab
        self.max_seq_len = max_seq_len
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, node_encoder, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, node_encoder, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        self.graph_pred_linear_list = torch.nn.ModuleList()

        if graph_pooling == "set2set":
            for i in range(max_seq_len):
                 self.graph_pred_linear_list.append(torch.nn.Linear(2*emb_dim, self.num_vocab))

        else:
            for i in range(max_seq_len):
                 self.graph_pred_linear_list.append(torch.nn.Linear(emb_dim, self.num_vocab))


    def forward(self, batched_data):
        '''
            Return:
                A list of predictions.
                i-th element represents prediction at i-th position of the sequence.
        '''

        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        pred_list = []

        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](h_graph))

        return pred_list


multicls_criterion = torch.nn.CrossEntropyLoss()


def train(model, device, loader, optimizer):
    model.train()

    loss_accum = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred_list = model(batch)
            optimizer.zero_grad()

            loss = 0
            for i in range(len(pred_list)):
                loss += multicls_criterion(pred_list[i].to(torch.float32), batch.y_arr[:, i])

            loss = loss / len(pred_list)

            loss.backward()
            optimizer.step()

            loss_accum += loss.item()

    print('Average training loss: {}'.format(loss_accum / (step + 1)))


def eval(model, device, loader, evaluator, arr_to_seq):
    model.eval()
    seq_ref_list = []
    seq_pred_list = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred_list = model(batch)

            mat = []
            for i in range(len(pred_list)):
                mat.append(torch.argmax(pred_list[i], dim=1).view(-1, 1))
            mat = torch.cat(mat, dim=1)

            seq_pred = [arr_to_seq(arr) for arr in mat]

            # PyG = 1.4.3
            # seq_ref = [batch.y[i][0] for i in range(len(batch.y))]

            # PyG >= 1.5.0
            seq_ref = [batch.y[i] for i in range(len(batch.y))]

            seq_ref_list.extend(seq_ref)
            seq_pred_list.extend(seq_pred)

    input_dict = {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list}

    return evaluator.eval(input_dict)


def main():
    # Training settings
    device = 0
    dataset_name = "ogbg-code2"
    batch_size = 64
    num_layer = 5
    emb_dim = 300
    drop_ratio = 0.0
    epochs = 30
    num_workers = 8
    feature = "full"
    filename = "pyq_model_gin_code2.pth" #TODO try again with new file name

    device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")

    random_split = False
    max_seq_len = 2 #5
    num_vocab = 100 #50 #5000

    # automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name=dataset_name)

    seq_len_list = np.array([len(seq) for seq in dataset.data.y])
    print('Target seqence less or equal to {} is {}%.'.format(max_seq_len,
                                                              np.sum(seq_len_list <= max_seq_len) / len(
                                                                  seq_len_list)))

    split_idx = dataset.get_idx_split()

    if random_split:
        print('Using random split')
        perm = torch.randperm(len(dataset))
        num_train, num_valid, num_test = int(0.01*len(split_idx['train'])),int(0.01*len(split_idx['valid'])), int(0.01*len(split_idx['test']))
        split_idx['train'] = perm[:num_train]
        split_idx['valid'] = perm[num_train:num_train + num_valid]
        split_idx['test'] = perm[-num_test:]#perm[num_train + num_valid:]
    else:
        print('Using project split')
        # Modify so that stuff' s from the same project
        num_train, num_valid, num_test = int(0.01 * len(split_idx['train'])), int(0.01 * len(split_idx['valid'])), int(0.01 * len(split_idx['test']))
        split_idx['train'] = split_idx['train'][:num_train]
        split_idx['valid'] = split_idx['valid'][num_train:num_train + num_valid]
        split_idx['test'] = split_idx['test'][num_train + num_valid: num_train + num_valid + num_test]  # perm[num_train + num_valid:]

        #assert (len(split_idx['train']) == num_train)
        #assert (len(split_idx['valid']) == num_valid)
        #assert (len(split_idx['test']) == num_test)

    vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], num_vocab)


    dataset.transform = transforms.Compose(
        [augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, max_seq_len)])

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(dataset_name)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    #for elem in train_loader:
    #    print(elem.y_arr.shape)

    nodetypes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'typeidx2type.csv.gz'))
    nodeattributes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'attridx2attr.csv.gz'))

    print(nodeattributes_mapping)

    node_encoder = ASTNodeEncoder(emb_dim, num_nodetypes=len(nodetypes_mapping['type']),
                                  num_nodeattributes=len(nodeattributes_mapping['attr']), max_depth=20) #20

    model = GNN(num_vocab=len(vocab2idx), max_seq_len=max_seq_len, node_encoder=node_encoder,
                num_layer=num_layer, gnn_type='gin', emb_dim=emb_dim, drop_ratio=drop_ratio,
                virtual_node=False).to(device)


    #exit(-1)


    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, epochs + 1):
        print("=====Epoch {}".format(epoch))
        print("Training...")
        train(model, device, train_loader, optimizer)#, dataset.task_type)

        print("Evaluating...")
        train_perf = eval(model, device, train_loader, evaluator,
                          arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab))
        if epoch % 5 == 0:
            valid_perf = eval(model, device, valid_loader, evaluator,
                              arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab))
            test_perf = eval(model, device, test_loader, evaluator,
                             arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab))

            print({"Train": train_perf, "Validation": valid_perf, "Test": test_perf})


            train_curve.append(train_perf[dataset.eval_metric])
            valid_curve.append(valid_perf[dataset.eval_metric])
            test_curve.append(test_perf[dataset.eval_metric])

    print('F1')
    best_val_epoch = np.argmax(np.array(valid_curve))
    best_train = max(train_curve)
    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    #exit(-1)
    bits = 8
    # !important: configure the quantization function and initialization
    layer_quantizer = TransformationLayerQuantizerWrapper
    layer_quantizer_arguments = {
        "quantizer_function": STEQuantizeFunction(),
        "initializer": MinMaxInitializer().to(device),
        "range_observer": UniformRangeObserver(bits=bits).to(device),
    }

    activation_quantizer = ActivationQuantizerWrapper
    activation_quantizer_arguments = {
        "quantizer_function": STEOffsetQuantizeFunction(),
        "initializer": MinMaxOffsetInitializer().to(device),
        "range_observer": UniformRangeObserver(bits=bits, is_positive=True).to(device),
    }

    skip_layer_by_type = [Sequential, ModuleList, BondEncoder, BatchNorm1d, AtomEncoder, Aggregation, GINConv, GNN_node, Embedding, ASTNodeEncoder]
    parser = TorchModelParser(
        callable_object=layer_quantizer,
        callable_object_kwargs=layer_quantizer_arguments,
        callable_object_for_nonparametric=activation_quantizer,
        callable_object_for_nonparametric_kwargs=activation_quantizer_arguments,
        skip_layer_by_type=skip_layer_by_type,
    )

    model = parser.apply(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, epochs + 1):
        print("=====Epoch {}".format(epoch))
        print("Training...")
        train(model, device, train_loader, optimizer)#, dataset.task_type)

        print("Evaluating...")
        train_perf = eval(model, device, train_loader, evaluator,
                          arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab))
        if epoch % 5 == 0:
            valid_perf = eval(model, device, valid_loader, evaluator,
                              arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab))
            test_perf = eval(model, device, test_loader, evaluator,
                             arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab))

            print({"Train": train_perf, "Validation": valid_perf, "Test": test_perf})

            train_curve.append(train_perf[dataset.eval_metric])
            valid_curve.append(valid_perf[dataset.eval_metric])
            test_curve.append(test_perf[dataset.eval_metric])

    if "classification" in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    unparser = TorchModelParser(
        callable_object=layer_quantizer.unwrap,
        callable_object_for_nonparametric=activation_quantizer.unwrap,
        skip_layer_by_type=skip_layer_by_type,
    )

    final_model = model#unparser.apply(model)

    print("Finished training!")
    print("Best validation score: {}".format(valid_curve[best_val_epoch]))
    print("Test score: {}".format(test_curve[best_val_epoch]))

    #print(final_model._wrapped_object._wrapped_object.gnn_node.convs[0].mlp[3]._wrapped_object.weight, flush=True)
    if not filename == "":
        torch.save(
            {
                "Val": valid_curve[best_val_epoch],
                "Test": test_curve[best_val_epoch],
                "Train": train_curve[best_val_epoch],
                "BestTrain": best_train,
                'Model':final_model
            },
            filename,
        )
if __name__ == "__main__":
    main()
