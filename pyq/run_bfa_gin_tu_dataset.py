import traceback

import torch.nn

from app.graph.gin_tu_quantization import *
#from ogb.graphproppred import PygGraphPropPredDataset as ogb_datasets
#from torch_geometric.data import DataLoader as GMDataLoader
from pyq.fia.bfa.attack_gin_tu_dataset.RBFA import RandomBFA as RBFA
from pyq.fia.bfa.attack_gin_tu_dataset.PBFA import ProgressiveBFA as PBFA
from pyq.fia.bfa.attack_gin_tu_dataset.IBFA import InjectivityBFA as IBFA
import datetime
import time
import random
#from ogb.graphproppred import Evaluator

from pyq.fia.bfa.utils import _WrappedGraphDataset, eval_ogbg, get_n_params

import argparse #new
import json


def setup_data(dataset_name, batch_size, shuffle=True, ibfa_limit=0.01):
    if dataset_name == 'tu-collab':
        dataset = TUDataset(root='/tmp/COLLAB', name='COLLAB').shuffle()  # transform=T.NormalizeFeatures()
    if dataset_name == 'tu-gitstar':
        dataset = TUDataset(root='/tmp/github_stargazers', name='github_stargazers').shuffle()
    # data = dataset[0]
    # perm = torch.randperm(len(dataset))
    # dataset = dataset[perm]
    train_dataset = dataset[:int(len(dataset) * 0.8*ibfa_limit)]
    test_dataset = dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)]
    val_dataset = dataset[int(len(dataset) * 0.9):]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

def setup_net(dataset_name):
    if dataset_name == 'tu-collab':
        net = torch.load("/media/lorenz/Volume/code/pyq_main/pyq/app/graph/model_copies/pyq_model_gin_collab.pth")['Model']
    if dataset_name == 'tu-gitstar':
        net = torch.load("/media/lorenz/Volume/code/pyq_main/pyq/app/graph/model_copies/pyq_model_gin_gitstar.pth")['Model']
    net.eval()
    return net

def setup_attack(attack_type, dataset):
    BFA, criterion, data, target = None, None, None, None
    if attack_type == 'RBFA':
        BFA = RBFA
        criterion = torch.nn.CrossEntropyLoss()#torch.nn.BCEWithLogitsLoss()
    if attack_type == 'PBFA':
        BFA = PBFA
        criterion = torch.nn.CrossEntropyLoss()#torch.nn.BCEWithLogitsLoss()
    if attack_type in ['IBFAv1', 'IBFAv2']:
        BFA = IBFA
        if dataset in ['tu-gitstar']:
            criterion = torch.nn.L1Loss()# L1 works better for molhiv than KLDDiv
        if dataset in ['tu-collab']:
             criterion = torch.nn.KLDivLoss(log_target=True)
    return BFA, criterion

@torch.no_grad()
def validate(model, test_loader):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            x = torch.rand(data.num_nodes, 3).to(device)
            #print(torch_geometric.utils.degree(data.edge_index.flatten()).shape)
            #x = torch_geometric.utils.degree(data.edge_index[0]).view(-1, 1)
            outputs = model(x , data.edge_index, data.batch)
            predicted = outputs.argmax(dim=-1)#torch.max(outputs.data, 1)
            total += data.y.size(0)
            correct += (predicted == data.y).sum().item()
                #if i == 2: print(predicted, data.y)

    acc = 100 * correct / total
    return acc

def main():
    print('asd', flush=True)
    parser = argparse.ArgumentParser(prog='python run_bfa_gin_molecular.py')
    parser.add_argument('--type', help='PBFA, RBFA, IBFAv1, IBFAv2', type=str)
    parser.add_argument('--data', help='ogbg-molpcba, ogbg-molhiv, ogbg-moltox21, ogbg-molclintox, ogbg-moltoxcast, ogbg-molbace', type=str)
    parser.add_argument('--n', help='Run experiment n times', type=int)
    parser.add_argument('--k', help='Run BFA k times', type=int)
    parser.add_argument('--sz', help='Batch size', type=int)
    parser.add_argument('--lim', help='IBFA data usage limit', type=float)
    args = parser.parse_args()
    print(args.type, args.data, args.n, args.k, args.sz)

    attack_type, dataset_name, device = args.type, args.data, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_runs, eval_runs, attack_runs, batch_size = args.n, 10, args.k, args.sz
    ibfa_lim = args.lim
    if not 'IBFA' in attack_type:
        ibfa_lim = 1.0

    #attack_type, dataset_name, device = 'RBFA', 'ogbg-molpcba', torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #experiment_runs, eval_runs, attack_runs, batch_size = 10, 10, 10, 256
    pre_metric, post_metric, bit_flips = [], [], []

    print('Start time', datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))
    experiment_accumulated_seconds = 0
    failure_counter = 0
    train_loader, test_loader = setup_data(dataset_name, batch_size, True, ibfa_lim) #shuffled every epoch -> should be False

    for r in range(experiment_runs):
        start_time = time.time()
        net = setup_net(dataset_name).to(device)#copy.deepcopy(net_clean).to(device)
        #net.eval()
        #print(eval_ogbg(net, device, test_loader, evaluator), flush=True)
        #print(eval_ogbg(net, device, test_loader, evaluator), flush=True)
        #print(eval_ogbg(net, device, test_loader, evaluator), flush=True)
        #exit(-1)
        BFA, criterion = setup_attack(attack_type, dataset_name)
        print('params', get_n_params(net))
        data, target = None, None
        if attack_type == 'PBFA':
            with torch.no_grad():
                x = random.randint(0, int(len(train_loader.dataset)/batch_size))
                for i, data in enumerate(train_loader):
                    #print(i, x)
                    if i == x:
                        print('PBFA data found', flush=True)
                        data = data.to(device)
                        target = data.y
                        break
        #print(target)
        #exit(1)
        if attack_type == 'RBFA':
            pass
        if attack_type == 'IBFAv1':
            with torch.no_grad():
                max_loss = float('-inf')
                d = train_loader#d = [x for x in train_loader]
                for i, data1 in enumerate(d):
                    for j, data2 in enumerate(d):
                        #print(data1.size(), data2.size(), flush=True)
                        #if data1.size() == data2.size() and i != j:
                        data1 = data1.to(device)
                        data2 = data2.to(device)

                        x1, x2 = torch.rand(data1.num_nodes, 3).to(device), torch.rand(data2.num_nodes, 3).to(device)
                        is_labeled1, is_labeled2 = data1.y == data1.y, data2.y == data2.y # Not all data always labeled.
                        out1, out2 = net(x1, data1.edge_index, data1.batch)[is_labeled1], net(x2, data2.edge_index, data2.batch)[is_labeled2]
                        cut = min(out1.shape[0], out2.shape[0])
                        #print(out1, out2)
                        loss = criterion(out1[:cut], out2[:cut])
                        #print(loss, flush=True)
                        if max_loss < loss:
                            print(i, loss)
                            max_loss = loss
                            max_data1 = data1
                            max_data2 = data2
                        #break
                        #break
                    #if i > 10:
                    #    break
            print(max_data1.size(), max_data2.size(), flush=True)
            print('IBFAv1 data found', flush=True)
            data = max_data1
            target = max_data2

        net = net.to(device)
        #net.train()
        #train_perf = eval_ogbg(net, device, train_loader, evaluator)
        test_perf = validate(net, test_loader)
        #print(test_perf)
        pre_metric.append(test_perf)
        attacker, attack_log = BFA(criterion, net, 25, True), None

        layers = []
        try:
            flips = 0
            if attack_type in ['RBFA', 'PBFA', 'IBFAv1']:
                for i in range(attack_runs):
                    attack_log = attacker.run_attack(net, data, target, attack_runs)
                    print(attack_log)
                    if len(attack_log) > 0:
                        flips = attack_log[-1][1]
                        name = attack_log[0][2]
                        #exit(0)
                        if 'conv' in name:
                            print()
                            layers.append(int(attack_log[0][2].split('.')[-1]))
                        else:
                            layers.append(-1)
                    else:
                        break
            if attack_type == 'IBFAv2':
                for _ in range(attack_runs):
                    with torch.no_grad():
                        max_loss = float('-inf')
                        for i, data1 in enumerate(train_loader):
                            for j, data2 in enumerate(train_loader):
                                #if data1.size() == data2.size() and i != j:
                                data1 = data1.to(device)
                                data2 = data2.to(device)
                                #break
                                x1, x2 = torch.rand(data1.num_nodes, 3).to(device), torch.rand(data2.num_nodes,
                                                                                               3).to(device)
                                is_labeled1, is_labeled2 = data1.y == data1.y, data2.y == data2.y  # Not all data always labeled.
                                out1, out2 = net(x1, data1.edge_index, data1.batch)[is_labeled1], \
                                             net(x2, data2.edge_index, data2.batch)[is_labeled2]
                                cut = min(out1.shape[0], out2.shape[0])
                                loss = criterion(out1[:cut], out2[:cut])

                                if max_loss < loss:
                                    print(i, loss)
                                    max_loss = loss
                                    max_data1 = data1
                                    max_data2 = data2
                                #break
                            #break
                            if i > 5:
                                break
                    attack_log = attacker.run_attack(net, max_data1.to(device), max_data2.to(device), attack_runs)
                    if len(attack_log) > 0:
                        flips = attack_log[-1][1]
                        name = attack_log[0][2]
                        # exit(0)
                        if 'conv' in name:
                            print()
                            layers.append(int(attack_log[0][2].split('.')[-1]))
                        else:
                            layers.append(-1)
                    else:
                        break
            if attack_log is None:
                raise Exception('No attack solution found')
            #if len(attack_log) > 0:
            bit_flips.append(flips)
            #train_perf = eval_ogbg(net, device, train_loader, evaluator)
            test_perf = validate(net, test_loader)
            #print(test_perf.values())
            post_metric.append(test_perf)
        except Exception as e:
            failure_counter+=1
            print(failure_counter, e.args)
            print(traceback.format_exc())
        ct = datetime.datetime.now()
        experiment_accumulated_seconds += time.time() - start_time

        #print(layers)
        print('Current time:', ct.strftime("%d/%m/%Y, %H:%M:%S"),
              'Completed:', (r + 1) / experiment_runs * 100, '%',
              'Duration per experiment:', round(time.time() - start_time, 2), 's',
              'ETA:', (ct+datetime.timedelta(seconds=((experiment_accumulated_seconds/(r+1)) * (experiment_runs-r-1)))).strftime("%d/%m/%Y, %H:%M:%S")
        )
    print('Pre and post attack average metric, bitflips, failures',
          sum(pre_metric) / len(pre_metric),
          sum(post_metric) / len(post_metric),
          sum(bit_flips) / len(bit_flips),
          failure_counter
          )
    print('Start time', datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))

    json_string = json.dumps({'pre' : sum(pre_metric) / len(pre_metric),
     'post' : sum(post_metric) / len(post_metric),
     'flips' : sum(bit_flips) / len(bit_flips),
     'fails' : failure_counter
     })

    with open("{}_{}_{}_{}_{}_{}.json".format(attack_type, dataset_name, device, experiment_runs, attack_runs, batch_size), 'w') as outfile:
        outfile.write(json_string)

if __name__ == "__main__":
    main()