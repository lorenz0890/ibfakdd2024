import random

import torch.nn
from torch_geometric.datasets import WebKB

from app.graph.gin_cora_quantization import *
from torch_geometric.data import DataLoader as GMDataLoader
from pyq.fia.bfa.attack_gin_cora.RBFA import RandomBFA as RBFA
from pyq.fia.bfa.attack_gin_cora.PBFA import ProgressiveBFA as PBFA
from pyq.fia.bfa.attack_gin_cora.IBFA import InjectivityBFA as IBFA
import datetime
import time
import traceback

from pyq.fia.bfa.utils import _WrappedGraphDataset, evaluate_cora

def setup_net(dataset_name):
    if dataset_name == 'Cora':
        net = torch.load("./models/pyq_model_gin_cora.pth")['Model']
    elif dataset_name == 'PubMed':
        net = torch.load("./models/pyq_model_gin_pubmed.pth")['Model']
    net.eval()
    return net

def setup_attack(attack_type):
    BFA, criterion, data, target = None, None, None, None
    if attack_type == 'RBFA':
        BFA = RBFA
        criterion = F.cross_entropy#torch.nn.BCEWithLogitsLoss()
    if attack_type == 'PBFA':
        BFA = PBFA
        criterion = F.cross_entropy#torch.nn.BCEWithLogitsLoss()
    if attack_type in ['IBFAv1', 'IBFAv2']:
        BFA = IBFA
        criterion = torch.nn.KLDivLoss(log_target=True)
    return BFA, criterion

@torch.no_grad()
def validate(model, data, fold=0):
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_weight).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

def main():
    parser = argparse.ArgumentParser(prog='python run_bfa_gin_planetoid.py')
    parser.add_argument('--type', help='PBFA, RBFA, IBFAv1, IBFAv2', type=str)
    parser.add_argument('--data', help='Wisconsin, Texas', type=str)
    parser.add_argument('--n', help='Run experiment n times', type=int)
    parser.add_argument('--k', help='Run BFA k times', type=int)
    parser.add_argument('--sz', help='Batch size', type=int)
    #parser.add_argument('--f', help='Fold for which to evaluate', type=int)
    fold = None # Not used for Cora/PubMed, kept here as dummy.
    args = parser.parse_args()
    print(args.type, args.data, args.n, args.k, args.sz)

    # attack_type, device = 'PBFA', torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # experiment_runs, eval_runs, attack_runs, batch_size = 1, 10, 100, 32
    attack_type, dataset_name, device = args.type, args.data, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_runs, eval_runs, attack_runs, batch_size = args.n, 10, args.k, args.sz
    pre_acc, post_acc, bit_flips = [], [], []

    print('Start time', datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))
    experiment_accumulated_seconds = 0
    failure_counter = 0

    for r in range(experiment_runs):
        start_time = time.time()

        net = setup_net(dataset_name).to(device)#copy.deepcopy(net_clean).to(device)
        BFA, criterion = setup_attack(attack_type)

        dataset = dataset_name #PubMed # Cora
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
        dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())[0]

        dataset = dataset.to(device)
        #dataset.x = dataset.x[:, :10]
        print(dataset.train_mask)
        if attack_type == 'PBFA':
            with torch.no_grad():

                true_indices = torch.where(dataset.train_mask)[0]
                if batch_size < len(true_indices):
                    selected_indices = true_indices[torch.randperm(len(true_indices))[:batch_size]]
                else:
                    selected_indices = true_indices
                dataset.train_mask = torch.zeros_like(dataset.train_mask, dtype=torch.bool)
                dataset.train_mask[selected_indices] = True

        target1 = None 
        target2 = None
        print(dataset.train_mask)
        #exit()

        if attack_type == 'RBFA':
            target1, target2 = None, None
        if attack_type == 'IBFAv1':
            train_mask_orig = dataset.train_mask.clone()
            out1_orig = net(dataset.x, dataset.edge_index, dataset.edge_weight)
            out2_orig = net(dataset.x, dataset.edge_index, dataset.edge_weight)
            with torch.no_grad():
                max_loss = float('-inf')
                for k in range(0, dataset.train_mask.shape[0]**2, batch_size):
                    true_indices = torch.where(train_mask_orig)[0]
                    if batch_size < len(true_indices):
                        selected_indices = true_indices[torch.randperm(len(true_indices))[:batch_size]]
                    else:
                        selected_indices = true_indices
                    mask1 = torch.zeros_like(train_mask_orig, dtype=torch.bool)
                    mask1[selected_indices] = True

                    if batch_size < len(true_indices):
                        selected_indices = true_indices[torch.randperm(len(true_indices))[:batch_size]]
                    else:
                        selected_indices = true_indices
                    mask2 = torch.zeros_like(train_mask_orig, dtype=torch.bool)
                    mask2[selected_indices] = True

                    out1 = out1_orig[mask1]
                    out2 = out2_orig[mask2]
                    if out1.shape[0] != out2.shape[0]:
                        continue
                    loss = criterion(out1, out2)
                    #print(loss)

                    if max_loss < loss and not torch.isnan(loss):
                        print(round(k/dataset.train_mask.shape[0]**2*100, 4), loss)
                        max_loss = loss
                        target1 = mask1
                        target2 = mask2
                    if k > batch_size*25000:
                        break

        net = net.to(device)

        print(validate(net, dataset, fold))
        #exit()
        pre_acc.append(validate(net, dataset, fold)[2])
        attacker, attack_log = BFA(criterion, net, 5, True), None #10 Texas, 5 Wisconsin - prevents PBFA from being stuck, 5 Cora/Pubmed

        flips = 0
        try:
            if attack_type in ['RBFA', 'PBFA', 'IBFAv1']:
                for i in range(attack_runs):
                    attack_log = attacker.run_attack(net, dataset, target1, target2, batch_size, attack_runs, fold)
                    if len(attack_log) > 0:
                        flips = attack_log[-1][1]
                    else:
                        break

            if attack_type == 'IBFAv2':
                train_mask_orig = dataset.train_mask.clone()
                for i in range(attack_runs):
                    out1_orig = net(dataset.x, dataset.edge_index, dataset.edge_weight)
                    out2_orig = net(dataset.x, dataset.edge_index, dataset.edge_weight)
                    with torch.no_grad():
                        max_loss = float('-inf')
                        for k in range(0, dataset.train_mask.shape[0]**2, batch_size):
                            true_indices = torch.where(train_mask_orig)[0]
                            if batch_size < len(true_indices):
                                selected_indices = true_indices[torch.randperm(len(true_indices))[:batch_size]]
                            else:
                                selected_indices = true_indices
                            mask1 = torch.zeros_like(train_mask_orig, dtype=torch.bool)
                            mask1[selected_indices] = True

                            if batch_size < len(true_indices):
                                selected_indices = true_indices[torch.randperm(len(true_indices))[:batch_size]]
                            else:
                                selected_indices = true_indices
                            mask2 = torch.zeros_like(train_mask_orig, dtype=torch.bool)
                            mask2[selected_indices] = True

                            out1 = out1_orig[mask1]
                            out2 = out2_orig[mask2]
                            if out1.shape[0] != out2.shape[0]:
                                continue
                            loss = criterion(out1, out2)
                            #print(loss)

                            if max_loss < loss and not torch.isnan(loss):
                                print(k, loss)
                                max_loss = loss
                                target1 = mask1
                                target2 = mask2
                            if k > batch_size*10000:
                                break

                    attack_log = attacker.run_attack(net, dataset, target1, target2, batch_size, attack_runs, fold)
                    if len(attack_log) > 0:
                        flips = attack_log[-1][1]
                    else:
                        break
            if attack_log is None:
                raise Exception('No attack solution found')

            bit_flips.append(flips)
            post_acc.append(validate(net, dataset, fold)[2])
        except Exception as e:
            failure_counter+=1
            print(failure_counter, e.args)
            print(traceback.format_exc())
            exit(-1)
        ct = datetime.datetime.now()
        experiment_accumulated_seconds += time.time() - start_time

        print('Current time:', ct.strftime("%d/%m/%Y, %H:%M:%S"),
              'Completed:', (r + 1) / experiment_runs * 100, '%',
              'Duration per experiment:', round(time.time() - start_time, 2), 's',
              'ETA:', (ct+datetime.timedelta(seconds=((experiment_accumulated_seconds/(r+1)) * (experiment_runs-r-1)))).strftime("%d/%m/%Y, %H:%M:%S")
        )
    print('Pre and post attack average ACC, bitflips, failures',
          sum(pre_acc) / len(pre_acc),
          sum(post_acc) / len(post_acc),
          sum(bit_flips) / len(bit_flips),
          failure_counter
          )
    print('Start time', datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))
if __name__ == "__main__":
    main()
