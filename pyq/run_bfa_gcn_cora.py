import torch.nn

from app.graph.gcn_cora_quantization import *
from torch_geometric.data import DataLoader as GMDataLoader
from pyq.fia.bfa.attack_gin_wkb.RBFA import RandomBFA as RBFA
from pyq.fia.bfa.attack_gin_wkb.PBFA import ProgressiveBFA as PBFA
from pyq.fia.bfa.attack_gin_wkb.IBFA import InjectivityBFA as IBFA
import datetime
import time
import traceback

from pyq.fia.bfa.utils import _WrappedGraphDataset, evaluate_cora

def setup_data(batch_size, shuffle=True):
    dataset = "Cora"
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    data = dataset#[0]
    train_loader = GMDataLoader(dataset=_WrappedGraphDataset(data, None),
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=12, pin_memory=True)
    test_loader = GMDataLoader(dataset=_WrappedGraphDataset(data, None),
                               batch_size=batch_size,
                               shuffle=shuffle,
                               num_workers=12, pin_memory=True)
    return train_loader, test_loader

def setup_net():
    net = torch.load("/media/lorenz/Volume/code/pyq_main/pyq/app/graph/pyq_model_gcn_cora.pth")['Model']
    #test_auroc = torch.load("/media/lorenz/Volume/code/pyq_main/pyq/app/graph/pyq_model.pth")['Test']
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
        criterion = torch.nn.L1Loss()
    return BFA, criterion


def main():
    attack_type, device = 'IBFAv1', torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_runs, eval_runs, attack_runs, batch_size = 1, 10, 5, 256
    pre_acc, post_acc, bit_flips = [], [], []

    print('Start time', datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))
    experiment_accumulated_seconds = 0
    failure_counter = 0
    train_loader, test_loader = setup_data(batch_size, True) #shuffled every epoch

    for r in range(experiment_runs):
        start_time = time.time()

        net = setup_net().to(device)#copy.deepcopy(net_clean).to(device)
        BFA, criterion = setup_attack(attack_type)

        data, target = None, None
        if attack_type == 'PBFA':
            with torch.no_grad():
                for i, data in enumerate(train_loader):
                    data = data.to(device)
                    target = data.y

        if attack_type == 'RBFA':
            pass
        if attack_type == 'IBFAv1':
            with torch.no_grad():
                max_loss = 0
                for i, data1 in enumerate(train_loader):
                    for j, data2 in enumerate(train_loader):
                        if data1.size() == data2.size():
                            data1 = data1.to(device)
                            data2 = data2.to(device)
                            #print(data1.train_mask.shape, flush=True)
                            #print(data1.train_mask, flush=True)
                            #exit(-1)
                            s = 70
                            for k in range(0, 140, s):
                                for m in range(0, 140, s):
                                    loss = criterion(net(data1.x, data1.edge_index, data1.edge_weight)[data1.train_mask][k:k+s, :],
                                                     net(data2.x, data2.edge_index, data2.edge_weight)[data2.train_mask][m:m+s, :])

                                    if max_loss < loss:
                                        print(k,m,loss)
                                        max_loss = loss
                                        max_data1 = data1
                                        max_data2 = data2
            #exit(-1)
            data = max_data1
            target = max_data2

        net = net.to(device)

        pre_acc.append(evaluate_cora(net, test_loader, device, eval_runs))
        attacker, attack_log = BFA(criterion, net, 100, True), None

        try:
            if attack_type in ['RBFA', 'PBFA', 'IBFAv1']:
                for i in range(attack_runs):
                    attack_log = attacker.run_attack(net, data, target)
            if attack_type == 'IBFAv2':
                for _ in range(attack_runs):
                    with torch.no_grad():
                        max_loss = 0
                        for i, data1 in enumerate(train_loader):
                            for j, data2 in enumerate(train_loader):
                                if data1.size() == data2.size():
                                    data1 = data1.to(device)
                                    data2 = data2.to(device)
                                    loss = criterion(
                                        net(data1.x, data1.edge_index, data1.edge_weight)[data1.train_mask],
                                        net(data2.x, data2.edge_index, data2.edge_weight)[data2.train_mask])
                                    if max_loss < loss:
                                        max_loss = loss
                                        max_data1 = data1
                                        max_data2 = data2
                    attack_log = attacker.run_attack(net, max_data1.to(device), max_data2.to(device))
            if attack_log is None:
                raise Exception('No attack solution found')
            bit_flips.append(attack_log[-1][1])
            post_acc.append(evaluate_cora(net, test_loader, device, eval_runs))
        except Exception as e:
            failure_counter+=1
            print(failure_counter, e.args)
            print(traceback.format_exc())
            exit(-1)
        ct = datetime.datetime.now()
        experiment_accumulated_seconds += time.time() - start_time
        #print(pre_acc)
        #print(post_acc)
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