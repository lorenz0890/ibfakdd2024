import os.path as osp
import warnings

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Linear, ModuleList
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from pyq.core.parser import TorchModelParser
from pyq.core.quantization.functional import STEOffsetQuantizeFunction, STEQuantizeFunction
from pyq.core.quantization.initializer import MinMaxInitializer, MinMaxOffsetInitializer
from pyq.core.quantization.observer import UniformRangeObserver
from pyq.core.quantization.wrapper import ActivationQuantizerWrapper, TransformationLayerQuantizerWrapper

# skip warnings
warnings.filterwarnings("ignore")


class Net(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, alpha, theta, shared_weights=True, dropout=0.0):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset.num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels, dataset.num_classes))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GCN2Conv(hidden_channels, alpha, theta, layer + 1, shared_weights, normalize=False))

        self.dropout = dropout

    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x.log_softmax(dim=-1)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def validate():
    model.eval()
    pred, accs = model(data.x, data.adj_t).argmax(dim=-1), []
    for _, mask in data("train_mask", "val_mask", "test_mask"):
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


if __name__ == "__main__":
    filename = "pyq_model_gcn_cora_2.pth"

    fp32_epochs = 1000
    int8_epochs = 1000

    bits = 8
    dataset = "Cora"

    path = osp.join("./datasets/", dataset)
    transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
    dataset = Planetoid(path, dataset, transform=transform)
    data = dataset[0]
    data.adj_t = gcn_norm(data.adj_t)  # Pre-process GCN normalization.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5, shared_weights=True, dropout=0.6).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(
        [
            dict(params=model.convs.parameters(), weight_decay=0.01),
            dict(params=model.lins.parameters(), weight_decay=5e-4),
        ],
        lr=0.01,
    )

    print("_" * 35, "FP32 Training", "_" * 35)

    best_val_acc = test_acc = 0
    for epoch in range(1, fp32_epochs + 1):
        loss = train()
        train_acc, val_acc, tmp_test_acc = validate()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        print(
            f"Epoch: {epoch:04d}, Loss: {loss:.4f} Train: {train_acc:.4f}, "
            f"Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f}, "
            f"Final Test: {test_acc:.4f}"
        )

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

    skip_layer_by_type = [
        GCN2Conv,
        ModuleList,
    ]
    parser = TorchModelParser(
        callable_object=layer_quantizer,
        callable_object_kwargs=layer_quantizer_arguments,
        callable_object_for_nonparametric=activation_quantizer,
        callable_object_for_nonparametric_kwargs=activation_quantizer_arguments,
        skip_layer_by_type=skip_layer_by_type,
    )

    model = parser.apply(model)
    optimizer = torch.optim.Adam(
        [
            dict(params=model._wrapped_object.convs.parameters(), weight_decay=0.01),
            dict(params=model._wrapped_object.lins.parameters(), weight_decay=5e-4),
        ],
        lr=0.0001,
    )

    print("_" * 35, "INT{} Training".format(bits), "_" * 35)

    best_val_acc = test_acc = 0
    for epoch in range(1, int8_epochs + 1):
        loss = train()
        train_acc, val_acc, tmp_test_acc = validate()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        print(
            f"Epoch: {epoch:04d}, Loss: {loss:.4f} Train: {train_acc:.4f}, "
            f"Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f}, "
            f"Final Test: {test_acc:.4f}"
        )

    final_model = model  # unparser.apply(model)

    if not filename == "":
        torch.save(
            {
                "Val": best_val_acc,
                'Model': final_model
            },
            filename,
        )
