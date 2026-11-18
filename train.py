"""飞桨学习赛：图神经网络入门节点分类。

数据目录里放这四个文件：
    feat.npy    节点特征，形状 (130644, 100)
    edges.csv   两列、无表头
    train.csv   列 nid,label
    test.csv    列 nid

运行：
    python train.py --data_dir data
生成当前目录下的 submission.csv，表头为 nid,label。
"""

import argparse
import os

import numpy as np
import pandas as pd
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pgl
from pgl.nn import GCNConv


def find_file(data_dir, name):
    direct = os.path.join(data_dir, name)
    if os.path.isfile(direct):
        return direct
    for root, _, files in os.walk(data_dir):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError("在 %s 下找不到 %s" % (data_dir, name))


def load_graph(data_dir):
    feat_path = find_file(data_dir, "feat.npy")
    edge_path = find_file(data_dir, "edges.csv")
    feat = np.load(feat_path).astype("float32")
    num_nodes = feat.shape[0]
    edges = pd.read_csv(edge_path, header=None, names=["src", "dst"]).values
    edges = edges.astype("int64")
    edges = np.vstack([edges, edges[:, ::-1]])
    loops = np.arange(num_nodes, dtype="int64")
    loops = np.stack([loops, loops], axis=1)
    edges = np.vstack([edges, loops])
    edges = np.unique(edges, axis=0)
    graph = pgl.Graph(num_nodes=num_nodes, edges=edges, node_feat={"feat": feat})
    return graph, feat


def split_train_valid(train_df, valid_ratio, seed):
    rng = np.random.default_rng(seed)
    train_idx = []
    valid_idx = []
    for _, group in train_df.groupby("label"):
        nids = group["nid"].to_numpy()
        labels = group["label"].to_numpy()
        order = rng.permutation(len(nids))
        nids = nids[order]
        labels = labels[order]
        n_valid = max(1, int(len(nids) * valid_ratio))
        valid_idx.append(np.stack([nids[:n_valid], labels[:n_valid]], axis=1))
        train_idx.append(np.stack([nids[n_valid:], labels[n_valid:]], axis=1))
    train_arr = np.concatenate(train_idx, axis=0)
    valid_arr = np.concatenate(valid_idx, axis=0)
    return train_arr[:, 0], train_arr[:, 1], valid_arr[:, 0], valid_arr[:, 1]


class ResGCN(nn.Layer):
    def __init__(self, in_dim, hidden, num_class, num_layers, dropout):
        super(ResGCN, self).__init__()
        self.dropout = dropout
        self.input_fc = nn.Linear(in_dim, hidden)
        self.convs = nn.LayerList(
            [GCNConv(hidden, hidden) for _ in range(num_layers)]
        )
        self.norms = nn.LayerList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.out = nn.Linear(hidden, num_class)

    def forward(self, graph, x):
        x = self.input_fc(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(graph, x)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, self.dropout, training=self.training)
            x = x + h
        return self.out(x)


def accuracy(logits, index, label):
    pred = paddle.argmax(logits[index], axis=1)
    label = paddle.to_tensor(label, dtype="int64")
    return float((pred == label).astype("float32").mean().numpy())


def train_one(graph, feat, train_n, train_y, valid_n, valid_y, args, seed):
    paddle.seed(seed)
    np.random.seed(seed)
    model = ResGCN(
        in_dim=feat.shape[1],
        hidden=args.hidden,
        num_class=args.num_class,
        num_layers=args.layers,
        dropout=args.dropout,
    )
    opt = paddle.optimizer.Adam(
        learning_rate=args.lr,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
    )
    x = paddle.to_tensor(feat)
    y = paddle.to_tensor(train_y, dtype="int64")
    index = paddle.to_tensor(train_n, dtype="int64")
    best_acc = -1.0
    best_state = None
    wait = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(graph, x)
        loss = F.cross_entropy(logits[index], y)
        loss.backward()
        opt.step()
        opt.clear_grad()
        model.eval()
        with paddle.no_grad():
            logits = model(graph, x)
            acc = accuracy(logits, valid_n, valid_y)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if epoch % 20 == 0 or epoch == 1:
            print(
                "seed %d epoch %d loss %.4f val_acc %.4f"
                % (seed, epoch, float(loss.numpy()), acc)
            )
        if wait >= args.patience:
            print("seed %d early stop at epoch %d, best val_acc %.4f" % (seed, epoch, best_acc))
            break
    model.set_state_dict(best_state)
    model.eval()
    with paddle.no_grad():
        logits = model(graph, x)
    return F.softmax(logits, axis=1), best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--num_class", type=int, default=35)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()

    paddle.set_device("gpu" if paddle.is_compiled_with_cuda() else "cpu")
    graph, feat = load_graph(args.data_dir)
    train_df = pd.read_csv(find_file(args.data_dir, "train.csv"))
    test_df = pd.read_csv(find_file(args.data_dir, "test.csv"))

    probs = None
    for seed in args.seeds:
        train_n, train_y, valid_n, valid_y = split_train_valid(
            train_df, args.valid_ratio, seed
        )
        prob, best_acc = train_one(
            graph, feat, train_n, train_y, valid_n, valid_y, args, seed
        )
        print("seed %d best val_acc %.4f" % (seed, best_acc))
        probs = prob if probs is None else probs + prob
    probs = probs / float(len(args.seeds))
    test_index = paddle.to_tensor(test_df["nid"].to_numpy(), dtype="int64")
    pred = paddle.argmax(probs[test_index], axis=1).numpy().astype("int64")
    submission = pd.DataFrame({"nid": test_df["nid"].to_numpy(), "label": pred})
    submission.to_csv(args.out, index=False)
    print("wrote %s, rows %d" % (args.out, len(submission)))


if __name__ == "__main__":
    main()
