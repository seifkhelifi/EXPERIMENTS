# Graph construction utilities — drop-in to your repo.
import numpy as np
import torch


def make_fully_connected_edge_index(num_nodes, device=None):
    """
    Returns edge_index shaped [2, num_nodes*num_nodes] for a fully connected directed graph.
    (Self-loops included). Deterministic and reproducible.
    """
    row = np.repeat(np.arange(num_nodes), num_nodes)
    col = np.tile(np.arange(num_nodes), num_nodes)
    edge_index = np.stack([row, col], axis=0)
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
    return edge_index


# -----------------------
# Strategy A: Explicit graph per sample (host/endpoint graph)
# -----------------------
def build_explicit_graphs_from_df(df_slice, node_type="ip", max_nodes=128, device=None):
    """
    Build a graph per sample row (or small time-window) using src/dst IPs and ports.
    - df_slice: pandas.DataFrame (subset: train/val/test) OR numpy array (then this function can't build explicit graph).
    - node_type: 'ip' or 'ip_port' to decide granularity.
    Returns:
        node_features_list: list of numpy arrays [num_nodes, feat_dim]
        edge_index_list: list of torch.LongTensor [2, num_edges]
        edge_attr_list: list of numpy arrays or None
    Notes:
        - This function assumes df_slice contains columns: 'Scr_IP','Des_IP','Scr_port','Des_port', plus numeric features.
        - If df_slice is the scaled numpy features only, skip this function and use the pseudo-node option.
    """
    import pandas as pd

    assert hasattr(
        df_slice, "columns"
    ), "Pass a pandas DataFrame for explicit graph construction."

    node_features_list = []
    edge_index_list = []
    edge_attr_list = []

    # iterate rows -> create graph with nodes = {src_ip, dst_ip} for that row
    for _, row in df_slice.iterrows():
        if node_type == "ip":
            nodes = [row["Scr_IP"], row["Des_IP"]]
            node_map = {nodes[i]: i for i in range(len(nodes))}
        else:  # ip_port
            nodes = [
                f"{row['Scr_IP']}:{row['Scr_port']}",
                f"{row['Des_IP']}:{row['Des_port']}",
            ]
            node_map = {nodes[i]: i for i in range(len(nodes))}

        # node features: choose a compact vector from numeric columns in row
        # pick numeric cols automatically
        numeric_cols = [
            c
            for c in df_slice.columns
            if df_slice[c].dtype != "object" and c not in ["class1", "class2", "class3"]
        ]
        # if no numeric cols, create simple degree-only features
        if len(numeric_cols) == 0:
            nf = np.ones((len(nodes), 4), dtype=np.float32)
        else:
            vals = row[numeric_cols].astype(float).fillna(0).values
            # repeat/trim to fixed node feature dim
            feat_dim = min(16, len(vals)) if len(vals) > 0 else 8
            nf = np.tile(vals[:feat_dim], (len(nodes), 1)).astype(np.float32)
            # pad if necessary
            if nf.shape[1] < feat_dim:
                pad = np.zeros((len(nodes), feat_dim - nf.shape[1]), dtype=np.float32)
                nf = np.concatenate([nf, pad], axis=1)

        # edges: src->dst and dst->src
        edges = np.array([[0, 1], [1, 0]]).T
        edges = torch.tensor(edges, dtype=torch.long, device=device)
        node_features_list.append(nf)
        edge_index_list.append(edges)
        edge_attr_list.append(
            None
        )  # placeholder; could add packet size/duration as edge_attr

    return node_features_list, edge_index_list, edge_attr_list


# -----------------------
# Strategy B: Pseudo-node partitioning (deterministic, works with scaled numpy arrays)
# -----------------------
def numpy_flat_to_pseudo_nodes(X, num_nodes=16):
    """
    Convert flat feature matrix X (N, F) -> pseudo-node tensor (N, num_nodes, node_dim)
    Pads deterministically if F % num_nodes != 0.
    Returns numpy array shape (N, num_nodes, node_dim)
    """
    N, F = X.shape
    # compute node_dim and padding
    node_dim = int(np.ceil(F / num_nodes))
    pad = num_nodes * node_dim - F
    if pad > 0:
        Xp = np.pad(X, ((0, 0), (0, pad)), mode="constant", constant_values=0)
    else:
        Xp = X
    X_nodes = Xp.reshape(N, num_nodes, node_dim).astype(np.float32)
    return X_nodes


def make_dataset_for_gnn_from_numpy(X_numpy, y_numpy, num_nodes=16, device=None):
    """
    Convert numpy features/labels into torch tensors suitable for the SimplifiedGNN_IDS / HybridAGNN_IDS
    - X_numpy: (N, F) scaled features from your pipeline
    - y_numpy: (N,) labels (encoded ints)
    Returns:
        x_tensor: torch.Tensor (N, num_nodes, node_dim)
        edge_index: torch.LongTensor [2, num_nodes*num_nodes] fully connected (same for all samples)
        y_tensor: torch.LongTensor (N,) or (N,1)
    """
    X_nodes = numpy_flat_to_pseudo_nodes(
        X_numpy, num_nodes=num_nodes
    )  # (N, num_nodes, node_dim)
    x_tensor = torch.from_numpy(X_nodes)
    # create fully connected edge_index once
    edge_index = make_fully_connected_edge_index(num_nodes, device=device)
    y_tensor = torch.tensor(
        y_numpy, dtype=torch.long if y_numpy.ndim == 1 else torch.float32
    )
    return x_tensor, edge_index, y_tensor


# -----------------------
# Small helper to create dataloaders (batching)
# -----------------------
from torch.utils.data import TensorDataset, DataLoader


def make_dataloaders_for_pseudo_nodes(x_tensor, y_tensor, batch_size=128, shuffle=True):
    """
    x_tensor: (N, num_nodes, node_dim)
    y_tensor: (N,) or (N, C)
    Returns PyTorch DataLoader
    """
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# -----------------------
# Example integration snippet (assumes you ran your pipeline and have X_train_bin_scaled etc.)
# -----------------------
def example_integration_with_pipeline(
    results_dict, classification="binary", num_nodes=16, device="cpu"
):
    """
    results_dict: the dict your pipeline returns (see commented return block in your pipeline).
    classification: 'binary' or 'multi'
    """
    if classification == "binary":
        X_train = results_dict["binary"]["X_train"]
        X_val = results_dict["binary"]["X_val"]
        X_test = results_dict["binary"]["X_test"]
        y_train = results_dict["binary"]["y_train"]
        y_val = results_dict["binary"]["y_val"]
        y_test = results_dict["binary"]["y_test"]
    else:
        X_train = results_dict["multi"]["X_train"]
        X_val = results_dict["multi"]["X_val"]
        X_test = results_dict["multi"]["X_test"]
        y_train = results_dict["multi"]["y_train"]
        y_val = results_dict["multi"]["y_val"]
        y_test = results_dict["multi"]["y_test"]

    dev = torch.device(device)
    x_tr, edge_index, y_tr = make_dataset_for_gnn_from_numpy(
        X_train, y_train, num_nodes=num_nodes, device=dev
    )
    x_val, _, y_valt = make_dataset_for_gnn_from_numpy(
        X_val, y_val, num_nodes=num_nodes, device=dev
    )
    x_test, _, y_testt = make_dataset_for_gnn_from_numpy(
        X_test, y_test, num_nodes=num_nodes, device=dev
    )

    tr_loader = make_dataloaders_for_pseudo_nodes(
        x_tr, y_tr, batch_size=128, shuffle=True
    )
    val_loader = make_dataloaders_for_pseudo_nodes(
        x_val, y_valt, batch_size=256, shuffle=False
    )
    test_loader = make_dataloaders_for_pseudo_nodes(
        x_test, y_testt, batch_size=256, shuffle=False
    )

    return tr_loader, val_loader, test_loader, edge_index.to(dev)


# if __name__ == "__main__":
#     filepath = "./X-IIoTID dataset.csv"
#     results = run_optimized_pipeline(filepath)

#     tr_loader, val_loader, test_loader, edge_index = example_integration_with_pipeline(
#         results, classification="binary", num_nodes=16, device="cuda"
#     )
