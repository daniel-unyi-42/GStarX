# -*- coding: utf-8 -*-
import torch
from torch.utils.data import random_split, Subset
from torch_geometric.data import DataLoader
from dig.xgraph.dataset import (
    MoleculeDataset,
    SynGraphDataset,
    SentiGraphDataset,
    BA_LRP,
)
from BA2MotifDataset import BA2MotifDataset
from BAMultiShapesDataset import BAMultiShapesDataset
from BenzeneDataset import BenzeneDataset
from torch_geometric.datasets import GNNBenchmarkDataset


import os
import torch
import pickle
import numpy as np
import os.path as osp
import networkx as nx
from torch_geometric.utils import dense_to_sparse, remove_self_loops, to_networkx
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.data.dataset import files_exist
import torch_geometric.transforms as T
import shutil


def get_dataset(dataset_root, dataset_name):
    if dataset_name.lower() in list(MoleculeDataset.names.keys()):
        return MoleculeDataset(root=dataset_root, name=dataset_name)
    elif dataset_name.lower() in ["graph_sst2", "graph_sst5", "twitter"]:
        return SentiGraphDataset(root=dataset_root, name=dataset_name)
    elif dataset_name.lower() in ["mnist"]:
        return GNNBenchmarkDataset(root=dataset_root, name=dataset_name)
    elif dataset_name.lower() in ["benzene"]:
        class AddTrueExplanation(object):
            def __call__(self, data):
                data.true = torch.any(data.true, dim=1).float()
                return data
        return BenzeneDataset(root=os.path.join(dataset_root, dataset_name), pre_transform=AddTrueExplanation())
    elif dataset_name.lower() in ["ba_2motifs"]:
        class RemoveSelfLoops(object):
            def __call__(self, data):
                data.edge_index, _ = remove_self_loops(data.edge_index)
                return data
        class AddTrueExplanation(object):
            def __call__(self, data):
                data.true = torch.zeros(data.num_nodes)
                data.true[-5:] = 1
                return data
        transform = T.Compose([RemoveSelfLoops(), AddTrueExplanation()])
        return BA2MotifDataset(root=os.path.join(dataset_root, dataset_name), transform=transform)
    elif dataset_name.lower() in ["ba_multishapes"]:
        class AddTrueExplanation(object):
            def __call__(self, data):
                def subgraph_matching(G, g):
                    gm = nx.algorithms.isomorphism.GraphMatcher(G, g)
                    return list(gm.mapping.keys()) if gm.subgraph_is_isomorphic() else []
                data.true = torch.zeros(data.num_nodes)
                G1 = to_networkx(data, to_undirected=True)
                g2 = nx.generators.small.house_graph()
                g3 = nx.generators.lattice.grid_2d_graph(3, 3)
                g4 = nx.generators.classic.wheel_graph(6)
                gm2 = subgraph_matching(G1, g2)
                gm3 = subgraph_matching(G1, g3)
                gm4 = subgraph_matching(G1, g4)
                data.true[gm2 + gm3 + gm4] = 1.0
                return data
        return BAMultiShapesDataset(root=os.path.join(dataset_root, dataset_name), transform=AddTrueExplanation())
    elif dataset_name.lower() in list(SynGraphDataset.names.keys()):
        class RemoveSelfLoops(object):
            def __call__(self, data):
                data.edge_index, _ = remove_self_loops(data.edge_index)
                return data
        return SynGraphDataset(root=dataset_root, name=dataset_name, transform=RemoveSelfLoops())
    elif dataset_name.lower() in ["ba_lrp"]:
        return BA_LRP(root=dataset_root)
    else:
        raise ValueError(f"{dataset_name} is not defined.")


def get_dataloader(
    dataset, batch_size, random_split_flag=True, data_split_ratio=None, seed=2
):
    """
    Args:
        dataset:
        batch_size: int
        random_split_flag: bool
        data_split_ratio: list, training, validation and testing ratio
        seed: random seed to split the dataset randomly
    Returns:
        a dictionary of training, validation, and testing dataLoader
    """

    if not random_split_flag and hasattr(dataset, "supplement"):
        assert "split_indices" in dataset.supplement.keys(), "split idx"
        split_indices = dataset.supplement["split_indices"]
        train_indices = torch.where(split_indices == 0)[0].numpy().tolist()
        dev_indices = torch.where(split_indices == 1)[0].numpy().tolist()
        test_indices = torch.where(split_indices == 2)[0].numpy().tolist()

        train = Subset(dataset, train_indices)
        eval = Subset(dataset, dev_indices)
        test = Subset(dataset, test_indices)
    else:
        num_train = int(data_split_ratio[0] * len(dataset))
        num_eval = int(data_split_ratio[1] * len(dataset))
        num_test = len(dataset) - num_train - num_eval

        train, eval, test = random_split(
            dataset,
            lengths=[num_train, num_eval, num_test],
            generator=torch.Generator().manual_seed(seed),
        )

    dataloader = dict()
    dataloader["train"] = DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloader["eval"] = DataLoader(eval, batch_size=batch_size, shuffle=False)
    dataloader["test"] = DataLoader(test, batch_size=batch_size, shuffle=False)
    return dataloader
