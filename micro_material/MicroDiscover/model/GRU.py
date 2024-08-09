from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Adam
from process_graph import graph_from_neo4j
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, negative_sampling
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np


class KnowledgeGraphDataset(Dataset):
    def __init__(self, nodes, attributes):
        self.nodes = nodes
        self.attributes = attributes

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        return self.nodes[idx], self.attributes[idx]


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, node_features):
        gru_out, _ = self.gru(node_features)
        node_embeddings = self.fc(gru_out)
        return node_embeddings


def compute_dot_product(microbe_vectors, material_vectors):

    if microbe_vectors.shape != material_vectors.shape:
        raise ValueError("error")

    dot_products = np.einsum('ij,ij->i', microbe_vectors, material_vectors)
    return dot_products


def normalize_scores(mrr_scores):
    min_score = min(mrr_scores)
    max_score = max(mrr_scores)

    normalized_scores = [(score - min_score) / (max_score - min_score) for score in mrr_scores]

    epsilon = 1e-10
    scaled_scores = [score * (1 - 2 * epsilon) + epsilon for score in normalized_scores]

    return scaled_scores


if __name__ == '__main__':
    NODE_FEATURE_DIM = 768
    HIDDEN_DIM = 64
    BATCH_SIZE = 48
    EPOCHS = 100
    LEARNING_RATE = 0.001
    OUTPUT_DIM = 768

    knowledge_graph = graph_from_neo4j()
    nodes_a = [n for n, d in knowledge_graph.nodes(data=True) if d["labels"][0] == "material"]
    node_ids = [node[0] for node in knowledge_graph.nodes(data=True)]
    id_mapping = {node_id: i for i, node_id in enumerate(node_ids)}
    nx.set_node_attributes(knowledge_graph, id_mapping, 'id')
    id_list = []
    name_list = []

    for node_a in nodes_a:

        ego_subgraph = nx.ego_graph(knowledge_graph, node_a, radius=2)

        for node_b in ego_subgraph.nodes():
            if knowledge_graph.nodes[node_b]["labels"][0] == "microorganism":

                name_a = knowledge_graph.nodes[node_a]["properties"]["name"]
                name_b = knowledge_graph.nodes[node_b]["properties"]["name"]

                id_list.append((node_a, node_b))
                name_list.append((name_a, name_b))

    material_microorganism_dict = {}
    material_microorganism_name_dict = {}
    for item in id_list:
        key, value = item
        if id_mapping[key] in material_microorganism_dict:
            material_microorganism_dict[id_mapping[key]].append(id_mapping[value])
        else:
            material_microorganism_dict[id_mapping[key]] = [id_mapping[value]]

    node_features_array = np.loadtxt('bert_all_features.txt', delimiter=',')
    ones_array = np.ones_like(node_features_array)
    node_features_tensor = torch.tensor(ones_array, dtype=torch.float)

    candidate_entities = []
    material_list = []
    for node, data in knowledge_graph.nodes(data=True):
        if data["properties"]["defineName"] == 'microorganism':
            mapping_id = id_mapping[node]
            candidate_entities.append(mapping_id)
    for node, data in knowledge_graph.nodes(data=True):
        if data["properties"]["defineName"] == 'material':
            mapping_id = id_mapping[node]
            material_list.append(mapping_id)
    candidate_mapping = {entity: idx for idx, entity in enumerate(candidate_entities, start=0)}
    model = GRU(NODE_FEATURE_DIM, HIDDEN_DIM, OUTPUT_DIM, num_layers=1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MarginRankingLoss(margin=1.0)
    for epoch in range(EPOCHS):
        model.train()
        out = model(node_features_tensor)
        MRR_list = []
        hits_20 = 0
        hits_10 = 0
        hits_5 = 0
        for material_item in material_list:
            material_feature = out[material_item].detach().numpy().reshape(1, -1)
            mrr_scores = []
            similarity_scores_dict = {}
            for microorganism_item in candidate_entities:
                microorganism_feature = out[microorganism_item].detach().numpy().reshape(1, -1)
                similarity_score = compute_dot_product(material_feature, microorganism_feature)
                mrr_scores.append(similarity_score.item())
            sorted_indices = np.argsort(mrr_scores)[::-1]
            if material_item in material_microorganism_dict:
                candidate_microorganism_list = material_microorganism_dict[material_item]
            else:
                continue
            min_rank = float('inf')
            for microorganism_item in candidate_microorganism_list:
                candidate_item = candidate_mapping[microorganism_item]
                rank = np.where(sorted_indices == candidate_item)[0][0] + 1
                min_rank = min(min_rank, rank)
            MRR_list.append(1 / min_rank)
            if min_rank <= 20:
                hits_20 += 1
            if min_rank <= 10:
                hits_10 += 1
            if min_rank <= 5:
                hits_5 += 1
        avg_mrr = sum(MRR_list) / len(MRR_list)
        mrr_tensor = -torch.tensor(avg_mrr, dtype=torch.float, requires_grad=True)
        print(avg_mrr)
        print("hits_5", hits_5 / len(material_list))
        print("hits_10", hits_10 / len(material_list))
        print("hits_20", hits_20 / len(material_list))
        optimizer.zero_grad()
        mrr_tensor.backward()
        optimizer.step()

    print("Training Complete")


